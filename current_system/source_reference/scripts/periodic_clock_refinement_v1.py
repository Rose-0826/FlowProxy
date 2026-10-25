#!/usr/bin/env python3
"""Small clock-only experiment. No new network training or CFD generation."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time

import h5py
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from geometry_attack_v2 import save, sha
from startup_observability_v1 import basis, frequency
from self_trajectory_closure_v1 import metrics
from round2_complete40_v1 import summarize

ORIGINAL = ROOT / 'outputs/corrected_benchmark_restart_v1'
OUT = ROOT / 'outputs/periodic_clock_refinement_v1'
DATA = ROOT / 'data/processed/dataset_corrected_benchmark_v1.h5'
TRAINVAL = ROOT / 'data/processed/dataset_corrected_trainval_v1.h5'
CACHE = ROOT / 'data/processed/corrected_benchmark_restart_v1_cases'
CANDIDATES = [
    {'id': 'mean10', 'center': 'shape_mean', 'radius': .10, 'identify': True},
    {'id': 'mean25', 'center': 'shape_mean', 'radius': .25, 'identify': True},
    {'id': 'rbf10', 'center': 'train80_RBF', 'radius': .10, 'identify': True},
    {'id': 'rbf_fixed', 'center': 'train80_RBF', 'radius': 0., 'identify': False},
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def vector(q):
    return np.array([q['Re'] / 200., q['cy'] / .8, q['D']], np.float64)


def q_from(g):
    return {key: str(g.attrs[key]) if key == 'shape' else float(g.attrs[key])
            for key in ['shape', 'Re', 'cx', 'cy', 'D']}


def summary(rows):
    return {h: summarize([r['horizons'][h] for r in rows])
            for h in ['2s', '8s', '20s', '40s']}


def build_priors(p):
    rows = []
    # No validation, development-test or confirmation labels enter the priors.
    with h5py.File(TRAINVAL, 'r') as h:
        for cid in p['splits']['train']:
            g = h['cases'][cid]
            rows.append({'id': cid, 'q': q_from(g), 'St': float(g.attrs['St'])})
    assert len(rows) == 80
    priors = {}
    audit = {}
    for shape in ['cylinder', 'square']:
        block = [r for r in rows if r['q']['shape'] == shape]
        X = np.stack([vector(r['q']) for r in block])
        y = np.array([r['St'] for r in block])
        priors[shape] = {'rbf': RBFInterpolator(X, y, kernel='thin_plate_spline', smoothing=1e-6),
                         'mean': float(y.mean())}
        loo = []
        for k in range(len(block)):
            use = np.arange(len(block)) != k
            reg = RBFInterpolator(X[use], y[use], kernel='thin_plate_spline', smoothing=1e-6)
            loo.append(float(reg(X[k:k+1])[0]))
        audit[shape] = {'min_St': float(y.min()), 'max_St': float(y.max()), 'mean_St': float(y.mean()),
                        'outside_old_10pct': int(np.sum((y < y.mean() * .9) | (y > y.mean() * 1.1))),
                        'LOO_RBF_absolute_error_mean': float(np.mean(abs(np.array(loo) - y))),
                        'LOO_shape_mean_absolute_error_mean': float(np.mean(abs(y - y.mean())))}
    save(OUT / 'training_priors.json', {'source': 'corrected train80 force-derived St only',
                                       'rows': rows, 'audit': audit})
    return priors


def extend(row, base, prior, candidate, override=None):
    N, H = 48, 4
    mask = row['mask'] > .5
    observed = np.concatenate([row['fields'][:4], base[:N]], axis=0).astype(np.float64)
    observed *= mask[None, None]
    t = row['times'][:4+N].astype(np.float64) - float(row['times'][3])
    model = prior[row['q']['shape']]
    St = model['mean'] if candidate['center'] == 'shape_mean' else float(model['rbf'](vector(row['q'])[None])[0])
    center = St / row['q']['D']
    assert center > 0 and np.isfinite(center)
    diag = {}
    if override is not None:
        f = override
        diag['uses_future_force_frequency_DIAGNOSTIC_ONLY'] = True
    elif candidate['identify']:
        f, diag = frequency(t, observed[:, 1][:, mask], H, center, candidate['radius'])
    else:
        f = center
    B = basis(t, f, H)
    coefficients = np.linalg.lstsq(B, observed.reshape(len(observed), -1), rcond=1e-10)[0]
    future = row['times'][4:164].astype(np.float64) - float(row['times'][3])
    out = (basis(future, f, H) @ coefficients).reshape(160, 3, 64, 64).astype(np.float32)
    out[:N] = base[:N]
    assert np.isfinite(out).all() and np.array_equal(out[:N], base[:N])
    return out, {'frequency': float(f), 'center_frequency': float(center),
                 'frequency_fit': diag, 'condition_number': float(np.linalg.cond(B))}


def run():
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    p = json.loads((ORIGINAL / 'preregister.json').read_text())
    lock = json.loads((ORIGINAL / 'selected_model_lock.json').read_text())
    assert sha(lock['chosen']['path']) == lock['chosen']['sha256']
    protected_paths = [ORIGINAL / 'selected_model_lock.json', ORIGINAL / 'preregister.json',
                       ORIGINAL / 'already_seen_test32/report.json', ORIGINAL / 'fresh_confirmation12/report.json',
                       Path(lock['chosen']['path'])]
    protected = {str(fp): sha(fp) for fp in protected_paths}
    save(OUT / 'preregister.json', {
        'registered_utc': now(), 'scope': 'small clock-only follow-up, university-competition budget',
        'network': lock['chosen'], 'original_protected_sha256': protected,
        'changed': 'frequency center or search radius only', 'candidates': CANDIDATES,
        'fixed': {'true_startup_frames': 4, 'self_bootstrap_steps': 48, 'H': 4, 'blend_steps': 0},
        'frequency_priors': 'corrected train80 St only; thin_plate_spline RBF smoothing1e-6, same features as prior experiments',
        'selection': 'val8 minimum 40s meanRi, then maximumRi, then prefer original mean10 when tied',
        'acceptance_before_development_rescore': 'selected meanRi <= 0.9 * original mean10, and maximumRi <= original mean10',
        'already_seen_cases': '32 development +12 previously scored confirmation; later scores are exploratory, not fresh blind confirmation',
        'oracle_diagnostic': 'target force St may be read only AFTER all candidate predictions are frozen; never a selectable candidate',
        'no_training_or_new_CFD': True, 'no_future_field_or_phase_for_prediction': True,
        'pressure_and_metrics': 'unchanged physical fields, fluid mask, no future pressure shift, same six-term Ri',
        'script_sha256': sha(__file__)})
    save(OUT / 'status.json', {'status': 'running', 'stage': 'train80_frequency_prior', 'started_utc': now()})
    priors = build_priors(p)

    # A reproducible forward pass creates validation trajectories; only startup fields enter predict().
    from corrected_benchmark_train_v1 import model_from, predict
    import torch
    torch.set_num_threads(1)
    model, _ = model_from(lock['chosen']['path'], p['config'])
    directory = OUT / 'validation'; directory.mkdir()
    candidates = {c['id']: [] for c in CANDIDATES}
    rows = []
    with h5py.File(TRAINVAL, 'r') as h:
        for cid in p['splits']['val']:
            g = h['cases'][cid]
            row = {'id': cid, 'q': q_from(g), 'fields': g['fields'][:4], 'geom': g['geom'][...],
                   'mask': g['mask'][...], 'times': g['times'][...]}
            base = predict(model, row)
            np.savez_compressed(directory / (cid + '_base.npz'), prediction=base)
            frozen = []
            for candidate in CANDIDATES:
                pred, diagnostic = extend(row, base, priors, candidate)
                path = directory / (candidate['id'] + '__' + cid + '.npz')
                np.savez_compressed(path, prediction=pred)
                frozen.append({'candidate': candidate['id'], 'path': str(path), 'sha256': sha(path), 'diagnostic': diagnostic})
            save(directory / (cid + '_predictions_frozen.json'), {'utc': now(), 'frozen_before_future_read': True, 'registry': frozen})
            # Future fields first read after freezing all four predictions for this validation case.
            truth = g['fields'][4:164]
            truth_St = float(g.attrs['St'])
            for item in frozen:
                pred = np.load(item['path'])['prediction']
                candidates[item['candidate']].append({'id': cid, 'q': row['q'], 'horizons': metrics(pred, truth, row['mask'], row['times']),
                    'clock': item['diagnostic'], 'truth_St_DIAGNOSTIC_ONLY': truth_St})
            oracle, diag = extend(row, base, priors, CANDIDATES[0], override=truth_St / row['q']['D'])
            rows.append({'id': cid, 'future_force_clock_DIAGNOSTIC_ONLY': metrics(oracle, truth, row['mask'], row['times'])})
            print('CLOCK_VALIDATION', cid, round(time.perf_counter()-started, 1), flush=True)
    summaries = {name: summary(block) for name, block in candidates.items()}
    locked_validation = json.loads((Path(lock['chosen']['path']).parent/'selected_validation.json').read_text())
    for horizon in ['2s', '8s', '20s', '40s']:
        frozen = locked_validation['summary']['closure'][horizon]
        rerun = summaries['mean10'][horizon]
        assert np.allclose(rerun['mean'], frozen['mean'], atol=1e-6, rtol=1e-5), (horizon, rerun, frozen)
        assert np.allclose(rerun['end'], frozen['end'], atol=1e-6, rtol=1e-5), (horizon, rerun, frozen)
        assert abs(rerun['mean_R'] - frozen['mean_R']) < 1e-4
    winner = min(CANDIDATES, key=lambda c: (summaries[c['id']]['40s']['mean_R'], summaries[c['id']]['40s']['max_R'], CANDIDATES.index(c)))
    original = summaries['mean10']['40s']; selected = summaries[winner['id']]['40s']
    accepted = selected['mean_R'] <= .9 * original['mean_R'] and selected['max_R'] <= original['max_R']
    save(OUT / 'validation_report.json', {'summary': summaries, 'cases': candidates, 'oracle_diagnostics': rows})
    save(OUT / 'selection.json', {'locked_utc': now(), 'candidate': winner, 'accepted': accepted,
                                 'original': original, 'selected': selected, 'selection_from_val8_only': True})
    print('CLOCK_SELECTED', winner['id'], 'ACCEPTED', accepted, 'R', selected['mean_R'], 'MAX', selected['max_R'], flush=True)
    del model
    torch.cuda.empty_cache()
    if accepted:
        for role in ['already_seen_test32', 'fresh_confirmation12']:
            folder = OUT / role; folder.mkdir()
            baseline = json.loads((ORIGINAL / role / 'report.json').read_text())
            scored = []; registry = []; startup_rows = []
            for existing in baseline['cases']['corrected_selected']:
                cid = existing['id']
                with np.load(CACHE / (cid + '_startup.npz')) as a:
                    row = {'id': cid, 'q': existing['q'], 'fields': a['fields'], 'mask': a['mask'], 'times': a['times']}
                base = np.load(ORIGINAL / role / ('corrected_selected__' + cid + '.npz'))['prediction']
                pred, diagnostic = extend(row, base, priors, winner)
                path = folder / (cid + '.npz'); np.savez_compressed(path, prediction=pred)
                registry.append({'id': cid, 'path': str(path), 'sha256': sha(path), 'clock': diagnostic})
                startup_rows.append(row)
            save(folder / 'predictions_frozen.json', {'utc': now(), 'selected_configuration_sha256': sha(OUT/'selection.json'),
                                                    'all_predictions_before_future_scoring': True, 'registry': registry})
            for row, entry in zip(startup_rows, registry):
                with np.load(CACHE / (row['id'] + '.npz')) as a:
                    truth = a['fields'][4:164]
                pred = np.load(entry['path'])['prediction']
                scored.append({'id': row['id'], 'q': row['q'], 'horizons': metrics(pred, truth, row['mask'], row['times']), 'clock': entry['clock']})
            old = {name: baseline['summary'][name] for name in ['corrected_selected', 'corrected_selected_closure',
                      'corrected_selected_cycle_replay', 'corrected_selected_self_DMD_rank9']}
            report = {'role': 'exploratory re-evaluation of already-seen cases', 'selected': winner,
                      'summary': summary(scored), 'cases': scored, 'frozen_original_baselines': old,
                      'no_new_blind_claim': True}
            save(folder/'report.json', report)
            print('CLOCK_EXPLORATORY', role, json.dumps(report['summary']['40s']), flush=True)
    unchanged = {path: sha(path)==value for path,value in protected.items()}
    assert all(unchanged.values())
    save(OUT/'status.json', {'status': 'complete', 'accepted': accepted, 'selected': winner['id'],
                           'original_files_unchanged': unchanged, 'seconds': time.perf_counter()-started})


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        if OUT.exists():
            save(OUT/'status.json', {'status': 'failed', 'error': repr(e), 'utc': now()})
        raise
