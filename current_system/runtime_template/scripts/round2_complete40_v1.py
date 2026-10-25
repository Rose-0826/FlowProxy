#!/usr/bin/env python3
"""Frozen F0/K8 supplemental evaluation. --preflight uses no PyTorch or GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/dataset_long40_v1.h5'
OUT = ROOT / 'outputs/cpu_round1/full40_eval'
MODELS = {
    'F0': ROOT / 'outputs/checkpoints/fno_long/best.pt',
    'K8': ROOT / 'outputs/checkpoints/fno_k8_base/base.pt',
}
TAUS = np.asarray([.05, .05, .08], dtype=np.float32)
NAMES = ('ux', 'uy', 'p')
TOL = 1e-4


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def horizon_selection(times, seconds):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 5 or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid clock')
    origin = float(times[3])
    future = times[4:]
    selected = np.flatnonzero(future <= origin + seconds + TOL)
    if not len(selected):
        raise ValueError('No future frame within horizon')
    endpoint = float(future[selected[-1]])
    return selected, abs(endpoint - (origin + seconds)) <= TOL


def frame_summary(err, origin, endpoint, requested, complete):
    mean, end = err.mean(axis=0), err[-1]
    decisions = [(float(values[c] / TAUS[c]), NAMES[c], name)
                 for values, name in [(mean, '整段平均'), (end, '末端')]
                 for c in range(3)]
    score, channel, which = max(decisions)
    return {'steps': len(err), 'seconds': float(endpoint - origin),
            'requested_seconds': requested, 't_end': float(endpoint),
            'complete': bool(complete), 'mean': mean.tolist(), 'end': end.tolist(),
            'max': err.max(axis=0).tolist(), 'R': score,
            'channel': channel, 'which': which}


def summarize(rows):
    return {'n': len(rows), 'steps': rows[0]['steps'],
            'seconds': rows[0]['seconds'], 'complete': all(r['complete'] for r in rows),
            'mean': np.mean([r['mean'] for r in rows], axis=0).tolist(),
            'end': np.mean([r['end'] for r in rows], axis=0).tolist(),
            'max_over_time_mean': np.mean([r['max'] for r in rows], axis=0).tolist(),
            'mean_R': float(np.mean([r['R'] for r in rows])),
            'max_R': float(max(r['R'] for r in rows)),
            'n_pass': sum(r['R'] < 1 for r in rows)}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def preflight(split):
    import h5py
    ids = json.loads((ROOT / 'configs/splits/body_120.json').read_text())[split]
    reference_path = ROOT / f'outputs/round2/horizon_{split}.json'
    reference = json.loads(reference_path.read_text())
    audit = json.loads((ROOT / 'outputs/cpu_round1/data_audit.json').read_text())
    expected = {DATA: audit['view_h5_sha256'],
                DATA.with_name('dataset_long40_tail_v1.h5'): audit['tail_h5_sha256'],
                DATA.with_name('dataset_long.h5'): audit['source_h5_sha256']}
    checked = []
    for path, digest in expected.items():
        if sha_file(path) != digest:
            raise ValueError(f'Data hash differs: {path.name}')
    with h5py.File(DATA, 'r') as handle:
        stored_ids = [v.decode() for v in handle['splits'][split][...]]
        if stored_ids != ids:
            raise ValueError('Split order differs')
        for cid in ids:
            group = handle['cases'][cid]
            times = group['times'][...].astype(np.float64)
            future, complete = horizon_selection(times, 40.)
            if not complete or not np.array_equal(future, np.arange(160)):
                raise ValueError(f'Full 40-second endpoint absent: {cid}')
            fields = group['fields'][:164]
            if fields.shape != (164, 3, 64, 64) or not np.isfinite(fields).all():
                raise ValueError(f'Invalid virtual data: {cid}')
            if not np.isfinite(group['geom'][...]).all():
                raise ValueError('Invalid geometry')
            mask = group['mask'][...]
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError('Mask differs from frozen binary mask')
            checked.append({'id': cid, 'origin': float(times[3]),
                            'endpoint': float(times[163]), 'steps': 160})
    return {'status': 'preflight_passed_no_model_evaluation', 'split': split,
            'data': str(DATA.relative_to(ROOT)), 'data_sha256': expected[DATA],
            'source_hashes': {p.name: v for p, v in expected.items()},
            'reference_report': str(reference_path.relative_to(ROOT)),
            'reference_report_sha256': sha_file(reference_path),
            'models': {name: {'checkpoint': str(path.relative_to(ROOT)),
                             'checkpoint_file_sha256': sha_file(path),
                             'expected_state_sha256': reference['models'][name]['sha256']}
                       for name, path in MODELS.items()},
            'script_sha256': sha_file(__file__), 'cases': checked,
            'future_truth_used_only_for_scoring': True,
            'pressure_reference': 'raw HDF5; no future-mean subtraction',
            'metric': 'fluid mask, separate channel L2, frame mean then case mean',
            'no_temporal_resampling': True,
            'legacy_157_prefix_checked_before_supplemental_report': True,
            'legacy_per_channel_absolute_tolerance': 1e-5}


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--time', action='store_true')
    args = parser.parse_args()
    note = preflight(args.split)
    if args.preflight:
        write_new(OUT / f'preflight_{args.split}.json', note)
        print(json.dumps({'status': note['status'], 'split': args.split,
                          'cases': len(note['cases']), 'steps': 160}))
        return
    import h5py
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for inference; --preflight is the CPU entry')
    sys.path.insert(0, str(ROOT / 'src'))
    from flowproxy.models import build_model
    from flowproxy.paths import load_config
    from flowproxy.round2.metrics import per_frame_rel
    dest = OUT / f'complete40_{args.split}.json'
    if dest.exists():
        raise FileExistsError('Refusing to overwrite a completed evaluation')
    write_new(OUT / f'execution_protocol_{args.split}.json', note)
    cfg = load_config(ROOT / 'configs/default.yaml')
    old = json.loads((ROOT / note['reference_report']).read_text())
    device = torch.device('cuda')
    report = {'protocol': note, 'models': {}}
    for name, path in MODELS.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        actual_sha = state_sha(saved['model'])
        if actual_sha != note['models'][name]['expected_state_sha256']:
            raise ValueError(f'Frozen model differs: {name}')
        model = build_model(saved['model_name'], saved['in_channels'], saved['out_channels'], cfg).to(device)
        model.load_state_dict(saved['model'])
        model.eval()

        @torch.inference_mode()
        def predict(geom, initial, steps):
            window = deque(initial.unbind(0), maxlen=4)
            predictions = []
            for _ in range(steps):
                hist = torch.stack(tuple(window)).reshape(-1, 64, 64)
                pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
                predictions.append(pred)
                window.append(pred)
            return torch.stack(predictions)

        cases = []
        with h5py.File(DATA, 'r') as handle:
            for item in note['cases']:
                cid = item['id']
                group = handle['cases'][cid]
                times = group['times'][...].astype(np.float64)
                geom = torch.from_numpy(group['geom'][...].astype(np.float32))[None].to(device)
                fields = torch.from_numpy(group['fields'][:164].astype(np.float32)).to(device)
                mask = torch.from_numpy(group['mask'][...].astype(np.float32)).to(device)
                with torch.inference_mode():
                    pred = predict(geom, fields[:4], 160)
                    err = per_frame_rel(pred, fields[4:164], mask).cpu().numpy()
                horizons = {}
                for seconds in (2., 8., 20., 40.):
                    selected, complete = horizon_selection(times, seconds)
                    horizons[f'{seconds:g}s'] = frame_summary(err[selected], times[3],
                        times[4 + selected[-1]], seconds, complete)
                legacy = frame_summary(err[:157], times[3], times[160], None, True)
                old_cases = old['models'][name]['horizons']['available']['cases']
                old_case = next(row for row in old_cases if row['id'] == cid)
                differences = [abs(a-b) for key in ('mean', 'end', 'max')
                               for a, b in zip(legacy[key], old_case[key])]
                if max(differences) > 1e-5:
                    raise ValueError(f'Legacy prefix no longer reproduces: {name} {cid}')
                horizons['legacy157'] = legacy
                cases.append({'id': cid, 'origin': float(times[3]), 'horizons': horizons,
                              'curve_times': times[4:164].tolist(), 'curve': err.tolist(),
                              'legacy_prefix_max_absolute_delta': max(differences)})
                print(name, cid, '40s R', f"{horizons['40s']['R']:.4f}", flush=True)
        aggregate = {}
        for horizon in cases[0]['horizons']:
            rows = [{'id': case['id'], **case['horizons'][horizon]} for case in cases]
            aggregate[horizon] = {'summary': summarize(rows), 'cases': rows}
        block = {'state_sha256': actual_sha, 'horizons': aggregate, 'curves': cases}
        if args.time:
            with h5py.File(DATA, 'r') as handle:
                group = handle['cases'][note['cases'][0]['id']]
                geom = torch.from_numpy(group['geom'][...].astype(np.float32))[None].to(device)
                initial = torch.from_numpy(group['fields'][:4].astype(np.float32)).to(device)
            timings = {}
            for steps in (32, 157, 160):
                for _ in range(3):
                    predict(geom, initial, steps)
                samples = []
                for _ in range(20):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    predict(geom, initial, steps)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - start)
                timings[str(steps)] = {'median': float(np.median(samples)), 'samples': samples}
            block['timing'] = {'case': note['cases'][0]['id'], 'seconds_by_steps': timings,
                               'includes': 'GPU inference and synchronization; no disk/H2D/scoring',
                               'inference_mode': True, 'gpu': torch.cuda.get_device_name(device)}
        report['models'][name] = block
        del model, saved, fields, pred, mask, geom
        torch.cuda.empty_cache()
    write_new(dest, report)
    print('wrote', str(dest))


if __name__ == '__main__':
    main()
