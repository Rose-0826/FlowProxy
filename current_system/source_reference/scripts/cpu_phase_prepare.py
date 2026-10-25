#!/usr/bin/env python3
"""Canonical startup snapshots and CPU transfer baselines; no test-set scoring.

Target phase is used only to define offline reference labels. Prediction sees
geometry metadata and the training donor bank, not target fields or forces.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / 'data/processed/dataset_long.h5'
OUT = ROOT / 'outputs/cpu_round1/phase0'
CHANNELS = ['Ux', 'Uy', 'p']


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def relative(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> list[float]:
    fluid = mask > 0.5
    return [float(np.linalg.norm((pred[c] - truth[c])[fluid]) /
                  max(float(np.linalg.norm(truth[c][fluid])), 1e-8)) for c in range(3)]


def interpolated_mean(t: np.ndarray, signal: np.ndarray, start: float, end: float) -> float:
    inside = (t > start) & (t < end)
    stamps = np.r_[start, t[inside], end]
    values = np.r_[np.interp(start, t, signal), signal[inside], np.interp(end, t, signal)]
    return float(np.sum(.5 * (values[1:] + values[:-1]) * np.diff(stamps)) / (end - start))


def rising_crossings(t: np.ndarray, signal: np.ndarray, level: float) -> np.ndarray:
    pick = np.flatnonzero((signal[:-1] <= level) & (signal[1:] > level))
    return t[pick] + (level - signal[pick]) * (t[pick + 1] - t[pick]) / (signal[pick + 1] - signal[pick])


def field_at(group, instant: float) -> np.ndarray:
    times = group['times'][...].astype(np.float64)
    if not times[0] <= instant <= times[-1]:
        raise ValueError('Canonical instant is outside saved stable fields')
    right = int(np.searchsorted(times, instant, side='left'))
    if right == 0 or times[right] == instant:
        return group['fields'][right].astype(np.float64)
    left = right - 1
    weight = (instant - times[left]) / (times[right] - times[left])
    return (1 - weight) * group['fields'][left].astype(np.float64) + weight * group['fields'][right].astype(np.float64)


def canonical(group) -> tuple[np.ndarray, np.ndarray, dict]:
    force = group['forces'][...].astype(np.float64)
    keep = (force[:, 0] >= 160.0) & (force[:, 0] <= 200.0)
    t, cl = force[keep, 0], force[keep, 2]
    if len(t) < 20 or np.any(np.diff(t) <= 0):
        raise ValueError('Insufficient or unordered stable force samples')
    level = interpolated_mean(t, cl, t[0], t[-1])
    levels = [level]
    for _ in range(3):
        crossings = rising_crossings(t, cl, level)
        if len(crossings) < 3:
            raise ValueError('Fewer than three upward crossings')
        level = interpolated_mean(t, cl, crossings[-3], crossings[-1])
        levels.append(level)
    crossings = rising_crossings(t, cl, level)
    if len(crossings) < 3:
        raise ValueError('Canonical complete cycles unavailable')
    periods = np.diff(crossings[-3:])
    diameter = float(group.attrs['D'])
    if np.any(periods < 2 * diameter) or np.any(periods > 20 * diameter):
        raise ValueError('Cycle period outside preregistered resolved range')
    first, second = [field_at(group, stamp) for stamp in crossings[-2:]]
    snapshot = (.5 * (first + second)).astype(np.float32)
    mask = group['mask'][...].astype(np.float32)
    noise = relative(first, second, mask)
    note = {'crossing_times': crossings[-3:].tolist(), 'periods_seconds': periods.tolist(),
            'phase_zero_level': level, 'mean_refinement_levels': levels,
            'cycle_repeatability_relative_ux_uy_p': noise,
            'canonical_snapshot': 'mean of last two upward-crossing fields',
            'snapshot_finite': bool(np.isfinite(snapshot).all())}
    return snapshot, mask, note


def metadata(cid: str, group) -> dict:
    return {'id': cid, 'shape': str(group.attrs['shape']), 'Re': float(group.attrs['Re']),
            'cy': float(group.attrs['cy']), 'D': float(group.attrs['D']), 'cx': float(group.attrs['cx'])}


def same_geometry(a: dict, b: dict) -> bool:
    return all(a[key] == b[key] for key in ['shape', 'cx', 'cy', 'D'])


def predict(q: dict, bank: list[dict], canonical_fields: np.ndarray,
            raw180: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    # This function never receives target reference fields, forces or phases.
    pool = [i for i, item in enumerate(bank) if item['id'] != q['id'] and same_geometry(item, q)]
    if not pool:
        raise ValueError('No same-geometry training donor; query outside stage-one scope')
    nearest = min(pool, key=lambda i: (abs(bank[i]['Re'] - q['Re']), bank[i]['id']))
    ordered = sorted(pool, key=lambda i: (bank[i]['Re'], bank[i]['id']))
    lower = [i for i in ordered if bank[i]['Re'] <= q['Re']]
    upper = [i for i in ordered if bank[i]['Re'] >= q['Re']]
    if len(ordered) < 2:
        pair, weights, kind = [nearest], [1.0], 'single_donor_fallback'
    elif lower and upper:
        lo, hi = lower[-1], upper[0]
        if lo == hi:
            pair, weights, kind = [lo], [1.0], 'exact_parameter'
        else:
            weight = (q['Re'] - bank[lo]['Re']) / (bank[hi]['Re'] - bank[lo]['Re'])
            pair, weights, kind = [lo, hi], [1 - weight, weight], 'interpolation'
    else:
        pair = ordered[:2] if not lower else ordered[-2:]
        lo, hi = pair
        weight = (q['Re'] - bank[lo]['Re']) / (bank[hi]['Re'] - bank[lo]['Re'])
        weights, kind = [1 - weight, weight], 'extrapolation'
    shape_pool = [i for i, item in enumerate(bank) if item['id'] != q['id'] and item['shape'] == q['shape']]
    z = np.asarray([[bank[i][k] for k in ['Re', 'cy', 'D']] for i in range(len(bank))])
    sigma = np.maximum(z.std(axis=0), 1e-8)
    query = np.asarray([q[k] for k in ['Re', 'cy', 'D']])
    global_nn = min(shape_pool, key=lambda i: (float(np.square((z[i] - query) / sigma).sum()), bank[i]['id']))
    linear = sum(weight * canonical_fields[i].astype(np.float64) for i, weight in zip(pair, weights)).astype(np.float32)
    preds = {'nearest_180': raw180[nearest], 'phase_nearest_same_geometry': canonical_fields[nearest],
             'phase_nearest_global_z': canonical_fields[global_nn], 'phase_linear': linear}
    note = {'same_geometry_nearest': bank[nearest]['id'], 'global_z_nearest': bank[global_nn]['id'],
            'linear_donors': [bank[i]['id'] for i in pair], 'linear_weights': weights,
            'linear_kind': kind, 'target_inputs': ['shape', 'Re', 'cy', 'D', 'cx'],
            'donor_input_split': 'training_only', 'target_flow_input': False}
    return preds, note


def aggregate(rows: list[dict]) -> dict:
    methods = list(rows[0]['errors'])
    return {name: {'mean_ux_uy_p': np.mean([r['errors'][name] for r in rows], axis=0).tolist(),
                   'max_ux_uy_p': np.max([r['errors'][name] for r in rows], axis=0).tolist()}
            for name in methods}


def main() -> None:
    protocol = {
        'protocol_id': 'phase0_transfer_cpu_v1', 'training_ids': 'original_train_80',
        'validation_ids': 'original_val_8_development_only', 'test_32_accessed': False,
        'confirmation_12_accessed': False, 'window': [160.0, 200.0],
        'phase_zero': 'upward crossing of refined cycle-mean Cl', 'mean_refinement_iterations': 3,
        'canonical_label': 'average snapshots at last two upward crossings, linear time interpolation',
        'resolved_period_range_nd': [2.0, 20.0], 'pressure_reference': 'raw HDF5, unchanged',
        'metric': 'per-channel fluid-mask relative L2 of phase-zero initial snapshot, not forecast score',
        'metric_denominator_floor': 1e-8, 'donors': 'training-only, same geometry for residual task',
        'nearest_tie': 'lexicographic case id', 'linear': 'two bracketing Re or nearest two at outer edge',
        'global_nearest_additional_strong_baseline': 'same shape, z(Re,cy,D) with training-population std',
        'cv': '4 folds hold out all cases of one Re in [80,100,150,200]',
        'label_gate_p90_limits_ux_uy_p': [.05, .05, .08],
        'model_gate': 'no model fitting if training label repeatability gate fails',
        'prediction_target_inputs': ['shape', 'Re', 'cy', 'D', 'cx'], 'threads': 1,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = OUT / 'preregister.json'
    if prereg.exists():
        raise ValueError('Refusing to overwrite phase-zero experiment')
    save_json(prereg, protocol)
    started = time.perf_counter()
    records, data = [], {}
    with h5py.File(H5, 'r') as h:
        for role in ['train', 'val']:
            ids = [item.decode() for item in h['splits'][role][...]]
            snapshots, masks, raw = [], [], []
            meta = []
            for cid in ids:
                g = h['cases'][cid]
                snapshot, mask, note = canonical(g)
                if not note['snapshot_finite']:
                    raise ValueError('Non-finite canonical snapshot')
                meta.append(metadata(cid, g))
                snapshots.append(snapshot)
                masks.append(mask)
                raw.append(field_at(g, 180.0).astype(np.float32))
                records.append({'id': cid, 'split': role, **note})
            data[role] = (meta, np.stack(snapshots), np.stack(masks), np.stack(raw))
            filename = 'train_bank.npz' if role == 'train' else 'val_reference_only.npz'
            np.savez(OUT / filename, ids=np.asarray(ids), fields=data[role][1], mask=data[role][2], raw180=data[role][3])
            save_json(OUT / (role + '_geometry_metadata.json'), meta)
            print('LABELS', role, len(ids), flush=True)
    train_notes = [r for r in records if r['split'] == 'train']
    noises = np.asarray([r['cycle_repeatability_relative_ux_uy_p'] for r in train_notes])
    q90 = np.quantile(noises, .9, axis=0)
    label_gate = bool(np.all(q90 <= np.asarray(protocol['label_gate_p90_limits_ux_uy_p'])))
    save_json(OUT / 'label_repeatability.json', {'records': records,
              'training_median_ux_uy_p': np.median(noises, axis=0).tolist(),
              'training_q90_ux_uy_p': q90.tolist(), 'label_gate_passed': label_gate})
    meta, fields, masks, raw = data['train']
    cv_rows = []
    for held_re in [80, 100, 150, 200]:
        training = [i for i, q in enumerate(meta) if q['Re'] != held_re]
        query_indices = [i for i, q in enumerate(meta) if q['Re'] == held_re]
        bank = [meta[i] for i in training]
        for index in query_indices:
            preds, note = predict(meta[index], bank, fields[training], raw[training])
            cv_rows.append({'id': meta[index]['id'], 'held_Re': held_re, **note,
                            'errors': {name: relative(pred, fields[index], masks[index])
                                       for name, pred in preds.items()}})
    val_meta = json.loads((OUT / 'val_geometry_metadata.json').read_text())
    predictions, notes = [], []
    for query in val_meta:
        preds, note = predict(query, meta, fields, raw)
        predictions.append(preds)
        notes.append(note)
    names = list(predictions[0])
    np.savez(OUT / 'val_baseline_predictions.npz',
             **{name: np.stack([r[name] for r in predictions]) for name in names})
    # Reference scoring begins only after all eight predictions are saved.
    reference = np.load(OUT / 'val_reference_only.npz')
    val_rows = [{'id': q['id'], **notes[i],
                 'errors': {name: relative(predictions[i][name], reference['fields'][i], reference['mask'][i])
                            for name in names}} for i, q in enumerate(val_meta)]
    report = {'protocol_id': protocol['protocol_id'], 'task': 'offline_periodic_initial_snapshot_shape',
              'label_gate_passed': label_gate, 'cv_train80': {'aggregate': aggregate(cv_rows), 'cases': cv_rows},
              'val8_development': {'aggregate': aggregate(val_rows), 'cases': val_rows},
              'all_predictions_saved_before_val_scoring': True, 'model_trained': False,
              'cfd_performance_tested': False, 'elapsed_seconds': time.perf_counter() - started,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save_json(OUT / 'baseline_report.json', report)
    print(json.dumps({'label_gate_passed': label_gate, 'q90': q90.tolist(),
                      'cv': report['cv_train80']['aggregate'],
                      'val': report['val8_development']['aggregate'],
                      'elapsed': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
