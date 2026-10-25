#!/usr/bin/env python3
"""Bounded NumPy-only screening of donor-conditioned residual models.

All selection uses the original training 80. Validation is a development check;
neither the old test 32 nor the confirmation 12 is read. No CFD is launched.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cpu_phase_prepare import aggregate, predict as baseline_predict, relative, same_geometry, save_json

BASE = ROOT / 'outputs/cpu_round1/phase0'
OUT = ROOT / 'outputs/cpu_round1/residual'


class MLP:
    def __init__(self, input_dim: int, output_dim: int, seed: int, hidden: int = 64):
        rng = np.random.default_rng(seed)
        self.p = {'W1': rng.normal(0, np.sqrt(2 / (input_dim + hidden)), (input_dim, hidden)),
                  'b1': np.zeros(hidden),
                  'W2': rng.normal(0, np.sqrt(1 / hidden), (hidden, hidden)),
                  'b2': np.zeros(hidden), 'W3': np.zeros((hidden, output_dim)),
                  'b3': np.zeros(output_dim)}

    def forward(self, x):
        h1 = np.tanh(x @ self.p['W1'] + self.p['b1'])
        h2 = np.tanh(h1 @ self.p['W2'] + self.p['b2'])
        return h2 @ self.p['W3'] + self.p['b3']

    def loss_grad(self, x, y):
        p = self.p
        h1 = np.tanh(x @ p['W1'] + p['b1'])
        h2 = np.tanh(h1 @ p['W2'] + p['b2'])
        error = h2 @ p['W3'] + p['b3'] - y
        d3 = 2 * error / error.size
        d2 = (d3 @ p['W3'].T) * (1 - h2 ** 2)
        d1 = (d2 @ p['W2'].T) * (1 - h1 ** 2)
        return float(np.mean(error ** 2)), {'W3': h2.T @ d3, 'b3': d3.sum(axis=0),
                 'W2': h1.T @ d2, 'b2': d2.sum(axis=0),
                 'W1': x.T @ d1, 'b1': d1.sum(axis=0)}

    def fit(self, x, y, seed: int, steps: int = 2000) -> dict:
        rng = np.random.default_rng(seed + 10000)
        m = {key: np.zeros_like(value) for key, value in self.p.items()}
        v = {key: np.zeros_like(value) for key, value in self.p.items()}
        hits, losses = 0, []
        for step in range(1, steps + 1):
            index = rng.integers(0, len(x), size=16)
            loss, grad = self.loss_grad(x[index], y[index])
            norm = np.sqrt(sum(float(np.square(g).sum()) for g in grad.values()))
            scale = min(1.0, 1.0 / max(norm, 1e-12))
            hits += int(scale < 1)
            for key, value in self.p.items():
                g = grad[key] * scale
                m[key] = .9 * m[key] + .1 * g
                v[key] = .999 * v[key] + .001 * g * g
                update = (m[key] / (1 - .9 ** step)) / (np.sqrt(v[key] / (1 - .999 ** step)) + 1e-8)
                if key.startswith('W'):
                    value *= 1 - 1e-3 * 1e-4
                value -= 1e-3 * update
            if step in [250, 500, 1000, 1500, 2000]:
                losses.append({'step': step, 'training_loss': float(np.mean((self.forward(x) - y) ** 2))})
        return {'fixed_steps': steps, 'clip_hit_fraction': hits / steps, 'training_losses': losses}


def gradient_check() -> dict:
    rng = np.random.default_rng(123)
    model = MLP(7, 3, 123, hidden=6)
    model.p['W3'][:] = rng.normal(0, .2, model.p['W3'].shape)
    x, y = rng.normal(size=(5, 7)), rng.normal(size=(5, 3))
    _, grad = model.loss_grad(x, y)
    errors = []
    for _ in range(3):
        direction = {key: rng.normal(size=value.shape) for key, value in model.p.items()}
        norm = np.sqrt(sum(float(np.square(v).sum()) for v in direction.values()))
        direction = {key: value / norm for key, value in direction.items()}
        analytic = sum(float((grad[key] * value).sum()) for key, value in direction.items())
        eps = 1e-5
        for key in model.p:
            model.p[key] += eps * direction[key]
        plus = np.mean((model.forward(x) - y) ** 2)
        for key in model.p:
            model.p[key] -= 2 * eps * direction[key]
        minus = np.mean((model.forward(x) - y) ** 2)
        for key in model.p:
            model.p[key] += eps * direction[key]
        numerical = (plus - minus) / (2 * eps)
        errors.append(float(abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)))
    if max(errors) > 1e-6:
        raise ValueError(f'MLP gradient check failed: {errors}')
    return {'relative_directional_errors': errors, 'passed': True}


def modes(x: np.ndarray, count: int = 32) -> tuple[np.ndarray, float]:
    gram = x @ x.T
    eig, vectors = np.linalg.eigh(gram)
    order = np.argsort(eig)[::-1]
    order = [i for i in order[:count] if eig[i] > max(float(eig.max()), 1.0) * 1e-12]
    basis = (vectors[:, order].T @ x) / np.sqrt(eig[order])[:, None]
    captured = float(np.maximum(eig[order], 0).sum() / max(float(np.maximum(eig, 0).sum()), 1e-12))
    return basis, captured


def params(meta: dict) -> np.ndarray:
    return np.asarray([float(meta['shape'] == 'square'), meta['Re'] / 200, meta['cy'], meta['D']], dtype=np.float64)


class Features:
    def __init__(self, meta: list[dict], fields: np.ndarray, masks: np.ndarray):
        self.meta, self.fields = meta, fields
        count = masks.sum()
        self.scales = np.sqrt(np.sum(fields.astype(np.float64) ** 2 * masks[:, None], axis=(0, 2, 3)) / count + 1e-12)
        self.factor = np.repeat(self.scales, 64 * 64)
        normalized = fields.reshape(len(fields), -1).astype(np.float64) / self.factor
        self.source_mean = normalized.mean(axis=0)
        self.source_basis, self.source_energy = modes(normalized - self.source_mean)
        self.source_coeff = (normalized - self.source_mean) @ self.source_basis.T
        pairs = [(i, j) for i, q in enumerate(meta) for j, d in enumerate(meta)
                 if i != j and same_geometry(q, d)]
        if len(pairs) < 10:
            raise ValueError('Insufficient same-geometry training pairs')
        residual = np.stack([normalized[i] - normalized[j] for i, j in pairs])
        self.residual_basis, self.residual_energy = modes(residual)
        self.pairs = pairs
        x = np.stack([self.raw_features(meta[i], j) for i, j in pairs])
        self.xmean, self.xstd = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        self.x = (x - self.xmean) / self.xstd
        delta = np.asarray([(meta[i]['Re'] - meta[j]['Re']) / 100 for i, j in pairs])
        projected = residual @ self.residual_basis.T
        secant = projected / delta[:, None]
        self.yscale = max(float(np.sqrt(np.mean(secant ** 2))), 1e-8)
        self.y = secant / self.yscale

    def raw_features(self, q: dict, donor: int) -> np.ndarray:
        a, b = params(q), params(self.meta[donor])
        return np.r_[a, b, a - b, self.source_coeff[donor]]

    def query(self, q: dict):
        candidates = [i for i, d in enumerate(self.meta) if d['id'] != q['id'] and same_geometry(q, d)]
        donor = min(candidates, key=lambda i: (abs(self.meta[i]['Re'] - q['Re']), self.meta[i]['id']))
        x = (self.raw_features(q, donor) - self.xmean) / self.xstd
        delta = (q['Re'] - self.meta[donor]['Re']) / 100
        return x, donor, delta

    def reconstruct(self, coefficient: np.ndarray, donor: int, delta: float) -> np.ndarray:
        correction = (coefficient * self.yscale) @ self.residual_basis
        return (self.fields[donor].astype(np.float64) +
                (delta * correction * self.factor).reshape(3, 64, 64)).astype(np.float32)


def kernel(a, b, width):
    distance = np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None]
                          - 2 * a @ b.T, 0.0) / a.shape[1]
    return np.exp(-.5 * distance / (width * width))


def fit_ridge(features: Features, alpha: float):
    mean = features.y.mean(axis=0)
    weight = np.linalg.solve(features.x.T @ features.x + len(features.x) * alpha * np.eye(features.x.shape[1]),
                             features.x.T @ (features.y - mean))
    return lambda x: x @ weight + mean


def fit_rbf(features: Features, width: float, alpha: float):
    mean = features.y.mean(axis=0)
    k = kernel(features.x, features.x, width)
    weight = np.linalg.solve(k + alpha * np.eye(len(k)), features.y - mean)
    return lambda x: kernel(x, features.x, width) @ weight + mean


def evaluate(features: Features, queries: list[dict], truth: np.ndarray, masks: np.ndarray,
             regressors: dict) -> list[dict]:
    x, donors, delta = zip(*(features.query(q) for q in queries))
    x = np.stack(x)
    rows = [{'id': q['id'], 'donor': features.meta[donors[i]]['id'], 'errors': {}}
            for i, q in enumerate(queries)]
    for name, regressor in regressors.items():
        coefficient = regressor(x)
        for i in range(len(queries)):
            pred = features.reconstruct(coefficient[i], donors[i], delta[i])
            rows[i]['errors'][name] = relative(pred, truth[i], masks[i])
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'protocol_id': 'cpu_residual_screen_v1', 'training_source': 'phase0/train_bank.npz',
        'baseline_family': 'donor-conditioned Re secant residual POD',
        'identity_constraint': 'correction multiplied by (Re_target-Re_donor)/100',
        'source_pod_max_modes': 32, 'residual_pod_max_modes': 32,
        'folds_hold_out_entire_Re': [80, 100, 150, 200], 'fold_preprocessing_fit_on_training_only': True,
        'ridge_alpha_grid': [1e-4, 1e-2, 1.0], 'rbf_width_grid': [.5, 1.0, 2.0],
        'rbf_alpha_grid': [1e-6, 1e-3, 1e-1],
        'regression_hyperparameters_selected_on_train80_cv_only': True,
        'cv_role': 'development and hyperparameter selection, not independent final confirmation',
        'mlp': {'hidden': [64, 64], 'activation': 'tanh', 'seeds': [11, 22, 33],
                'last_layer_zero_initialized': True, 'steps': 2000, 'batch': 16,
                'learning_rate': 1e-3, 'optimizer': 'AdamW', 'weight_decay_weights_only': 1e-4,
                'gradient_clip': 1.0, 'loss': 'MSE of training-scaled secant POD coefficients'},
        'selection': 'fixed final step, no validation checkpoint selection',
        'cv_gate': 'at least 2/3 seeds reduce mean Uy by >=15% vs strongest non-MLP comparator',
        'val_gate': 'CV median seed improves Uy in >=6/8; mean Ux,p <=1.05 times CV-selected comparator',
        'test32_accessed': False, 'confirmation12_accessed': False, 'cfd_launched': False,
        'threads': 1}
    if (OUT / 'preregister.json').exists():
        raise ValueError('Refusing to overwrite existing residual screening')
    save_json(OUT / 'preregister.json', protocol)
    check = gradient_check()
    save_json(OUT / 'gradient_check.json', check)
    label = json.loads((BASE / 'label_repeatability.json').read_text())
    if not label['label_gate_passed']:
        save_json(OUT / 'model_report.json', {'status': 'stopped_label_gate', 'models_trained': False})
        return
    meta = json.loads((BASE / 'train_geometry_metadata.json').read_text())
    bank = np.load(BASE / 'train_bank.npz')
    fields, masks = bank['fields'], bank['mask']
    baseline = json.loads((BASE / 'baseline_report.json').read_text())
    cv_rows, fit_notes, started = [], [], time.perf_counter()
    for held_re in [80, 100, 150, 200]:
        train = [i for i, q in enumerate(meta) if q['Re'] != held_re]
        held = [i for i, q in enumerate(meta) if q['Re'] == held_re]
        f = Features([meta[i] for i in train], fields[train], masks[train])
        models = {}
        for alpha in protocol['ridge_alpha_grid']:
            models[f'ridge_a{alpha:g}'] = fit_ridge(f, alpha)
        for width in protocol['rbf_width_grid']:
            for alpha in protocol['rbf_alpha_grid']:
                models[f'rbf_w{width:g}_a{alpha:g}'] = fit_rbf(f, width, alpha)
        notes = {'held_Re': held_re, 'train_cases': len(train), 'pairs': len(f.pairs),
                 'source_modes': len(f.source_basis), 'residual_modes': len(f.residual_basis),
                 'source_energy': f.source_energy, 'residual_energy': f.residual_energy, 'seeds': []}
        for seed in [11, 22, 33]:
            model = MLP(f.x.shape[1], f.y.shape[1], seed)
            training_note = model.fit(f.x, f.y, seed)
            notes['seeds'].append({'seed': seed, **training_note})
            models[f'mlp_seed{seed}'] = model.forward
        rows = evaluate(f, [meta[i] for i in held], fields[held], masks[held], models)
        for row in rows:
            row['held_Re'] = held_re
        cv_rows.extend(rows)
        fit_notes.append(notes)
        save_json(OUT / 'cv_progress.json', {'folds_complete': len(fit_notes), 'fit_notes': fit_notes,
                                           'aggregate': aggregate(cv_rows)})
        print('FOLD_COMPLETE', held_re, 'cases', len(held), 'residual_energy', f.residual_energy, flush=True)
    cv = aggregate(cv_rows)
    non_neural = dict(baseline['cv_train80']['aggregate'])
    non_neural.update({name: result for name, result in cv.items() if not name.startswith('mlp_')})
    strongest = min(non_neural, key=lambda name: non_neural[name]['mean_ux_uy_p'][1])
    best_error = non_neural[strongest]['mean_ux_uy_p'][1]
    gains = {seed: 1 - cv[f'mlp_seed{seed}']['mean_ux_uy_p'][1] / best_error for seed in [11, 22, 33]}
    cv_gate = sum(gain >= .15 for gain in gains.values()) >= 2
    median_seed = sorted([11, 22, 33], key=lambda seed: (cv[f'mlp_seed{seed}']['mean_ux_uy_p'][1], seed))[1]
    best_ridge = min([name for name in cv if name.startswith('ridge')], key=lambda name: cv[name]['mean_ux_uy_p'][1])
    best_rbf = min([name for name in cv if name.startswith('rbf')], key=lambda name: cv[name]['mean_ux_uy_p'][1])
    f = Features(meta, fields, masks)
    regressors = {best_ridge: fit_ridge(f, float(best_ridge.split('_a')[1])),
                  best_rbf: fit_rbf(f, float(best_rbf.split('_w')[1].split('_a')[0]), float(best_rbf.split('_a')[1]))}
    final_notes = []
    model_dir = OUT / 'models'
    model_dir.mkdir()
    for seed in [11, 22, 33]:
        model = MLP(f.x.shape[1], f.y.shape[1], seed)
        final_notes.append({'seed': seed, **model.fit(f.x, f.y, seed)})
        regressors[f'mlp_seed{seed}'] = model.forward
        np.savez(model_dir / f'mlp_seed{seed}.npz', **model.p, source_basis=f.source_basis,
                 source_mean=f.source_mean, residual_basis=f.residual_basis,
                 factor=f.factor, xmean=f.xmean, xstd=f.xstd, yscale=f.yscale,
                 donor_ids=np.asarray([q['id'] for q in meta]))
    val_meta = json.loads((BASE / 'val_geometry_metadata.json').read_text())
    x, donors, delta = zip(*(f.query(q) for q in val_meta))
    x = np.stack(x)
    saved_predictions = {}
    for name, regressor in regressors.items():
        coeff = regressor(x)
        saved_predictions[name] = np.stack([f.reconstruct(coeff[i], donors[i], delta[i]) for i in range(8)])
    np.savez(OUT / 'val_predictions.npz', **saved_predictions)
    reference = np.load(BASE / 'val_reference_only.npz')
    val_rows = [{'id': q['id'], 'errors': {name: relative(preds[i], reference['fields'][i], reference['mask'][i])
                                         for name, preds in saved_predictions.items()}}
                for i, q in enumerate(val_meta)]
    val = aggregate(val_rows)
    if strongest in val:
        comparison = {row['id']: row['errors'][strongest] for row in val_rows}
    else:
        comparison = {row['id']: row['errors'][strongest] for row in baseline['val8_development']['cases']}
    candidate = f'mlp_seed{median_seed}'
    better = sum(row['errors'][candidate][1] < comparison[row['id']][1] for row in val_rows)
    comparator_mean = np.mean(list(comparison.values()), axis=0)
    val_gate = bool(better >= 6 and all(val[candidate]['mean_ux_uy_p'][c] <= 1.05 * comparator_mean[c] for c in [0, 2]))
    report = {'status': 'complete', 'task': 'offline_periodic_initial_snapshot_shape_not_forecast',
              'cv_train80': {'aggregate': cv, 'cases': cv_rows, 'fit_notes': fit_notes},
              'strongest_non_mlp_cv': strongest, 'strongest_non_mlp_cv_mean_uy': best_error,
              'cv_relative_gain_by_seed': gains, 'cv_gate_passed': bool(cv_gate),
              'median_seed_selected_on_cv': median_seed, 'val8_development': {'aggregate': val, 'cases': val_rows},
              'val_improved_uy_cases': better, 'val_gate_passed': val_gate,
              'continue_to_cfd_gate_passed': bool(cv_gate and val_gate),
              'source_energy_full_train': f.source_energy, 'residual_energy_full_train': f.residual_energy,
              'final_training_notes': final_notes, 'gradient_check_passed': check['passed'],
              'elapsed_seconds': time.perf_counter() - started,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'future_target_flow_not_used_by_predictor': True, 'test32_and_confirmation12_used': False}
    save_json(OUT / 'model_report.json', report)
    print(json.dumps({'strongest': strongest, 'baseline_uy': best_error, 'gains': gains,
                      'cv_gate': cv_gate, 'val_better': better, 'val_gate': val_gate,
                      'continue': bool(cv_gate and val_gate), 'elapsed': report['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gradient-check', action='store_true')
    args = parser.parse_args()
    if args.gradient_check:
        print(json.dumps(gradient_check()))
    else:
        main()
