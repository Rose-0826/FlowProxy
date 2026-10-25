#!/usr/bin/env python3
"""Frozen val/test persistence and linear baselines; no model selection."""
from pathlib import Path
import hashlib
import json
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cpu_round1/full40_baselines'
sys.path.insert(0, str(ROOT / 'scripts'))
from round2_complete40_v1 import frame_summary, summarize, horizon_selection


def write(path, content):
    path.write_text(json.dumps(content, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    split = json.loads((ROOT / 'configs/splits/body_120.json').read_text())
    protocol = {'task': 'frozen40 supplemental baseline evaluation',
                'roles': ['val', 'test'], 'no_model_or_direction_selection': True,
                'history': 'only four true support frames, indices0..3',
                'methods': {'copy': 'constant last support frame',
                            'linear_last2': 'last support + actual elapsed time * last2 temporal slope',
                            'linear_4_ls': 'least-squares slope from four support frames, anchored at last frame'},
                'true_future_read_after_prediction_parameters_saved': True,
                'pressure': 'raw HDF5, unchanged', 'horizons': [2, 8, 20, 40],
                'source': 'data/processed/dataset_long40_v1.h5',
                'confirmation12_used': False, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT / 'preregister.json', protocol)
    params_dir = OUT / 'prediction_parameters'; params_dir.mkdir()
    with h5py.File(ROOT / protocol['source'], 'r') as handle:
        for role in protocol['roles']:
            cases = []
            for cid in split[role]:
                group = handle['cases'][cid]
                support = group['fields'][:4].astype(np.float32)
                times = group['times'][...].astype(np.float64)
                origin = times[3]; elapsed = times[4:164] - origin
                last = support[-1]
                slope2 = (support[-1] - support[-2]) / (times[3] - times[2])
                centered = times[:4] - times[:4].mean()
                slope4 = np.tensordot(centered, support, axes=(0,0)) / np.square(centered).sum()
                param_path = params_dir / (cid + '.npz')
                np.savez_compressed(param_path, last=last, slope_last2=slope2,
                                    slope_4_ls=slope4, elapsed_seconds=elapsed)
                predicted = {'copy': np.broadcast_to(last, (160, *last.shape)).copy(),
                             'linear_last2': (last[None] + elapsed[:,None,None,None] * slope2[None]).astype(np.float32),
                             'linear_4_ls': (last[None] + elapsed[:,None,None,None] * slope4[None]).astype(np.float32)}
                hashes = {name: hashlib.sha256(values.tobytes()).hexdigest() for name, values in predicted.items()}
                write(params_dir / (cid + '.json'), {'origin': float(origin), 'prediction_sha256': hashes,
                      'parameter_file_sha256': hashlib.sha256(param_path.read_bytes()).hexdigest()})
                # The future fields are first read after the baseline parameters and hashes are saved.
                truth = group['fields'][4:164].astype(np.float32)
                mask = (group['mask'][...] > .5)[None,None]
                denominator = np.maximum(np.linalg.norm(truth * mask, axis=(2,3)), 1e-8)
                row = {'id': cid, 'methods': {}}
                for name, values in predicted.items():
                    err = np.linalg.norm((values - truth) * mask, axis=(2,3)) / denominator
                    horizons = {}
                    for seconds in protocol['horizons']:
                        take, complete = horizon_selection(times, seconds)
                        horizons[f'{seconds}s'] = frame_summary(err[take], origin, times[4+take[-1]], seconds, complete)
                    row['methods'][name] = {'horizons': horizons, 'curve': err.tolist()}
                cases.append(row)
            aggregate = {}
            for name in protocol['methods']:
                aggregate[name] = {}
                for seconds in protocol['horizons']:
                    key = f'{seconds}s'
                    rows = [{'id': c['id'], **c['methods'][name]['horizons'][key]} for c in cases]
                    aggregate[name][key] = summarize(rows)
            write(OUT / f'{role}.json', {'protocol': protocol, 'summary': aggregate, 'cases': cases})
            print(role, json.dumps({name: {key: {'R': s['mean_R'], 'Uy': s['mean'][1], 'pass': s['n_pass']}
                                           for key, s in blocks.items()}
                                    for name, blocks in aggregate.items()}), flush=True)


if __name__ == '__main__':
    main()
