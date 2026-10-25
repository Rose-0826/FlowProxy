#!/usr/bin/env python3
"""Losslessly archive finished old experiments; keep active CFD seeds available."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'cfd/scripts')]
import corrected_benchmark_restart_v1 as producer
from cpu_data_prepare import frames

OUT = ROOT / 'outputs/storage_cleanup_20261003'
FREE_TARGET = 65 * 2**30


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def size(case):
    return sum(p.stat().st_size for p in case.rglob('*') if p.is_file())


def verify(protected):
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Protected file changed: ' + name)


def run():
    os.nice(15)
    OUT.mkdir(exist_ok=False)
    protocol = json.loads((ROOT / 'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    protected = dict(protocol['original_protected_sha256'])
    for p in (ROOT / 'data/processed').glob('*.h5'):
        if 'corrected' not in p.name:
            protected[str(p.relative_to(ROOT))] = digest(p)
    for name in ['configs/splits/body_120.json', 'outputs/corrected_benchmark_restart_v1/preregister.json']:
        protected[name] = digest(ROOT / name)
    # All target/source t=201 VTK files remain in their original, directly readable form.
    for case in sorted((ROOT / 'data/raw/cases_long40').iterdir()):
        stamp, p = min(frames(case), key=lambda z: abs(z[0] - 201))
        assert abs(stamp - 201) < 1e-7
        protected[str(p.relative_to(ROOT))] = digest(p)
    decisions = sorted((ROOT / 'outputs/warmstart_round1/runs').glob('*/*/decision.json'))
    assert len(decisions) == 16
    jobs = []
    for record in decisions:
        value = json.loads(record.read_text())
        assert value['status'] == 'online_confirmed'
        cid, initial = record.parent.parent.name, record.parent.name
        assert initial in ['cold', 'perturbed', 'nearest', 'ai']
        case = ROOT / 'data/raw/warmstart_pilot' / cid / initial
        assert case.is_dir() and case.resolve().is_relative_to((ROOT / 'data/raw/warmstart_pilot').resolve())
        assert not (case / 'lossless_storage_manifest.json').exists()
        jobs.append({'group': 'warmstart_pilot', 'path': str(case)})
    old = ROOT / 'data/raw/cases_long'
    ids = {cid for rows in protocol['splits'].values() for cid in rows}
    assert {p.name for p in old.iterdir() if p.is_dir()} == ids
    for case in sorted(old.iterdir()):
        assert case.resolve().is_relative_to(old.resolve())
        assert not (case / 'lossless_storage_manifest.json').exists()
        assert (case / 'VTK').is_dir() and (case / 'meta.json').exists()
        jobs.append({'group': 'cases_long', 'path': str(case)})
    # Archive only closed experiments, never worker directories or current source seeds.
    active = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / 'comm').read_text().strip() == 'icoFoam':
                active.append((p / 'cwd').resolve())
        except (OSError, ProcessLookupError):
            continue
    for job in jobs:
        case = Path(job['path']).resolve()
        assert not any(p == case or p.is_relative_to(case) for p in active)
        assert not any(p.is_symlink() for p in case.rglob('*'))
    for p in (ROOT / 'outputs/warmstart_round1').rglob('*'):
        if p.is_file():
            protected[str(p.relative_to(ROOT))] = digest(p)
    initial_free = shutil.disk_usage(ROOT).free
    save(OUT / 'inventory.json', {
        'started_utc': now(), 'target_free_GiB': FREE_TARGET / 2**30,
        'initial_free_GiB': initial_free / 2**30, 'allowed_closed_cases': jobs,
        'protected_sha256': protected, 'script_sha256': digest(__file__),
        'representation': 'Exact-byte reversible gzip/native fields and deduplicated VTK; originals removed only after successful round-trip verification',
        'excluded': ['cases_long40', 'corrected_benchmark_restart_v1', 'processed datasets', 'model weights', 'experiment reports'],
    })
    done = []
    for job in jobs:
        case = Path(job['path'])
        if done and shutil.disk_usage(ROOT).free >= FREE_TARGET:
            break
        producer.RAW = ROOT / 'data/raw' / job['group']
        before = size(case)
        start = time.perf_counter()
        count = producer.compress_generated_raw(case)
        after = size(case)
        row = {**job, 'before_bytes': before, 'after_bytes': after,
               'bytes_saved': before - after, 'files': count,
               'seconds': time.perf_counter() - start,
               'manifest_sha256': digest(case / 'lossless_storage_manifest.json')}
        done.append(row)
        save(OUT / 'status.json', {
            'status': 'running', 'updated_utc': now(), 'completed': done,
            'bytes_saved': sum(z['bytes_saved'] for z in done),
            'free_GiB': shutil.disk_usage(ROOT).free / 2**30,
            'target_free_GiB': FREE_TARGET / 2**30,
        })
        print('ARCHIVED', case, 'saved_GiB', round((before - after) / 2**30, 3), flush=True)
    verify(protected)
    save(OUT / 'status.json', {
        'status': 'complete', 'updated_utc': now(), 'completed': done,
        'bytes_saved': sum(z['bytes_saved'] for z in done),
        'free_GiB': shutil.disk_usage(ROOT).free / 2**30,
        'target_free_GiB': FREE_TARGET / 2**30,
        'protected_files_verified': len(protected), 'protected_files_unchanged': True,
    })
    (OUT / 'README.txt').write_text(
        'This archive preserves old CFD values and experiment evidence exactly.\n'
        'Native fields/logs: gzip -dk <file.gz>\n'
        'VTK: python scripts/corrected_benchmark_storage_v1.py <file.vtu.dedup.json>\n'
        'Each case retains lossless_storage_manifest.json and byte SHA256 checksums.\n'
        'All cases_long40 t=201 seed VTK files, original datasets, selected weights,\n'
        'warmstart CSV/JSON records and the active corrected rebuild are preserved.\n', encoding='utf-8')
    print('DONE', 'free_GiB', round(shutil.disk_usage(ROOT).free / 2**30, 2), flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as e:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', {'error': repr(e), 'traceback': traceback.format_exc(), 'at_utc': now()})
        raise
