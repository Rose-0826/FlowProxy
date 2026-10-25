#!/usr/bin/env python3
"""User-authorized deletion of obsolete CFD raw fields, retaining active seeds."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'cfd/scripts')]
from cpu_data_prepare import frames, read_point_fields

OUT = ROOT / 'outputs/obsolete_raw_cleanup_20261003'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    p = Path(path)
    t = p.with_name(p.name + '.partial')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run():
    os.nice(10)
    raw = (ROOT / 'data/raw').resolve()
    old = raw / 'cases_long'
    sources = raw / 'cases_long40'
    assert raw == Path('/root/autodl-tmp/flowproxy/data/raw')
    for p in [old, sources]:
        assert p.is_dir() and not p.is_symlink() and p.resolve().parent == raw
    OUT.mkdir(exist_ok=False)
    protocol = json.loads((ROOT / 'outputs/corrected_benchmark_restart_v1/preregister.json').read_text())
    ids = {cid for rows in protocol['splits'].values() for cid in rows}
    assert len(ids) == 120
    for group in [old, sources]:
        assert {p.name for p in group.iterdir()} == ids
        assert not any(p.is_symlink() for p in group.rglob('*'))
    archiver = json.loads((ROOT / 'outputs/storage_cleanup_20261003_driver.json').read_text())['pid']
    proc = Path('/proc') / str(archiver)
    assert not proc.exists() or (proc / 'stat').read_text().split()[2] == 'Z'
    # No CFD solver runs in either obsolete raw directory.
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if (p / 'comm').read_text().strip() == 'icoFoam':
                    cwd = (p / 'cwd').resolve()
                    assert not cwd.is_relative_to(old) and not cwd.is_relative_to(sources)
            except (OSError, ProcessLookupError):
                pass
    protected = dict(protocol['original_protected_sha256'])
    for p in (ROOT / 'data/processed').glob('*.h5'):
        if 'corrected' not in p.name:
            protected[str(p.relative_to(ROOT))] = digest(p)
    for name in ['configs/splits/body_120.json', 'outputs/corrected_benchmark_restart_v1/preregister.json']:
        protected[name] = digest(ROOT / name)
    plans = []
    keep = set()
    seeds = []
    evidence_names = ['meta.json', 'initialization.json', 'lossless_storage_manifest.json',
                      'log.checkMesh', 'log.checkMesh.gz', 'constant/transportProperties',
                      'system/controlDict', 'system/fvSchemes', 'system/fvSolution',
                      'system/decomposeParDict', 'system/blockMeshDict', 'system/snappyHexMeshDict']
    # Keep exactly the same seed paths and metadata that the frozen initializer reads.
    # Retaining all .vtm/.series indexes avoids races with frames() in active workers.
    for cid in sorted(ids):
        case = sources / cid
        stamp, seed = min(frames(case), key=lambda z: abs(z[0] - 201))
        assert abs(stamp - 201) < 1e-7
        assert seed.is_file() and seed.resolve().is_relative_to(case.resolve())
        points, values = read_point_fields(seed)
        assert len(points) == len(values) and len(points) > 0
        h = digest(seed)
        seeds.append({'id': cid, 'time': stamp, 'source_VTK': str(seed),
                      'sha256': h, 'points': len(points)})
        protected[str(seed.relative_to(ROOT))] = h
        keep.add(seed)
        for p in (case / 'VTK').rglob('*'):
            if p.is_file() and (p.suffix == '.vtm' or p.name.endswith('.series')):
                keep.add(p)
        for name in evidence_names:
            p = case / name
            if p.is_file():
                keep.add(p)
    # Save small mesh/configuration evidence before deleting old full-resolution fields.
    for group in [old, sources]:
        for cid in sorted(ids):
            case = group / cid
            for name in evidence_names:
                p = case / name
                if p.is_file():
                    dest = OUT / 'evidence' / group.name / cid / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)
                    assert digest(p) == digest(dest)
    totals = {'cases_long': 0, 'cases_long40': 0}
    counts = {'cases_long': 0, 'cases_long40': 0}
    for group in [old, sources]:
        for p in group.rglob('*'):
            if p.is_file():
                totals[group.name] += p.stat().st_size
                counts[group.name] += 1
    deleted_source_bytes = 0
    for cid in sorted(ids):
        case = sources / cid
        selected = [p for p in case.rglob('*') if p.is_file() and p not in keep]
        n = sum(p.stat().st_size for p in selected)
        deleted_source_bytes += n
        plans.append({'id': cid, 'paths': selected, 'files': len(selected), 'bytes': n})
    before = shutil.disk_usage(ROOT).free
    save(OUT / 'inventory.json', {
        'recorded_utc': now(), 'authorization': 'User explicitly requests deletion of the two obsolete raw CFD datasets',
        'old_raw_bytes': totals, 'old_raw_files': counts, 'free_before_GiB': before / 2**30,
        'entire_directory_to_delete': str(old), 'source_directory_to_reduce': str(sources),
        'source_fields_to_delete_bytes': deleted_source_bytes,
        'retained_source_seeds': seeds, 'protected_sha256': protected,
        'old_data_status': 'Obsolete numerical reference with known mesh configuration issue; historical results are not official corrected benchmark scores',
        'excluded': ['current corrected raw CFD', 'processed HDF5 data', 'all model weights', 'all experiment score/report files'],
        'script_sha256': digest(__file__),
    })
    assert old.resolve() == raw / 'cases_long' and old.resolve().parent == raw
    shutil.rmtree(old)
    print('DELETED_OBSOLETE_DIRECTORY', old, flush=True)
    done = []
    for plan in plans:
        case = sources / plan['id']
        assert case.resolve().parent == sources.resolve()
        for p in plan['paths']:
            assert p.resolve().is_relative_to(case.resolve()) and p not in keep
            p.unlink()
        for p in sorted([p for p in case.rglob('*') if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
            if not any(p.iterdir()):
                p.rmdir()
        stamp, seed = min(frames(case), key=lambda z: abs(z[0] - 201))
        expected = next(z for z in seeds if z['id'] == plan['id'])
        assert abs(stamp - 201) < 1e-7 and str(seed) == expected['source_VTK']
        assert digest(seed) == expected['sha256']
        (case / 'SEED_ONLY.txt').write_text(
            'Obsolete full time-series raw fields were deleted with user authorization.\n'
            'Only t=201 internal VTK, original VTM/series reading indexes and small configuration evidence remain.\n'
            'Indexes include historical times whose field files have intentionally been deleted.\n'
            'Frozen corrected_benchmark_restart_v1 uses only the unchanged t=201 seed; this is not corrected reference truth.\n',
            encoding='utf-8')
        done.append({k: v for k, v in plan.items() if k != 'paths'})
        save(OUT / 'status.json', {'status': 'running', 'updated_utc': now(),
                                  'cases_long_deleted': True, 'source_cases_reduced': len(done),
                                  'completed': done, 'free_GiB': shutil.disk_usage(ROOT).free / 2**30})
        if len(done) % 20 == 0:
            print('REDUCED_SOURCE_CASES', len(done), 'free_GiB', round(shutil.disk_usage(ROOT).free / 2**30, 2), flush=True)
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, 'Protected file changed: ' + name
    after = shutil.disk_usage(ROOT).free
    save(OUT / 'status.json', {
        'status': 'complete', 'updated_utc': now(), 'cases_long_deleted': not old.exists(),
        'source_cases_reduced': len(done), 'completed': done,
        'retained_seed_count': len(seeds), 'all_seed_paths_bytes_and_reader_verified': True,
        'deleted_file_bytes': totals['cases_long'] + deleted_source_bytes,
        'free_before_GiB': before / 2**30, 'free_GiB': after / 2**30,
        'protected_files_verified': len(protected), 'protected_files_unchanged': True,
    })
    print('CLEANUP_COMPLETE', 'free_GiB', round(after / 2**30, 2), flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as e:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', {'error': repr(e), 'traceback': traceback.format_exc(), 'at_utc': now()})
        raise
