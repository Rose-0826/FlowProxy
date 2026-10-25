#!/usr/bin/env python3
"""Losslessly archive exactly eight finished repaired-reference cases only."""
from pathlib import Path
import sys, json, time, shutil, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'cfd/scripts')]
import corrected_benchmark_restart_v1 as producer
from geometry_attack_v2 import save, sha
import true2d_recheck_v1 as solver

OUT = ROOT / 'outputs/corrected_benchmark_restart_v1'
IDS = ['cylinder_Re110_cy-0.2_D1', 'square_Re170_cy0.2_D1']
JOBS = [(group, variant, cid, end)
        for group, variant, end in [
            ('true2d_recheck_v1', 'coarse_dt002', 201),
            ('true2d_recheck_v1', 'fine_dt0005', 302),
            ('corrected_benchmark_quality_v1', 'fine_dt00025', 403),
            ('corrected_benchmark_quality_v1', 'finer_dt0005', 403)]
        for cid in IDS]


def bytes_in(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def run():
    p = json.loads((OUT / 'preregister.json').read_text())
    protected = {ROOT / path: digest for path, digest in p['original_protected_sha256'].items()}
    for path in [OUT / 'preregister.json',
                 ROOT / 'outputs/corrected_benchmark_quality_v1/report.json',
                 ROOT / 'outputs/true2d_recheck_v1/report.json',
                 ROOT / 'configs/splits/body_120.json']:
        if path.exists():
            protected[path] = sha(path)
    allowed = []
    for group, variant, cid, end in JOBS:
        case = ROOT / 'data/raw' / group / variant / cid
        assert case.resolve().is_relative_to((ROOT / 'data/raw' / group).resolve())
        assert cid in IDS and not (case / 'lossless_storage_manifest.json').exists()
        log = case / ('log.icoFoam_quality_dense' if group == 'corrected_benchmark_quality_v1' else 'log.icoFoam')
        solver.assert_solver(log, end)
        assert (case / str(end) / 'U').exists() and (case / str(end) / 'p').exists()
        allowed.append({'path': str(case), 'group': group, 'end': end,
                        'normal_solver_end_checked': True, 'solver_log_sha256': sha(log)})
    save(OUT / 'reference_storage_preregister.json', {
        'registered_utc': solver.now(), 'allowed_finalized_cases': allowed,
        'original_cases_long40_excluded': True, 'old_dataset_weights_reports_unchanged': True,
        'reason': 'reserve disk for complete132 corrected cases; exact-byte reversible representation only',
        'script_sha256': sha(__file__), 'producer_compressor_sha256': sha(producer.__file__)
    })
    done = []
    for record in allowed:
        case = Path(record['path'])
        # Isolated child process only: permit this finalized-reference group.
        producer.RAW = ROOT / 'data/raw' / record['group']
        before = bytes_in(case)
        begin = time.perf_counter()
        count = producer.compress_generated_raw(case)
        after = bytes_in(case)
        assert (case / 'lossless_storage_manifest.json').exists()
        done.append({'case': str(case), 'before_bytes': before, 'after_bytes': after,
                     'files_losslessly_encoded': count, 'seconds': time.perf_counter() - begin,
                     'manifest_sha256': sha(case / 'lossless_storage_manifest.json')})
        for path, expected in protected.items():
            assert sha(path) == expected
        save(OUT / 'reference_storage_status.json', {
            'status': 'running', 'total': len(allowed), 'completed': done,
            'disk_free_GB': shutil.disk_usage(ROOT).free / 2**30,
            'source_originals_and_report_hashes_unchanged': True
        })
        print('FINISHED_REFERENCE_LOSSLESSLY_ARCHIVED', case, round((before - after) / 2**30, 3), flush=True)
    save(OUT / 'reference_storage_status.json', {
        'status': 'complete', 'total': len(allowed), 'completed': done,
        'total_bytes_saved': sum(r['before_bytes'] - r['after_bytes'] for r in done),
        'source_originals_and_report_hashes_unchanged': True
    })


if __name__ == '__main__':
    try:
        run()
    except BaseException as e:
        save(OUT / 'reference_storage_failure.json', {'error': repr(e), 'traceback': traceback.format_exc()})
        raise
