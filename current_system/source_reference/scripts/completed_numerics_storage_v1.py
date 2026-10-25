#!/usr/bin/env python3
"""Archive twelve finalized numerical controls, preserving all error evidence."""
from pathlib import Path
import sys, json, re, time, shutil, traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT / 'cfd/scripts')]
import corrected_benchmark_restart_v1 as producer
from geometry_attack_v2 import save, sha
import true2d_recheck_v1 as solver
OUT = ROOT / 'outputs/corrected_benchmark_restart_v1'
GROUPS = ['cfd_numerics_restart_v1', 'cfd_numerics_round2_v1', 'true2d_time_control_v1']
IDS = {'cylinder_Re110_cy-0.2_D1', 'square_Re170_cy0.2_D1'}

def size(path):
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

def run():
    p = json.loads((OUT / 'preregister.json').read_text())
    protected = {ROOT / path: digest for path, digest in p['original_protected_sha256'].items()}
    jobs = []
    for group in GROUPS:
        directory = ROOT / 'outputs' / group
        status = json.loads((directory / 'status.json').read_text())
        assert status['status'] == 'complete' and not status['errors']
        for f in directory.rglob('*.json'):
            protected[f] = sha(f)
        for row in status['completed']:
            assert row['id'] in IDS and re.fullmatch(r'(coarse|fine|third)_dt[0-9]+', row['variant'])
            case = ROOT / 'data/raw' / group / row['variant'] / row['id']
            assert case.resolve().is_relative_to((ROOT / 'data/raw' / group).resolve())
            assert not (case / 'lossless_storage_manifest.json').exists()
            log = case / 'log.icoFoam'
            content = log.read_text()
            stamps = re.findall(r'^Time = (\S+)', content, re.M)
            assert stamps
            end = float(stamps[-1]); solver.assert_solver(log, end)
            assert (case / f'{end:.10g}' / 'U').exists() and (case / f'{end:.10g}' / 'p').exists()
            jobs.append({'path': str(case), 'group': group, 'final_time': end, 'solver_log_sha256': sha(log)})
    assert len(jobs) == 12
    save(OUT / 'completed_numerics_storage_preregister.json', {
        'registered_utc': solver.now(), 'cases': jobs, 'script_sha256': sha(__file__),
        'scope': 'finished numerical controls only; original cases_long40 and active rebuild excluded',
        'raw_values_and_negative_results_preserved': True,
        'reason': 'ensure complete132 rebuild has disk reserve; no solver/model/label changes'
    })
    done = []
    for job in jobs:
        case = Path(job['path']); producer.RAW = ROOT / 'data/raw' / job['group']
        before = size(case); begin = time.perf_counter()
        n = producer.compress_generated_raw(case); after = size(case)
        for path, expected in protected.items(): assert sha(path) == expected
        done.append({'case': str(case), 'before_bytes': before, 'after_bytes': after,
                     'files': n, 'seconds': time.perf_counter() - begin,
                     'manifest_sha256': sha(case / 'lossless_storage_manifest.json')})
        save(OUT / 'completed_numerics_storage_status.json', {
            'status': 'running', 'completed': done, 'total': len(jobs),
            'disk_free_GB': shutil.disk_usage(ROOT).free / 2**30,
            'original_source_weights_reports_unchanged': True
        })
        print('NUMERICAL_CONTROL_ARCHIVED_LOSSLESSLY', case, round((before-after)/2**30, 3), flush=True)
    save(OUT / 'completed_numerics_storage_status.json', {
        'status': 'complete', 'completed': done, 'total': len(jobs),
        'bytes_saved': sum(r['before_bytes']-r['after_bytes'] for r in done),
        'original_source_weights_reports_unchanged': True
    })

if __name__ == '__main__':
    try: run()
    except BaseException as e:
        save(OUT / 'completed_numerics_storage_failure.json', {'error': repr(e), 'traceback': traceback.format_exc()})
        raise
