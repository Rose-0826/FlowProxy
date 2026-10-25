"""Bounded real icoFoam stop check, using a training case's cached mesh.

Creates a new case directory. Does not train, generate a mesh, or score AI.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from aic_stop_control import active_group_members, stop_job

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/warmstart_iface/cylinder_Re100_cy0_D1'
OUT = ROOT / 'outputs/stop_engineering_2026-10-02'
CASE = OUT / 'case'


def file_state() -> dict:
    files = [CASE / 'log.icoFoam']
    files.extend((CASE / 'postProcessing').glob('**/coefficient.dat'))
    return {str(path.relative_to(CASE)): {'bytes': path.stat().st_size,
             'mtime_ns': path.stat().st_mtime_ns} for path in files if path.is_file()}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f'Refusing to overwrite engineering record: {OUT}')
    if not (SOURCE / 'constant/polyMesh').is_dir():
        raise RuntimeError('Cached training-case mesh is missing')
    OUT.mkdir(parents=True)
    CASE.mkdir()
    for name in ['constant', 'system', '0.orig']:
        shutil.copytree(SOURCE / name, CASE / name)
    shutil.copytree(CASE / '0.orig', CASE / '0')
    control = CASE / 'system/controlDict'
    text = control.read_text()
    for key, value in [('startFrom', 'startTime'), ('startTime', '0'),
                       ('endTime', '10')]:
        text = re.sub(rf'\b{key}\s+[^;]+;', f'{key} {value};', text)
    control.write_text(text)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['PWD'] = str(CASE)
    cmd = (f'source {shlex.quote(str(ROOT / "cfd/env.sh"))} && '
           f'cd {shlex.quote(str(CASE))} && icoFoam > log.icoFoam 2>&1')
    started = time.perf_counter()
    proc = subprocess.Popen(['bash', '-lc', cmd], cwd=CASE, env=env,
                            start_new_session=True)
    solver_seen = False
    stop_record = None
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            log = CASE / 'log.icoFoam'
            text = log.read_text(errors='replace') if log.exists() else ''
            if re.search(r'^Time = ', text, re.M) and any(
                    member['name'] == 'icoFoam'
                    for member in active_group_members(proc.pid)):
                solver_seen = True
                time.sleep(1.0)
                break
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        stop_record = stop_job(proc)
    finally:
        # A diagnostic exception must not leave an unobserved solver running.
        if active_group_members(proc.pid):
            stop_job(proc)
    stopped_wall = time.perf_counter() - started
    first = file_state()
    time.sleep(2.0)
    second = file_state()
    text = (CASE / 'log.icoFoam').read_text(errors='replace')
    stamps = re.findall(r'^Time = ([^\n]+)', text, re.M)
    report = {
        'kind': 'engineering_only_not_AI_performance',
        'case_id': 'cylinder_Re100_cy0_D1', 'split': 'train',
        'solver_seen_before_stop': solver_seen,
        'stop_control': stop_record, 'returncode': proc.returncode,
        'last_solver_time': float(stamps[-1]) if stamps else None,
        'solver_wall_through_verified_stop': stopped_wall,
        'log_and_forces_unchanged_after_2s': first == second,
        'file_state_at_stop': first, 'file_state_after_2s': second,
        'fatal': 'FOAM FATAL' in text,
    }
    report['passed'] = bool(solver_seen and not report['fatal'] and
                            not active_group_members(proc.pid) and first == second)
    (OUT / 'stop_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
