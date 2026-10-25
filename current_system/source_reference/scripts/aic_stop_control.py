"""Linux process-group control for one isolated OpenFOAM job.

Launch with Popen(start_new_session=True). This module has no ML dependencies.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def active_group_members(pgid: int) -> list[dict]:
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
            rest = raw[raw.rfind(')') + 2:].split()
            if int(rest[2]) == pgid and rest[0] != 'Z':
                members.append({'pid': int(entry.name), 'state': rest[0],
                                'name': raw[raw.find('(') + 1:raw.rfind(')')]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sorted(members, key=lambda member: member['pid'])


def stop_job(proc: subprocess.Popen, grace_seconds: float = 10.0) -> dict:
    """Wait for the entire job, including a redirected solver child, to exit."""
    pgid = proc.pid
    started = time.perf_counter()
    before = active_group_members(pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while True:
        proc.poll()  # Reap the direct child; a zombie is not an active solver.
        remaining = active_group_members(pgid)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    escalated = bool(remaining)
    if escalated:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5.0)
    deadline = time.monotonic() + 5.0
    while active_group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    after = active_group_members(pgid)
    if after:
        raise RuntimeError(f'Job group {pgid} has active members after stop: {after}')
    return {'pgid': pgid, 'active_before_stop': before,
            'active_after_stop': after, 'sigkill_needed': escalated,
            'stop_wall_seconds': time.perf_counter() - started}
