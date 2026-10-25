#!/usr/bin/env python3
"""按冻结顺序，最多四路并行跑四例四组。"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runner_count() -> int:
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if "warmstart_runner.py" in cmd and " job " in cmd:
            count += 1
    return count


def already_running(case_id: str, method: str) -> bool:
    needle = f"warmstart_runner.py job {case_id} {method}"
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if needle in cmd:
            return True
    return False


def main() -> None:
    cfg = json.loads((ROOT / "outputs/warmstart_round1/preregister.json").read_text(encoding="utf-8"))
    jobs = [(case_id, method) for case_id, methods in cfg["execution"]["method_order_by_case"].items() for method in methods]
    print("jobs", len(jobs), flush=True)
    pending = []
    for case_id, method in jobs:
        result = ROOT / "outputs/warmstart_round1/runs" / case_id / method / "result.json"
        if result.is_file() or already_running(case_id, method):
            print("skip", case_id, method, flush=True)
            continue
        pending.append((case_id, method))
    active: list[subprocess.Popen] = []
    while pending or active:
        active = [proc for proc in active if proc.poll() is None]
        while pending and runner_count() < 4:
            case_id, method = pending.pop(0)
            if already_running(case_id, method):
                print("skip", case_id, method, flush=True)
                continue
            log_path = ROOT / "outputs/warmstart_round1/runs" / f"{case_id}_{method}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                ["/root/miniconda3/bin/python", "-u", str(ROOT / "scripts/warmstart_runner.py"), "job", case_id, method],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append(proc)
            print("start", case_id, method, proc.pid, flush=True)
        time.sleep(5)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
