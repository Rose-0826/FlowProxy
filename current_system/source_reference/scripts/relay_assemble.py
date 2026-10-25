#!/usr/bin/env python3
"""等正在进行的组包结束。若中途退出且没有写出 dataset_120.h5，再自动重跑一次。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "dataset_120.h5"
LOG = ROOT / "tmp" / "assemble_120.log"
STATUS = ROOT / "tmp" / "assemble_120.status"
PIDFILE = ROOT / "tmp" / "assemble_120.pid"
CMD = [
    sys.executable,
    str(ROOT / "scripts" / "build_dataset.py"),
    "--assemble",
    str(ROOT / "data" / "processed" / "dataset_timefixed.h5"),
    "--out",
    str(OUT),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return "build_dataset.py" in cmdline(pid) and "dataset_120.h5" in cmdline(pid)


def succeeded() -> bool:
    if not OUT.is_file() or not LOG.is_file():
        return False
    text = LOG.read_text(encoding="utf-8", errors="replace")
    return "wrote 120 cases" in text


def wait_pid(pid: int) -> None:
    while alive(pid):
        log(f"waiting assemble pid {pid}")
        time.sleep(60)


def run_once() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("\nrelay restart\n")
        proc = subprocess.Popen(CMD, stdout=handle, stderr=subprocess.STDOUT)
    PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    log(f"restarted assemble pid {proc.pid}")
    return proc.wait()


def verify() -> None:
    import h5py
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from flowproxy.splits import VAL_CASE_IDS, held_out_geometry

    protected = {
        ROOT / "data/processed/dataset.h5": 422064283,
        ROOT / "data/processed/dataset.h5.bak": 422064283,
        ROOT / "outputs/checkpoints/unetex_snapshot/best.pt": 2354577,
        ROOT / "outputs/checkpoints/fno_sequence/best.pt": 50497425,
        ROOT / "outputs/checkpoints/fno_sequence/last.pt": 50497617,
    }
    for path, size in protected.items():
        if path.stat().st_size != size:
            raise RuntimeError(f"保护文件大小变了: {path}")
    problems = []
    with h5py.File(OUT, "r") as handle:
        ids = list(handle["cases"].keys())
        if len(ids) != 120:
            problems.append(f"算例数 {len(ids)}")
        roles = {}
        for name in ("train", "val", "test"):
            raw = handle["splits"][name][()]
            roles[name] = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        if [len(roles[k]) for k in ("train", "val", "test")] != [80, 8, 32]:
            problems.append("划分不是 80/8/32")
        if set(roles["val"]) != set(VAL_CASE_IDS):
            problems.append("验证集名单不对")
        if set(roles["train"]) & set(roles["val"]) or set(roles["train"]) & set(roles["test"]) or set(roles["val"]) & set(roles["test"]):
            problems.append("划分有重叠")
        for cid in ids:
            group = handle["cases"][cid]
            times = np.asarray(group["times"][()], dtype=np.float64)
            fields = group["fields"]
            if fields.shape[0] != 161 or abs(float(times[0]) - 40) > 1e-3 or abs(float(times[-1]) - 80) > 0.3:
                problems.append(f"{cid} 帧或时间不对")
                break
            if np.any(np.diff(times) <= 1e-6):
                problems.append(f"{cid} 时间不递增")
                break
        for cid in roles["test"]:
            group = handle["cases"][cid]
            if not held_out_geometry(float(group.attrs["cy"]), float(group.attrs["D"])):
                problems.append(f"{cid} 不该在测试集")
                break
    if problems:
        raise RuntimeError("; ".join(problems))
    log("check ok 120 cases 80/8/32")


def main() -> None:
    if PIDFILE.is_file():
        text = PIDFILE.read_text(encoding="utf-8").strip()
        if text.isdigit():
            wait_pid(int(text))
    if not succeeded():
        rc = run_once()
        if rc != 0 or not succeeded():
            STATUS.write_text(f"failed rc={rc}\n", encoding="utf-8")
            raise SystemExit(f"assemble failed rc={rc}")
    verify()
    STATUS.write_text("check ok, training next\n", encoding="utf-8")
    log("check ok, start training")
    train("unetex", "snapshot", "unetex_120")
    train("fno", "sequence", "fno_120")
    STATUS.write_text("done dataset_120.h5 and both models\n", encoding="utf-8")
    log("all done")


def train(model: str, task: str, tag: str) -> None:
    if tag in {"unetex_snapshot", "fno_sequence"}:
        raise SystemExit(f"拒绝写入已有圆柱权重目录 {tag}")
    log_path = ROOT / "tmp" / f"train_{tag}.log"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train.py"),
        "--model",
        model,
        "--task",
        task,
        "--data",
        str(OUT),
        "--tag",
        tag,
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    STATUS.write_text(f"training {tag} pid {proc.pid}\n", encoding="utf-8")
    log(f"training {tag} pid {proc.pid}")
    rc = proc.wait()
    if rc != 0:
        STATUS.write_text(f"failed {tag} rc={rc}\n", encoding="utf-8")
        raise SystemExit(f"{tag} failed rc={rc}")
    log(f"{tag} finished")


if __name__ == "__main__":
    main()
