"""保证两条实验不会写到对方或旧权重上。"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

_HELD_LOCKS: list[object] = []

PROTECTED_TAGS = frozenset(
    {
        "unetex_snapshot",
        "fno_sequence",
        "unetex_120",
        "unetex_120_time",
        "unetex_120_phase",
        "unetex_120_prev1",
        "fno_120",
        "unetex_car_prev1",
        "unetex_car_snapshot",
        "fno_car",
        "unetex_120_mode",
    }
)

FORBIDDEN_EVAL = frozenset(
    {
        "test_120.json",
        "test_120_time.json",
        "test_120_phase.json",
        "test_120_prev1.json",
        "test_car_prev1.json",
        "test_car_snapshot.json",
        "test_car_fno.json",
        "rollout_120.json",
        "speedup.json",
        "test_120_mode.json",
    }
)


def claim_line(root: Path, tag: str, rivals: frozenset[str]) -> tuple[Path, object]:
    """独占自己的权重目录。锁在进程结束前一直拿着。"""
    if tag in PROTECTED_TAGS or tag in rivals or not tag:
        raise SystemExit(f"拒绝占用权重目录 {tag}")
    parent = (root / "outputs" / "checkpoints").resolve()
    out = (parent / tag).resolve()
    if out.parent != parent or out.name != tag:
        raise SystemExit(f"权重目录越界 {out}")
    out.mkdir(parents=True, exist_ok=True)
    handle = open(out / ".lock", "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f"{tag} 已有训练在写这个目录") from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _HELD_LOCKS.append(handle)
    return out, handle


def write_status(out_dir: Path, text: str) -> None:
    path = out_dir / "status.txt"
    if path.parent.resolve() != out_dir.resolve():
        raise SystemExit(f"状态文件越界 {path}")
    path.write_text(text + "\n", encoding="utf-8")


def write_eval(root: Path, filename: str, text: str, own: str, rivals: frozenset[str]) -> Path:
    if filename in FORBIDDEN_EVAL or filename in rivals or filename != own:
        raise SystemExit(f"拒绝写入考试文件 {filename}")
    if "/" in filename or filename.startswith("."):
        raise SystemExit(f"拒绝写入考试文件 {filename}")
    dest_dir = (root / "outputs" / "eval").resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / filename).resolve()
    if dest.parent != dest_dir:
        raise SystemExit(f"考试文件越界 {dest}")
    dest.write_text(text, encoding="utf-8")
    return dest
