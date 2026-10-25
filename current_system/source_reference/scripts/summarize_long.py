#!/usr/bin/env python3
"""把长窗口实验的考试结果汇成一张表，写 outputs/eval/summary_long.md。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "eval"


def pct(row: dict) -> str:
    return f"{row['rel_l2_ux']*100:.2f}% | {row['rel_l2_uy']*100:.2f}% | {row['rel_l2_p']*100:.2f}%"


def load(name: str) -> dict | None:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    lines = [
        "# 长窗口（160–200 秒）考试结果",
        "",
        "32 个留出算例。相对 L2，前后流速 | 上下流速 | 压力。",
        "",
        "| 做法 | 输入里有什么 | 前后 | 上下 | 压力 |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows = [
        ("只看形状 UNetEx", "形状、雷诺数", "test_long_unetex.json", lambda d: d["groups"]["all"]),
        ("第二条线 形状周期", "形状、雷诺数、开头 10 秒升力", "test_long_liftcycle.json", lambda d: d["geom_freq"]["all"]),
        ("第二条线 频率也用升力", "形状、雷诺数、开头 10 秒升力", "test_long_liftcycle.json", lambda d: d["lift_freq"]["all"]),
        ("第一条线 只画周期", "形状、雷诺数、第一帧", "rollout_long_cyclelock.json", lambda d: d["cycle_only_first_frame"]["all"]),
        ("第一条线 连着猜 40 秒", "形状、雷诺数、第一帧", "rollout_long_cyclelock.json", lambda d: d["rollout"]["mean_40s"]["all"]),
        ("第一条线修正版 连着猜 40 秒", "形状、雷诺数、第一帧", "rollout_long_cyclelock_oof.json", lambda d: d["rollout"]["mean_40s"]["all"]),
        ("第一条线修正版 不给周期帧", "形状、雷诺数、第一帧", "rollout_long_cyclelock_oof.json", lambda d: d["rollout_no_prior"]["mean_40s"]["all"]),
        ("上一帧 UNetEx 连着猜 40 秒", "形状、雷诺数、第一帧", "test_long_prev1.json", lambda d: d["rollout"]["mean_40s"]["all"]),
        ("上一帧 UNetEx 每步喂真值", "每一步的真实上一帧", "test_long_prev1.json", lambda d: d["groups"]["all"]),
        ("第一条线 每步喂真值", "每一步的真实上一帧", "rollout_long_cyclelock.json", lambda d: d["one_step_true_prev"]["all"]),
        ("第一条线修正版 每步喂真值", "每一步的真实上一帧", "rollout_long_cyclelock_oof.json", lambda d: d["one_step_true_prev"]["all"]),
        ("FNO 每步喂真实前 4 帧", "每一步的真实前 4 帧", "test_long_fno.json", lambda d: d["groups"]["all"]),
    ]
    for label, inputs, name, pick in rows:
        data = load(name)
        cells = pct(pick(data)) if data else "未完成 | | "
        lines.append(f"| {label} | {inputs} | {cells} |")
    lock = load("rollout_long_cyclelock.json")
    if lock:
        lines += ["", f"第一条线拉回比例在验证集上选的是 {lock['pull_selected_on_val']}。最后一步上下流速："
                  f"{lock['rollout']['last_step']['all']['rel_l2_uy']*100:.2f}%。"]
    dest = EVAL / "summary_long.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
