#!/usr/bin/env python3
"""第一条线修正版：交叉分组的周期帧，推演网络必须会看自己的上一帧。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines import cyclelock_oof  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data" / "processed" / "dataset_long.h5")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "raw" / "cases_long")
    parser.add_argument("--tag", default=cyclelock_oof.TAG)
    parser.add_argument("--eval-name", default=cyclelock_oof.EVAL_NAME)
    parser.add_argument("--cycle-epochs", type=int, default=150)
    parser.add_argument("--roll-epochs", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = cyclelock_oof.run(cfg, args.h5, args.cases, args.tag, args.eval_name, args.cycle_epochs, args.roll_epochs)
    print(f"[oof] best checkpoint -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
