#!/usr/bin/env python3
"""第二条线：形状给稳定周期，开头 10 秒升力给相位。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines import liftcycle  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data" / "processed" / "dataset_long.h5")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "raw" / "cases_long")
    parser.add_argument("--tag", default=liftcycle.TAG)
    parser.add_argument("--eval-name", default=liftcycle.EVAL_NAME)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = liftcycle.run(cfg, args.h5, args.cases, args.tag, args.eval_name, args.epochs)
    print(f"[lift] best checkpoint -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
