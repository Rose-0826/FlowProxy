#!/usr/bin/env python3
"""只训练相位锚定。不写多步推演，也不写旧权重。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines.mode_phase import train_mode  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = train_mode(cfg)
    print(f"[mode] best checkpoint -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
