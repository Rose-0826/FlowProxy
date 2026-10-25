#!/usr/bin/env python3
"""只训练多步推演。不写相位锚定，也不写旧权重。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines.pushforward import train_roll  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = train_roll(cfg)
    print(f"[roll] best checkpoint -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
