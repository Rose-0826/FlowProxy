#!/usr/bin/env python3
"""把延长算例 160–200 秒的 161 帧收成新的 h5，划分沿用 configs/splits 里那一份。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.build_dataset import _check_physical_clock, case_to_arrays  # noqa: E402
from flowproxy.dataset import save_cases_h5  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402

SKIP = 160.0
END = 200.0


def _one(case_dir: str) -> dict:
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
    cfg["cfd"]["skip_time"] = SKIP
    case = case_to_arrays(cfg, Path(case_dir))
    _check_physical_clock(case["case_id"], case["times"], SKIP, END)
    if len(case["times"]) != 161:
        raise RuntimeError(f"{case['case_id']}: {len(case['times'])} 帧，不是 161")
    return case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "data" / "raw" / "cases_long")
    parser.add_argument("--split", type=Path, default=ROOT / "configs" / "splits" / "body_120.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "dataset_long.h5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", default=None, help="只收这些算例，冒烟测试用")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在，不覆盖")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    wanted = [cid for role in ("train", "val", "test") for cid in split[role]]
    if args.only is not None:
        split = {role: [c for c in ids if c in set(args.only)] for role, ids in split.items()}
        wanted = [c for c in wanted if c in set(args.only)]
    missing = [cid for cid in wanted if not (args.runs / cid / "VTK").is_dir()]
    if missing:
        raise SystemExit(f"缺 {len(missing)} 个算例的 VTK：" + ", ".join(missing[:10]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(_one, [str(args.runs / cid) for cid in wanted]))
    for case in cases:
        print(f"ingested {case['case_id']}: t={case['times'][0]:.2f}..{case['times'][-1]:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_cases_h5(args.out, cases, split)
    args.out.with_suffix(".splits.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    counts = {role: len(ids) for role, ids in split.items()}
    print(f"wrote {len(cases)} cases {counts} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
