#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSequenceDataset, FlowSnapshotDataset  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.train import train_loop  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--model", choices=["unetex", "fno", "unet"], default="unetex")
    p.add_argument("--task", choices=["snapshot", "sequence"], default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--init", type=Path, default=None, help="只加载模型权重，轮次从 1 开始")
    p.add_argument("--time", action="store_true", help="几何模型额外输入物理时刻")
    p.add_argument("--phase", action="store_true", help="几何模型额外输入这一帧涡转到哪一边")
    p.add_argument("--history", type=int, default=None, help="序列任务看前几帧真图，默认用配置里的 4")
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["with_time"] = bool(args.time)
    cfg["data"]["with_phase"] = bool(args.phase)
    if args.history is not None:
        if args.history < 1:
            raise SystemExit("--history 至少为 1")
        cfg["data"]["history_k"] = int(args.history)
    task = args.task or ("snapshot" if args.model in {"unetex", "unet"} else "sequence")
    h5 = args.data or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    if not Path(h5).exists():
        raise SystemExit(f"missing dataset {h5}. Run scripts/build_dataset.py or scripts/smoke_test.py first.")
    if (args.time or args.phase) and task != "snapshot":
        raise SystemExit("时刻和相位只用于几何模型的单帧任务")
    if args.time and args.phase:
        raise SystemExit("时刻和相位不要同时加")
    cases_root = resolve_under_root(cfg, "data_raw") / "cases"
    if task == "snapshot":
        train_set = FlowSnapshotDataset(
            h5, "train", frame="all", with_time=bool(args.time), with_phase=bool(args.phase), cases_root=cases_root
        )
        val_set = FlowSnapshotDataset(
            h5, "val", frame="all", with_time=bool(args.time), with_phase=bool(args.phase), cases_root=cases_root
        )
    else:
        k, t = cfg["data"]["history_k"], cfg["data"]["future_t"]
        train_set = FlowSequenceDataset(h5, "train", k, t)
        val_set = FlowSequenceDataset(h5, "val", k, t)
    tag = args.tag or f"{args.model}_{task}"
    if tag in {
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
        "unetex_120_roll4",
    }:
        raise SystemExit(f"拒绝写入已有权重目录 {tag}，请换一个 --tag")
    if args.resume and args.init:
        raise SystemExit("续训和只加载权重不能同时用")
    existing = resolve_under_root(cfg, "outputs") / "checkpoints" / tag / "best.pt"
    if existing.exists() and not args.resume:
        raise SystemExit(f"{existing} 已存在，换一个 --tag")
    ckpt = train_loop(
        cfg, args.model, task, train_set, val_set, tag=tag, resume=args.resume, init_weights=args.init
    )
    print(f"best checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
