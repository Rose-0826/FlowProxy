#!/usr/bin/env python3
"""周期网络的干净内插考试。

88 个非考试算例（训练 80 + 验证 8）随机分 FOLDS 组。--fold k 训练除第 k 组以外的算例，
固定轮数、余弦学习率、取最后一轮权重，第 k 组只用来考试。
--fold -1 在 80 个训练算例上同样训满，考原来的 32 个外推算例。
三种考法：每帧真实升力相位（只看模态）、开头 10 秒升力定相位、第一帧读相位。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines import liftcycle  # noqa: E402
from flowproxy.lines.cycle import CaseBank, cycle_field, cycle_freq, group_keys, rel_frame, summarize  # noqa: E402
from flowproxy.lines.cycle_train import train_cycle_net  # noqa: E402
from flowproxy.lines.cyclelock import cycle_from_first_frame  # noqa: E402
from flowproxy.lines.isolate import claim_line, write_eval  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.train import pick_device  # noqa: E402

FOLDS = 5
H5 = ROOT / "data" / "processed" / "dataset_long.h5"
CASES = ROOT / "data" / "raw" / "cases_long"


def pooled(device) -> CaseBank:
    train = CaseBank(H5, "train", CASES, device)
    val = CaseBank(H5, "val", CASES, device)
    pool = object.__new__(CaseBank)
    pool.ids = train.ids + val.ids
    for name in ("geom", "mask", "fields", "times", "phase", "freq"):
        setattr(pool, name, torch.cat([getattr(train, name), getattr(val, name)], dim=0))
    pool.r2 = train.r2 + val.r2
    return pool


@torch.no_grad()
def true_phase(net, bank: CaseBank) -> list[torch.Tensor]:
    raw = net(bank.geom).float()
    return [
        rel_frame(cycle_field(raw[i : i + 1].expand(bank.phase.shape[1], -1, -1, -1), bank.phase[i]), bank.fields[i], bank.mask[i]).cpu()
        for i in range(len(bank.ids))
    ]


def exam(net, bank: CaseBank) -> dict:
    net.eval()
    protocols = {
        "true_phase": true_phase(net, bank),
        "lift_window": liftcycle.predict_bank(net, bank, CASES, "geom"),
        "first_frame": cycle_from_first_frame(net, bank),
    }
    with torch.no_grad():
        freq = cycle_freq(net(bank.geom).float(), bank.mask)
    per_case = []
    for i, cid in enumerate(bank.ids):
        row = {"case": cid, "freq_err_pct": float(abs(freq[i] / bank.freq[i] - 1) * 100)}
        for name, rels in protocols.items():
            mean = rels[i].mean(dim=0)
            row[name] = {"ux": float(mean[0]), "uy": float(mean[1]), "p": float(mean[2])}
        per_case.append(row)
    groups = {}
    for name, rels in protocols.items():
        buckets: dict[str, list[torch.Tensor]] = {}
        for cid, rel in zip(bank.ids, rels):
            for key in group_keys(cid):
                buckets.setdefault(key, []).append(rel)
        groups[name] = summarize(buckets)
    return {"per_case": per_case, "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    root = Path(cfg["_root"])
    device = pick_device(str(cfg["train"]["device"]))
    if args.fold < 0:
        tag, eval_name = "cycle_fixed_full", "test_long_cycle_fixed.json"
    else:
        tag, eval_name = f"cv_cycle_fold{args.fold}", f"cv_cycle_fold{args.fold}.json"
    out_dir, _lock = claim_line(root, tag, frozenset())
    if args.fold < 0:
        fit = CaseBank(H5, "train", CASES, device)
        watch = CaseBank(H5, "val", CASES, device)
        held = CaseBank(H5, "test", CASES, device)
    else:
        pool = pooled(device)
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        order = torch.randperm(len(pool.ids), generator=generator).tolist()
        held_idx = sorted(order[args.fold :: FOLDS])
        fit = pool.subset([i for i in range(len(pool.ids)) if i not in set(held_idx)])
        held = pool.subset(held_idx)
        watch = held
    print(f"[{tag}] fit {len(fit.ids)} cases, exam {len(held.ids)} cases, {args.epochs} fixed epochs", flush=True)

    def log_only(net) -> dict:
        return summarize({"all": cycle_from_first_frame(net, watch)})["all"]

    net, history = train_cycle_net(cfg, fit, log_only, out_dir, tag, args.epochs, args.epochs, device, fixed=True)
    torch.save({"line": "cycle_cv", "tag": tag, "cycle": net.state_dict(), "history": history,
                "fit_ids": fit.ids, "held_ids": held.ids}, out_dir / "final.pt")
    result = exam(net, held)
    result.update({"tag": tag, "fold": args.fold, "epochs": args.epochs, "final_train_loss": history[-1]["train_loss"],
                   "fit_ids": fit.ids, "held_ids": held.ids,
                   "note": "固定轮数、取最后一轮权重；留出算例没有参与挑选权重或调学习率。"})
    write_eval(root, eval_name, json.dumps(result, indent=2), eval_name, frozenset())
    for name, table in result["groups"].items():
        g = table["all"]
        print(f"[{tag}] {name:11s} uy {g['rel_l2_uy']*100:.2f}% p {g['rel_l2_p']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
