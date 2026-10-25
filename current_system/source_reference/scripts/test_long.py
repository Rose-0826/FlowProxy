#!/usr/bin/env python3
"""长窗口基线的考试：单帧或逐帧模型在测试集上的相对 L2；上一帧模型再加整段连着猜。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.lines.cycle import CaseBank, group_keys, rel_frame, summarize  # noqa: E402
from flowproxy.lines.cyclelock import pack_rollout  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.train import pick_device  # noqa: E402

_spec = importlib.util.spec_from_file_location("test_split", ROOT / "scripts" / "test_split.py")
test_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_split)


@torch.no_grad()
def rollout_prev1(model, bank: CaseBank) -> torch.Tensor:
    state = bank.fields[:, 0]
    rows = []
    for k in range(1, bank.fields.shape[1]):
        pred = model(torch.cat([bank.geom, state], dim=1)).float()
        rows.append(torch.stack([rel_frame(pred[i], bank.fields[i, k], bank.mask[i])[0] for i in range(pred.shape[0])]))
        state = pred
    return torch.stack(rows, dim=1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--h5", type=Path, default=ROOT / "data" / "processed" / "dataset_long.h5")
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--rollout", action="store_true", help="上一帧模型：从第一帧真值出发连着猜整段")
    args = parser.parse_args()
    dest = ROOT / "outputs" / "eval" / args.out_name
    if dest.exists():
        raise SystemExit(f"{dest} 已存在，不覆盖")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = pick_device(str(cfg["train"]["device"]))
    row = test_split.run_one(cfg, args.ckpt, args.h5, device)
    if args.rollout:
        pack = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if int(pack["in_channels"]) != 7:
            raise SystemExit("连着猜只给上一帧模型用")
        model = build_model(pack["model_name"], 7, 3, cfg).to(device)
        model.load_state_dict(pack["model"])
        model.eval()
        bank = CaseBank(args.h5, "test", ROOT / "data" / "raw" / "cases_long", device, with_labels=False)
        row["rollout"] = pack_rollout(bank.ids, rollout_prev1(model, bank), bank.times)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2), encoding="utf-8")
    g = row["groups"]["all"]
    print(f"{args.out_name}: ux {g['rel_l2_ux']*100:.2f}% uy {g['rel_l2_uy']*100:.2f}% p {g['rel_l2_p']*100:.2f}%", flush=True)
    if args.rollout:
        m = row["rollout"]["mean_40s"]["all"]
        print(f"  rollout 40s mean uy {m['rel_l2_uy']*100:.2f}% p {m['rel_l2_p']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
