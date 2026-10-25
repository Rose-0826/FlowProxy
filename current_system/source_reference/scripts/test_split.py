#!/usr/bin/env python3
"""在留出测试集上按形状、流速、几何类型分别算相对 L2。不覆盖旧的 compare.json。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.dataset import FlowSequenceDataset, FlowSnapshotDataset  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.splits import _meta_from_id, held_out_geometry  # noqa: E402
from flowproxy.train import pick_device  # noqa: E402


def per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    pred = pred * mask
    target = target * mask
    if target.shape[1] > 3:
        target = target[:, :3]
        pred = pred[:, :3]
    num = torch.linalg.vector_norm(pred - target, dim=(2, 3))
    den = torch.linalg.vector_norm(target, dim=(2, 3)).clamp_min(1e-8)
    return num / den


def kind_of(cy: float, diameter: float) -> str:
    if held_out_geometry(cy, diameter) and abs(abs(cy) - 0.8) < 1e-6:
        return "far"
    if held_out_geometry(cy, diameter):
        return "extreme_d"
    return "other"


def summarize(buckets: dict[str, list[torch.Tensor]]) -> dict[str, dict]:
    out = {}
    for key, rows in buckets.items():
        stacked = torch.stack(rows, dim=0)
        mean = stacked.mean(dim=0)
        out[key] = {
            "n": int(stacked.shape[0]),
            "rel_l2_ux": float(mean[0]),
            "rel_l2_uy": float(mean[1]),
            "rel_l2_p": float(mean[2]),
            "rel_l2_mean": float(mean.mean()),
        }
    return out


@torch.no_grad()
def run_one(cfg: dict, ckpt: Path, h5: Path, device: torch.device) -> dict:
    pack = torch.load(ckpt, map_location="cpu")
    task = pack["task"]
    with_time = bool(pack.get("config", {}).get("data", {}).get("with_time", False))
    with_phase = bool(pack.get("config", {}).get("data", {}).get("with_phase", False))
    if task == "snapshot":
        cases_root = resolve_under_root(cfg, "data_raw") / "cases"
        ds = FlowSnapshotDataset(
            h5, "test", frame="all", with_time=with_time, with_phase=with_phase, cases_root=cases_root
        )
    else:
        data_cfg = pack.get("config", {}).get("data", {})
        k = int(data_cfg.get("history_k", cfg["data"]["history_k"]))
        t = int(data_cfg.get("future_t", cfg["data"]["future_t"]))
        ds = FlowSequenceDataset(h5, "test", k, t)
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]))
    buckets: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        pred = model(batch["x"].to(device))
        rel = per_sample(pred, batch["y"].to(device), batch["mask"].to(device)).cpu()
        for i, cid in enumerate(batch["case_id"]):
            meta = _meta_from_id(cid)
            row = rel[i]
            buckets["all"].append(row)
            buckets[f"shape:{meta['shape']}"].append(row)
            buckets[f"Re:{int(meta['Re'])}"].append(row)
            buckets[f"kind:{kind_of(meta['cy'], meta['D'])}"].append(row)
    return {
        "model": pack["model_name"],
        "task": task,
        "ckpt": str(ckpt),
        "n": len(ds),
        "groups": summarize(buckets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--unetex", type=Path, default=None)
    parser.add_argument("--fno", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    h5 = args.data or (resolve_under_root(cfg, "data_processed") / "dataset_120.h5")
    out_root = resolve_under_root(cfg, "outputs")
    unetex = args.unetex or out_root / "checkpoints" / "unetex_120" / "best.pt"
    fno = args.fno or out_root / "checkpoints" / "fno_120" / "best.pt"
    device = pick_device(str(cfg["train"]["device"]))
    rows = [run_one(cfg, Path(ckpt), Path(h5), device) for ckpt in (unetex, fno)]
    dest = out_root / "eval" / "test_120.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    for row in rows:
        print(f"\n{row['model']} {row['task']} n={row['n']}")
        for key in sorted(row["groups"]):
            g = row["groups"][key]
            print(
                f"  {key:16s} n={g['n']:5d}  mean={g['rel_l2_mean']:.4f}  "
                f"ux={g['rel_l2_ux']:.4f}  uy={g['rel_l2_uy']:.4f}  p={g['rel_l2_p']:.4f}"
            )


if __name__ == "__main__":
    main()
