#!/usr/bin/env python3
"""F0：现有四帧 FNO 从每条序列的前四帧自主预测，导出验证集逐帧逐通道误差。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402

HORIZONS = [1, 2, 4, 8, 20, 39]


def main() -> None:
    device = torch.device("cuda")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    ckpt = ROOT / "outputs" / "checkpoints" / "fno_long" / "best.pt"
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    store = CaseStore("val")
    cases = []
    with torch.no_grad():
        for i, cid in enumerate(store.ids):
            geom = torch.from_numpy(store.geom[i])[None].to(device)
            fields = torch.from_numpy(store.fields[i]).to(device)
            mask = torch.from_numpy(store.mask[i]).to(device)
            times = store.times[i]
            window = [fields[j] for j in range(4)]
            preds = []
            for _k in range(4, fields.shape[0]):
                hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
                pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
                preds.append(pred)
                window.append(pred)
            err = per_frame_rel(torch.stack(preds), fields[4:], mask).cpu().numpy()
            origin = float(times[3])
            future_t = times[4:] - origin
            horizons = {}
            for seconds in HORIZONS:
                hit = np.nonzero(future_t <= seconds + 1e-4)[0]
                if hit.size == 0 or (seconds >= 20 and future_t[hit[-1]] < seconds - 0.3):
                    continue
                end = err[hit[-1]]
                horizons[str(seconds)] = {
                    "actual_seconds": float(future_t[hit[-1]]),
                    "ux": float(end[0]),
                    "uy": float(end[1]),
                    "p": float(end[2]),
                    "J": float(end.mean()),
                }
            cases.append(
                {
                    "id": cid,
                    "origin_t": origin,
                    "mean": err.mean(axis=0).tolist(),
                    "end": err[-1].tolist(),
                    "horizons": horizons,
                    "frames": err.tolist(),
                }
            )
    summary = {}
    for seconds in HORIZONS:
        rows = [c["horizons"][str(seconds)] for c in cases if str(seconds) in c["horizons"]]
        if not rows:
            continue
        summary[str(seconds)] = {
            "n": len(rows),
            "J_mean": float(np.mean([r["J"] for r in rows])),
            "ux": float(np.mean([r["ux"] for r in rows])),
            "uy": float(np.mean([r["uy"] for r in rows])),
            "p": float(np.mean([r["p"] for r in rows])),
            "worst_J": float(max(r["J"] for r in rows)),
            "worst_id": max((c for c in cases if str(seconds) in c["horizons"]), key=lambda c: c["horizons"][str(seconds)]["J"])["id"],
        }
    dest = OUT / "f0_val.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"ckpt": str(ckpt), "summary": summary, "cases": cases}, indent=2), encoding="utf-8")
    for key, row in summary.items():
        print(f"H {key}s J {row['J_mean']:.4f} ux {row['ux']:.4f} uy {row['uy']:.4f} p {row['p']:.4f} worst {row['worst_id']} {row['worst_J']:.4f}")
    print("targets", TARGET.tolist())


if __name__ == "__main__":
    main()
