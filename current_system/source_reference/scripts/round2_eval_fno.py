#!/usr/bin/env python3
"""同一套自主预测评测。前四帧是启动，之后不再喂真值。"""

from __future__ import annotations

import argparse
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

NAMES = ("ux", "uy", "p")
TAUS = TARGET.numpy()


def decide(mean: np.ndarray, end: np.ndarray) -> dict:
    ratios = []
    for i, name in enumerate(NAMES):
        ratios.append((float(mean[i] / TAUS[i]), name, "整段平均"))
        ratios.append((float(end[i] / TAUS[i]), name, "末端"))
    value, channel, which = max(ratios)
    return {"R": value, "channel": channel, "which": which}


def first_exceed(err: np.ndarray, times: np.ndarray) -> dict | None:
    for i in range(err.shape[0]):
        for c, tau in enumerate(TAUS):
            if err[i, c] > tau:
                return {"t": float(times[i]), "channel": NAMES[c], "value": float(err[i, c]), "step": int(i)}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    saved = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    store = CaseStore(args.split)
    cases = []
    with torch.no_grad():
        for i, cid in enumerate(store.ids):
            geom = torch.from_numpy(store.geom[i])[None].to(device)
            fields = torch.from_numpy(store.fields[i]).to(device)
            mask = torch.from_numpy(store.mask[i]).to(device)
            times = store.times[i]
            window = [fields[j] for j in range(4)]
            preds = []
            for _ in range(4, fields.shape[0]):
                hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
                pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
                preds.append(pred)
                window.append(pred)
            err = per_frame_rel(torch.stack(preds), fields[4:], mask).cpu().numpy()
            future_t = times[4:]
            mean = err.mean(axis=0)
            end = err[-1]
            choice = decide(mean, end)
            cases.append(
                {
                    "id": cid,
                    "origin_index": 3,
                    "origin_t": float(times[3]),
                    "end_index": int(fields.shape[0] - 1),
                    "end_t": float(times[-1]),
                    "pred_steps": int(err.shape[0]),
                    "mean": mean.tolist(),
                    "end": end.tolist(),
                    "max": err.max(axis=0).tolist(),
                    "R": choice,
                    "first_exceed": first_exceed(err, future_t),
                    "curve_t": future_t.tolist(),
                    "curve": err.tolist(),
                }
            )
    rs = [c["R"]["R"] for c in cases]
    summary = {
        "name": args.name,
        "ckpt": str(args.ckpt),
        "epoch_or_update": saved.get("update", saved.get("epoch")),
        "n": len(cases),
        "mean_R": float(np.mean(rs)),
        "max_R": float(np.max(rs)),
        "n_pass": int(sum(r < 1 for r in rs)),
        "mean_over_cases": {
            "mean": np.mean([c["mean"] for c in cases], axis=0).tolist(),
            "end": np.mean([c["end"] for c in cases], axis=0).tolist(),
            "max": np.mean([c["max"] for c in cases], axis=0).tolist(),
        },
        "end_J": float(np.mean([np.mean(c["end"]) for c in cases])),
        "worst_end_J": max(cases, key=lambda c: float(np.mean(c["end"]))),
    }
    worst = summary["worst_end_J"]
    summary["worst_end_J"] = {"id": worst["id"], "J": float(np.mean(worst["end"])), "end": worst["end"]}
    dest = OUT / f"eval_{args.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8")
    m = summary["mean_over_cases"]
    print(
        f"{args.name} steps {cases[0]['pred_steps']} t {cases[0]['origin_t']:.2f}->{cases[0]['end_t']:.2f} "
        f"mean uy {m['mean'][1]*100:.2f}% end uy {m['end'][1]*100:.2f}% end J {summary['end_J']*100:.2f}% "
        f"mean R {summary['mean_R']:.2f} max R {summary['max_R']:.2f} pass {summary['n_pass']}/{summary['n']}"
    )
    print(f"  worst end J {summary['worst_end_J']['id']} {summary['worst_end_J']['J']*100:.2f}% -> {dest}")


if __name__ == "__main__":
    main()
