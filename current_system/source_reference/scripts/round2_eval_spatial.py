#!/usr/bin/env python3
"""按正式口径重评 8 步底座、对照、涡量和空间修正。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import case_score, per_frame_rel  # noqa: E402
from round2_spatial import Corrector, step_forward  # noqa: E402


def load(path: Path, device):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    corrector = None
    if saved.get("corrector"):
        corrector = Corrector().to(device)
        corrector.load_state_dict(saved["corrector"])
        corrector.eval()
    return model, corrector, saved


@torch.no_grad()
def evaluate(path: Path, device) -> dict:
    model, corrector, saved = load(path, device)
    store = CaseStore("val")
    rows = []
    for i, cid in enumerate(store.ids):
        geom = torch.from_numpy(store.geom[i])[None].to(device)
        fields = torch.from_numpy(store.fields[i]).to(device)
        mask = torch.from_numpy(store.mask[i])
        window = [fields[j][None] for j in range(4)]
        preds = []
        for _ in range(4, fields.shape[0]):
            pred = step_forward(model, corrector, geom, window)
            preds.append(pred[0].cpu())
            window.append(pred)
        err = per_frame_rel(torch.stack(preds), fields[4:].cpu(), mask)
        score = case_score(err)
        ratio = torch.maximum(err.mean(0) / torch.tensor([0.05, 0.05, 0.08]), err[-1] / torch.tensor([0.05, 0.05, 0.08]))
        channel = ["ux", "uy", "p"][int(ratio.argmax())]
        which = "末端" if float(err[-1][int(ratio.argmax())]) >= float(err.mean(0)[int(ratio.argmax())]) else "整段"
        rows.append({"id": cid, "mean": score["mean"], "end": score["end"], "R": score["R"], "channel": channel, "which": which})
    rs = [row["R"] for row in rows]
    return {
        "ckpt": str(path.relative_to(ROOT)),
        "update": saved.get("update"),
        "mean_R": float(np.mean(rs)),
        "max_R": float(np.max(rs)),
        "n_pass": int(sum(r < 1 for r in rs)),
        "mean_uy": float(np.mean([row["mean"][1] for row in rows])),
        "end_uy": float(np.mean([row["end"][1] for row in rows])),
        "worst_end_uy": max(rows, key=lambda row: row["end"][1]),
        "cases": rows,
    }


def main() -> None:
    device = torch.device("cuda")
    names = {
        "base": ROOT / "outputs/checkpoints/fno_k8_base/base.pt",
        "ctrl": ROOT / "outputs/checkpoints/round2_k8_ctrl/best.pt",
        "vort": ROOT / "outputs/checkpoints/round2_k8_vort/best.pt",
        "corrector": ROOT / "outputs/checkpoints/round2_k8_corrector/best.pt",
    }
    base_rows = None
    for name, path in names.items():
        report = evaluate(path, device)
        if name == "base":
            base_rows = {row["id"]: row for row in report["cases"]}
        else:
            for row in report["cases"]:
                row["dR"] = row["R"] - base_rows[row["id"]]["R"]
                row["d_end_uy"] = row["end"][1] - base_rows[row["id"]]["end"][1]
            report["n_cases_R_down"] = int(sum(row["dR"] < 0 for row in report["cases"]))
        OUT.joinpath(f"eval_k8_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        worst = report["worst_end_uy"]
        print(
            name,
            "update", report["update"],
            "meanR", f"{report['mean_R']:.3f}",
            "maxR", f"{report['max_R']:.3f}",
            "pass", report["n_pass"],
            "uy", f"{report['mean_uy']*100:.2f}%",
            "endUy", f"{report['end_uy']*100:.2f}%",
            "worstEnd", worst["id"], f"{worst['end'][1]*100:.2f}%",
            "better", report.get("n_cases_R_down"),
        )


if __name__ == "__main__":
    main()
