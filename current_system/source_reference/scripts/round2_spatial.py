#!/usr/bin/env python3
"""以 8 步 FNO 为底座的三条线：小学习率对照、涡量监督、冻结 FNO 的空间修正。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402

BASE = ROOT / "outputs" / "checkpoints" / "fno_k8_base" / "base.pt"
DX = (20.0 - (-8.0)) / 63
DY = (8.0 - (-8.0)) / 63


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pred: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        return pred + self.net(torch.cat([pred, geom], dim=1))


def fluid_vorticity(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """中心差分。只保留四个邻点都在流体内的格子。返回 [B,H-2,W-2] 和同样形状的掩膜。"""
    dvdx = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * DX)
    dudy = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * DY)
    omega = dvdx - dudy
    center = mask[:, 1:-1, 1:-1] > 0.5
    left = mask[:, 1:-1, :-2] > 0.5
    right = mask[:, 1:-1, 2:] > 0.5
    down = mask[:, :-2, 1:-1] > 0.5
    up = mask[:, 2:, 1:-1] > 0.5
    valid = center & left & right & down & up
    return omega, valid


def omega_scale(store: CaseStore) -> float:
    acc = 0.0
    count = 0.0
    for fields, mask in zip(store.fields, store.mask):
        u = torch.from_numpy(fields[:, 0])
        v = torch.from_numpy(fields[:, 1])
        m = torch.from_numpy(mask)[None].expand(u.shape[0], -1, -1)
        omega, valid = fluid_vorticity(u, v, m)
        acc += float((omega.square() * valid).sum())
        count += float(valid.sum())
    return float(np.sqrt(acc / max(count, 1.0) + 1e-12))


def frame_loss(pred, target, mask, scales) -> torch.Tensor:
    fluid = (mask > 0.5)[:, None]
    err = (pred - target) / scales.reshape(1, 3, 1, 1)
    return (err.square() * fluid).sum() / fluid.sum().clamp_min(1) / 3


def vorticity_loss(pred, target, mask, scale: float) -> torch.Tensor:
    op, valid = fluid_vorticity(pred[:, 0], pred[:, 1], mask)
    ot, _ = fluid_vorticity(target[:, 0], target[:, 1], mask)
    err = (op - ot) / scale
    return (err.square() * valid).sum() / valid.sum().clamp_min(1)


def step_forward(model, corrector, geom, window):
    hist = torch.stack(window[-4:], dim=1).reshape(geom.shape[0], -1, *geom.shape[-2:])
    pred = model(torch.cat([geom, hist], dim=1))[:, :3]
    if corrector is not None:
        pred = corrector(pred, geom)
    return pred


def rollout_loss(model, corrector, geom, fields, mask, scales, omega_s, start: int, use_vorticity: bool) -> torch.Tensor:
    window = [fields[:, start + j] for j in range(4)]
    losses = []
    vort = []
    for k in range(8):
        pred = step_forward(model, corrector, geom, window)
        target = fields[:, start + 4 + k]
        losses.append(frame_loss(pred, target, mask, scales))
        if use_vorticity:
            vort.append(vorticity_loss(pred, target, mask, omega_s))
        window.append(pred)
    stacked = torch.stack(losses)
    loss = 0.5 * stacked.mean() + 0.5 * stacked[-1]
    if use_vorticity:
        vs = torch.stack(vort)
        loss = loss + 0.5 * vs.mean() + 0.5 * vs[-1]
    return loss


@torch.no_grad()
def validate(model, corrector, store: CaseStore, device) -> dict:
    model.eval()
    if corrector is not None:
        corrector.eval()
    scores = []
    channels = []
    for i in range(len(store.ids)):
        geom = torch.from_numpy(store.geom[i])[None].to(device)
        fields = torch.from_numpy(store.fields[i]).to(device)
        mask = torch.from_numpy(store.mask[i]).to(device)
        window = [fields[j][None] for j in range(4)]
        preds = []
        for _ in range(4, fields.shape[0]):
            pred = step_forward(model, corrector, geom, window)
            preds.append(pred[0])
            window.append(pred)
        err = per_frame_rel(torch.stack(preds), fields[4:], mask).cpu()
        mean = err.mean(dim=0)
        end = err[-1]
        scores.append(float(torch.maximum(mean / TARGET, end / TARGET).max()))
        channels.append(mean)
    mean_ch = torch.stack(channels).mean(dim=0)
    return {"R": float(np.mean(scores)), "ux": float(mean_ch[0]), "uy": float(mean_ch[1]), "p": float(mean_ch[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=["ctrl", "vort", "corrector"], required=True)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    lr = args.lr if args.lr is not None else (1e-4 if args.line == "corrector" else 1e-5)
    device = torch.device("cuda")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    saved = torch.load(BASE, map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    corrector = None
    if args.line == "corrector":
        for p in model.parameters():
            p.requires_grad_(False)
        corrector = Corrector().to(device)
        opt = torch.optim.AdamW(corrector.parameters(), lr=lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_set = CaseStore("train")
    val_set = CaseStore("val")
    scales = torch.tensor(train_set.channel_scales(), device=device)
    omega_s = omega_scale(train_set)
    tag = f"round2_k8_{args.line}"
    out = ROOT / "outputs" / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise SystemExit(f"{out} 已有权重")
    (out / "config.json").write_text(
        json.dumps({"line": args.line, "lr": lr, "updates": args.updates, "K": 8, "base": str(BASE),
                    "omega_rms": omega_s, "vorticity": args.line == "vort", "optimizer_of_base": "not_saved_by_previous_run"}, indent=2),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    best = float("inf")
    history = []
    started = time.perf_counter()
    print(f"[{tag}] lr {lr} omega_rms {omega_s:.4f}", flush=True)
    for step in range(1, args.updates + 1):
        model.train()
        if corrector is not None:
            corrector.train()
        pick = rng.choice(len(train_set.ids), size=4, replace=False)
        start = int(rng.integers(0, 161 - 4 - 8))
        geom = torch.from_numpy(np.stack([train_set.geom[i] for i in pick])).to(device)
        fields = torch.from_numpy(np.stack([train_set.fields[i] for i in pick])).to(device)
        mask = torch.from_numpy(np.stack([train_set.mask[i] for i in pick])).to(device)
        opt.zero_grad(set_to_none=True)
        loss = rollout_loss(model, corrector, geom, fields, mask, scales, omega_s, start, args.line == "vort")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corrector.parameters() if corrector is not None else model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            (out / "status.txt").write_text(f"{tag} {step}/{args.updates} loss {float(loss.detach()):.4e}\n", encoding="utf-8")
        if step % 250 == 0 or step == args.updates:
            metrics = validate(model, corrector, val_set, device)
            row = {"update": step, "loss": float(loss.detach()), "minutes": (time.perf_counter() - started) / 60, **metrics}
            history.append(row)
            print(f"[{tag}] update {step} loss {row['loss']:.4e} val R {metrics['R']:.3f} uy {metrics['uy']:.3f} p {metrics['p']:.3f}", flush=True)
            payload = {
                "model": model.state_dict(),
                "corrector": None if corrector is None else corrector.state_dict(),
                "model_name": saved["model_name"],
                "in_channels": saved["in_channels"],
                "out_channels": saved["out_channels"],
                "line": tag,
                "update": step,
                "history": history,
                "omega_rms": omega_s,
            }
            torch.save(payload, out / "last.pt")
            if metrics["R"] < best:
                best = metrics["R"]
                torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{tag}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
