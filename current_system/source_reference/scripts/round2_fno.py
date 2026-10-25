#!/usr/bin/env python3
"""E/F：从现有四帧 FNO 出发，做真正的多步反向传播。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402

F0 = ROOT / "outputs" / "checkpoints" / "fno_long" / "best.pt"
SCHEDULE = {
    "E": [(1000, 4), (1000, 8)],
    "F": [(500, 4), (500, 8), (1000, 16)],
}


def unroll_of(line: str, step: int) -> int:
    cursor = 0
    for count, width in SCHEDULE[line]:
        cursor += count
        if step <= cursor:
            return width
    return SCHEDULE[line][-1][1]


def one_step(model, geom, window):
    hist = torch.stack(window[-4:], dim=1).reshape(geom.shape[0], -1, *geom.shape[-2:])
    return model(torch.cat([geom, hist], dim=1))[:, :3]


def frame_loss(pred, target, mask, scales) -> torch.Tensor:
    fluid = (mask > 0.5)[:, None]
    err = (pred - target) / scales.reshape(1, 3, 1, 1)
    return (err.square() * fluid).sum() / fluid.sum().clamp_min(1) / 3


def rollout_loss(model, geom, fields, mask, scales, start: int, width: int) -> torch.Tensor:
    window = [fields[:, start + j] for j in range(4)]
    losses = []
    for k in range(width):
        pred = one_step(model, geom, window)
        losses.append(frame_loss(pred, fields[:, start + 4 + k], mask, scales))
        window.append(pred)
    stacked = torch.stack(losses)
    return 0.5 * stacked.mean() + 0.5 * stacked[-1]


def truncated_loss(model, geom, fields, mask, scales, start: int, push: int, backprop: int) -> torch.Tensor:
    """先无梯度推进 push 步，再只对后面 backprop 步反传。途中不重置为真值。"""
    window = [fields[:, start + j] for j in range(4)]
    with torch.no_grad():
        for k in range(push):
            window.append(one_step(model, geom, window))
    losses = []
    for k in range(backprop):
        pred = one_step(model, geom, window)
        losses.append(frame_loss(pred, fields[:, start + 4 + push + k], mask, scales))
        window.append(pred)
    stacked = torch.stack(losses)
    return 0.5 * stacked.mean() + 0.5 * stacked[-1]


@torch.no_grad()
def validate(model, store: CaseStore, device) -> dict:
    model.eval()
    scores = []
    channels = []
    for i in range(len(store.ids)):
        geom = torch.from_numpy(store.geom[i])[None].to(device)
        fields = torch.from_numpy(store.fields[i]).to(device)
        mask = torch.from_numpy(store.mask[i]).to(device)
        window = [fields[j] for j in range(4)]
        preds = []
        for _k in range(4, fields.shape[0]):
            hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
            pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
            preds.append(pred)
            window.append(pred)
        pred_t = torch.stack(preds)
        err = per_frame_rel(pred_t, fields[4:], mask).cpu()
        mean = err.mean(dim=0)
        end = err[-1]
        scores.append(float(torch.maximum(mean / TARGET, end / TARGET).max()))
        channels.append(mean)
    mean_ch = torch.stack(channels).mean(dim=0)
    return {"R": float(np.mean(scores)), "ux": float(mean_ch[0]), "uy": float(mean_ch[1]), "p": float(mean_ch[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=["E", "F"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--fixed-k", type=int, default=None)
    parser.add_argument("--push", default=None, help="无梯度推进步数，逗号分隔，例如 32,64,96,128")
    parser.add_argument("--bp", type=int, default=8)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    init = args.init or F0
    saved = torch.load(init, map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    train_set = CaseStore("train")
    val_set = CaseStore("val")
    scales = torch.tensor(train_set.channel_scales(), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tag = args.tag or f"round2_{args.line}_seed{args.seed}"
    out = ROOT / "outputs" / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise SystemExit(f"{out} 已有权重")
    push_plan = [int(x) for x in args.push.split(",")] if args.push else None
    best = float("inf")
    history = []
    started = time.perf_counter()
    total = args.updates or sum(count for count, _width in SCHEDULE[args.line])
    print(f"[{tag}] {total} updates lr {args.lr} from {init}", flush=True)
    for step in range(1, total + 1):
        if push_plan:
            push = push_plan[min((step - 1) * len(push_plan) // total, len(push_plan) - 1)]
            width = push + args.bp
        else:
            push = 0
            width = args.fixed_k or unroll_of(args.line, step)
        model.train()
        pick = rng.choice(len(train_set.ids), size=4, replace=False)
        start = int(rng.integers(0, 161 - 4 - width))
        geom = torch.from_numpy(np.stack([train_set.geom[i] for i in pick])).to(device)
        fields = torch.from_numpy(np.stack([train_set.fields[i] for i in pick])).to(device)
        mask = torch.from_numpy(np.stack([train_set.mask[i] for i in pick])).to(device)
        opt.zero_grad(set_to_none=True)
        if push_plan:
            loss = truncated_loss(model, geom, fields, mask, scales, start, push, args.bp)
        else:
            loss = rollout_loss(model, geom, fields, mask, scales, start, width)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            (out / "status.txt").write_text(
                f"{args.line} update {step}/{total} K {width} loss {float(loss.detach()):.4e}\n", encoding="utf-8"
            )
        if step % 250 == 0 or step == total:
            metrics = validate(model, val_set, device)
            row = {"update": step, "K": width, "push": push, "loss": float(loss.detach()), "minutes": (time.perf_counter() - started) / 60, **metrics}
            history.append(row)
            print(
                f"[{args.line}] update {step} K {width} loss {row['loss']:.4e} val R {metrics['R']:.3f} "
                f"ux {metrics['ux']:.3f} uy {metrics['uy']:.3f} p {metrics['p']:.3f}",
                flush=True,
            )
            payload = {
                "model": model.state_dict(),
                "model_name": saved["model_name"],
                "in_channels": saved["in_channels"],
                "out_channels": saved["out_channels"],
                "line": tag,
                "seed": args.seed,
                "update": step,
                "history": history,
            }
            torch.save(payload, out / "last.pt")
            if metrics["R"] < best:
                best = metrics["R"]
                torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{args.line}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
