"""周期网络的训练循环。两条线各调一次，各存各的权重。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from flowproxy.lines.cycle import N_OUT, CaseBank, cycle_field, freq_loss
from flowproxy.lines.isolate import write_status
from flowproxy.losses import channel_weights, physics_bundle, supervised_loss
from flowproxy.models import build_model
from flowproxy.train import _format_bar

CASES_PER_STEP = 4
FRAMES_PER_CASE = 32
STEPS_PER_EPOCH = 100
LAMBDA_F = 10.0


def train_cycle_net(
    cfg: dict,
    bank: CaseBank,
    validate: Callable[[nn.Module], dict],
    out_dir: Path,
    label: str,
    epochs: int,
    patience: int,
    device,
    fixed: bool = False,
) -> tuple[nn.Module, list[dict]]:
    """fixed=True：训满 epochs 轮，学习率余弦下降，返回最后一轮权重，验证只记录不参与挑选。"""
    net = build_model("unetex", 4, N_OUT, cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    if fixed:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * STEPS_PER_EPOCH, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    lam_div = float(cfg["train"]["lambda_div"])
    lam_wall = float(cfg["train"]["lambda_wall"])
    weights = channel_weights(bank.fields[:, ::20].reshape(-1, 3, *bank.fields.shape[-2:]))
    n_cases, n_frames = bank.fields.shape[:2]
    best = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        net.train()
        started = time.perf_counter()
        running = 0.0
        for step in range(1, STEPS_PER_EPOCH + 1):
            pick = torch.randperm(n_cases, device=device)[:CASES_PER_STEP]
            frames = torch.randint(0, n_frames, (pick.shape[0], FRAMES_PER_CASE), device=device)
            geom = bank.geom[pick]
            mask = bank.mask[pick]
            raw = net(geom)
            rows = pick.unsqueeze(1).expand_as(frames)
            target = bank.fields[rows, frames].reshape(-1, 3, *bank.fields.shape[-2:])
            phase = bank.phase[rows, frames].reshape(-1)
            mask_rep = mask.repeat_interleave(FRAMES_PER_CASE, dim=0)
            pred = cycle_field(raw.repeat_interleave(FRAMES_PER_CASE, dim=0), phase)
            loss = supervised_loss(pred, target, weights)
            loss = loss + physics_bundle(pred, mask_rep, dx, dy, lam_div, lam_wall)["phys"]
            loss = loss + LAMBDA_F * freq_loss(raw, mask, bank.freq[pick])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if fixed:
                sched.step()
            running += float(loss.detach())
            if step == 1 or step % 20 == 0 or step == STEPS_PER_EPOCH:
                write_status(
                    out_dir,
                    _format_bar(f"{label} cycle ep{epoch}/{epochs}", step, STEPS_PER_EPOCH, time.perf_counter() - started, float(loss.detach())),
                )
        net.eval()
        with torch.no_grad():
            val = validate(net)
        score = val["rel_l2_mean"]
        if not fixed:
            sched.step(score)
        row = {"epoch": epoch, "train_loss": running / STEPS_PER_EPOCH, **val}
        history.append(row)
        print(
            f"[{label}] cycle epoch {epoch:03d}  train {row['train_loss']:.4e}  val relL2 {score:.4f}  "
            f"ux {val['rel_l2_ux']:.4f}  uy {val['rel_l2_uy']:.4f}  p {val['rel_l2_p']:.4f}",
            flush=True,
        )
        if fixed:
            continue
        if score < best:
            best = score
            stale = 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            print(f"[{label}] cycle early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history
