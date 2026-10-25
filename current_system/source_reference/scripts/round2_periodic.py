#!/usr/bin/env python3
"""周期模型。P 是共同预训练；A 从 P 继续做普通未来训练；D 从同一个 P 做启动—未来内循环。"""

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

from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402
from flowproxy.round2.periodic import (  # noqa: E402
    BETA,
    PeriodicModel,
    generate,
    variable_updates,
)

UPDATES = {"P": 2000, "A": 2000, "D": 2000}
VAL_EVERY = 250


def sample_indices(rng: np.random.Generator) -> tuple[int, int, np.ndarray]:
    length = int(rng.choice([4, 8, 16]))
    if rng.random() < 0.5:
        start = 0
    else:
        start = int(rng.integers(0, 161 - length - 8))
    future = np.arange(start + length, 161)
    pick = np.linspace(0, future.shape[0] - 1, 8).astype(int)
    query = np.unique(np.concatenate([future[pick], future[-1:]]))
    return length, start, query


def batch_from(store: CaseStore, length: int, start: int, query: np.ndarray, device, rng) -> dict:
    pick = rng.choice(len(store.ids), size=8, replace=False)
    geom = torch.from_numpy(np.stack([store.geom[i] for i in pick])).to(device)
    coord = torch.from_numpy(np.stack([store.coord[i] for i in pick])).to(device)
    mask = torch.from_numpy(np.stack([store.mask[i] for i in pick])).to(device)
    fields = torch.from_numpy(np.stack([store.fields[i] for i in pick])).to(device)
    times = torch.from_numpy(np.stack([store.times[i] for i in pick])).to(device)
    phase = torch.from_numpy(np.stack([store.phase[i] for i in pick])).to(device)
    omega = torch.tensor([store.omega[i] for i in pick], device=device)
    oracle = None
    if store.oracle[0] is not None:
        oracle = torch.from_numpy(np.stack([store.oracle[i] for i in pick])).to(device)
    end = start + length - 1
    tau = times[:, start : start + length] - times[:, end : end + 1]
    return {
        "geom": geom,
        "coord": coord,
        "mask": mask,
        "support": fields[:, start : start + length],
        "tau": tau,
        "query": fields[:, query],
        "query_tau": times[:, query] - times[:, end : end + 1],
        "theta_true": phase[:, end],
        "omega_true": omega,
        "oracle": oracle,
        "pick": pick,
    }


def future_loss(pred, truth, mask, scales) -> torch.Tensor:
    fluid = (mask > 0.5)[:, None, None]
    err = (pred - truth) / scales.reshape(1, 1, 3, 1, 1)
    denom = fluid.sum(dim=(3, 4)).clamp_min(1)
    per = (err.square() * fluid).sum(dim=(3, 4)) / denom / 3
    return 0.5 * per.mean() + 0.5 * per[:, -1].mean()


def startup_loss(pred, truth, mask, scales) -> torch.Tensor:
    return future_loss(pred, truth, mask, scales)


@torch.no_grad()
def validate(model, store: CaseStore, scales, device) -> dict:
    model.eval()
    scores = []
    channels = []
    for i, cid in enumerate(store.ids):
        geom = torch.from_numpy(store.geom[i])[None].to(device)
        coord = torch.from_numpy(store.coord[i])[None].to(device)
        mask = torch.from_numpy(store.mask[i]).to(device)
        fields = torch.from_numpy(store.fields[i]).to(device)
        times = torch.from_numpy(store.times[i]).to(device)
        tau = (times[:4] - times[3])[None]
        z, theta, nu = model.encode(geom, fields[:4][None], tau)
        coeff = model.coefficients(geom, coord, z)
        omega = model.omega(geom, nu)
        theta_t = theta + omega * (times[4:] - times[3])
        pred = generate(coeff, theta_t.unsqueeze(0))
        err = per_frame_rel(pred[0], fields[4:], mask).cpu()
        mean = err.mean(dim=0)
        end = err[-1]
        ratio = torch.maximum(mean / TARGET, end / TARGET)
        scores.append(float(ratio.max()))
        channels.append(mean)
    mean_ch = torch.stack(channels).mean(dim=0)
    return {"R": float(np.mean(scores)), "ux": float(mean_ch[0]), "uy": float(mean_ch[1]), "p": float(mean_ch[2]), "n": len(scores)}


def train(line: str, seed: int, init: Path | None) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    train_set = CaseStore("train")
    val_set = CaseStore("val")
    scales = torch.tensor(train_set.channel_scales(), device=device)
    model = PeriodicModel().to(device)
    if init is not None:
        saved = torch.load(init, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        print(f"[{line}] init {init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if init is None else 1e-4, weight_decay=1e-4)
    tag = f"round2_{line}_seed{seed}"
    out = ROOT / "outputs" / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise SystemExit(f"{out} 已有权重")
    best = float("inf")
    history = []
    updates = UPDATES[line]
    started = time.perf_counter()
    print(f"[{line}] {updates} updates seed {seed} beta {BETA}", flush=True)
    for step in range(1, updates + 1):
        model.train()
        length, start, query = sample_indices(rng)
        batch = batch_from(train_set, length, start, query, device, rng)
        z, theta, nu = model.encode(batch["geom"], batch["support"], batch["tau"])
        z0, theta0, nu0 = z, theta, nu
        if line == "D":
            z, theta, nu = variable_updates(
                model, batch["geom"], batch["coord"], batch["support"], batch["tau"], batch["mask"], scales,
                z, theta, nu, z0.detach(), nu0.detach(), 5, projected=True,
            )
        coeff = model.coefficients(batch["geom"], batch["coord"], z)
        omega = model.omega(batch["geom"], nu)
        pred_q = generate(coeff, theta[:, None] + omega[:, None] * batch["query_tau"])
        loss = future_loss(pred_q, batch["query"], batch["mask"], scales)
        pred_s = generate(coeff, theta[:, None] + omega[:, None] * batch["tau"])
        loss = loss + 0.1 * startup_loss(pred_s, batch["support"], batch["mask"], scales)
        loss = loss + 0.01 * (torch.log(omega.clamp_min(1e-3)) - torch.log(batch["omega_true"].clamp_min(1e-3))).square().mean()
        loss = loss + 0.1 * (1 - torch.cos(theta - batch["theta_true"])).mean()
        if batch["oracle"] is not None and line != "D":
            fluid = (batch["mask"] > 0.5)[:, None]
            anchored = (coeff - batch["oracle"]) / scales.reshape(1, 3, 1, 1).repeat(1, 9, 1, 1)
            loss = loss + 0.01 * (anchored.square() * fluid).sum() / fluid.sum().clamp_min(1) / coeff.shape[1]
        if line == "D":
            coeff0 = model.coefficients(batch["geom"], batch["coord"], z0)
            omega0 = model.omega(batch["geom"], nu0)
            direct = generate(coeff0, theta0[:, None] + omega0[:, None] * batch["query_tau"])
            loss = loss + 0.1 * future_loss(direct, batch["query"], batch["mask"], scales)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 1:
            (out / "status.txt").write_text(
                f"{line} seed {seed} update {step}/{updates} loss {float(loss.detach()):.4e}\n", encoding="utf-8"
            )
        if step % VAL_EVERY == 0 or step == updates:
            metrics = validate(model, val_set, scales, device)
            row = {"update": step, "loss": float(loss.detach()), "minutes": (time.perf_counter() - started) / 60, **metrics}
            history.append(row)
            print(
                f"[{line}] update {step} loss {row['loss']:.4e} val R {metrics['R']:.3f} "
                f"ux {metrics['ux']:.3f} uy {metrics['uy']:.3f} p {metrics['p']:.3f}",
                flush=True,
            )
            payload = {"model": model.state_dict(), "line": line, "seed": seed, "update": step, "history": history, "beta": BETA}
            torch.save(payload, out / "last.pt")
            if metrics["R"] < best:
                best = metrics["R"]
                torch.save(payload, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{line}] done -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=["P", "A", "D"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=None)
    args = parser.parse_args()
    init = args.init
    if args.line in {"A", "D"} and init is None:
        init = ROOT / "outputs" / "checkpoints" / "round2_P_seed0" / "best.pt"
    train(args.line, args.seed, init)


if __name__ == "__main__":
    main()
