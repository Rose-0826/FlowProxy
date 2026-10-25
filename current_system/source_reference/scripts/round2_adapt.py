#!/usr/bin/env python3
"""在 A 的冻结权重上比较三种在线适配：联合、交替、交替加投影。只看 8 组验证。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402
from flowproxy.round2.periodic import (  # noqa: E402
    INNER_LR,
    PeriodicModel,
    generate,
    joint_support_loss,
    projected_spatial_loss,
    support_fields,
)

CHECKS = (0, 5, 20, 50)


def future_error(model, geom, coord, mask, fields, times, z, theta, nu):
    coeff = model.coefficients(geom, coord, z)
    omega = model.omega(geom, nu)
    theta_t = theta + omega * (times[4:] - times[3])
    pred = generate(coeff, theta_t.unsqueeze(0))
    return per_frame_rel(pred[0], fields[4:], mask).detach().cpu()


def adapt(model, geom, coord, support, tau, mask, scales, z, theta, nu, steps: int, policy: str):
    z0, nu0 = z.detach(), nu.detach()
    z = z.detach().clone().requires_grad_(True)
    theta = theta.detach().clone().requires_grad_(True)
    nu = nu.detach().clone().requires_grad_(True)
    for step in range(steps):
        pred, dtheta = support_fields(model, geom, coord, z, theta, nu, tau)
        phase_step = policy == "joint" or step % 2 == 0
        spatial_step = policy == "joint" or step % 2 == 1
        if phase_step:
            loss = joint_support_loss(pred, support, mask, scales, z, z0, nu, nu0)
            grads = torch.autograd.grad(loss, [theta, nu], retain_graph=spatial_step)
            theta = theta - INNER_LR["theta"] * grads[0].detach()
            nu = nu - INNER_LR["nu"] * grads[1].detach()
        if spatial_step:
            if policy == "projected":
                loss_z = projected_spatial_loss(pred, support, dtheta, tau, mask, scales, z, z0)
            else:
                loss_z = joint_support_loss(pred, support, mask, scales, z, z0, nu, nu0)
            grad_z, = torch.autograd.grad(loss_z, [z])
            z = z.detach() - INNER_LR["z"] * grad_z.detach()
            z.requires_grad_(True)
    return z.detach(), theta.detach(), nu.detach()


def main() -> None:
    device = torch.device("cuda")
    ckpt = ROOT / "outputs" / "checkpoints" / "round2_A_seed0" / "best.pt"
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model = PeriodicModel().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    store = CaseStore("val")
    scales = torch.tensor(store.channel_scales() if False else CaseStore("train").channel_scales(), device=device)
    report = {"ckpt": str(ckpt), "policies": {}}
    for policy in ("joint", "alternating", "projected"):
        rows = []
        for i, cid in enumerate(store.ids):
            geom = torch.from_numpy(store.geom[i])[None].to(device)
            coord = torch.from_numpy(store.coord[i])[None].to(device)
            mask = torch.from_numpy(store.mask[i]).to(device)
            fields = torch.from_numpy(store.fields[i]).to(device)
            times = torch.from_numpy(store.times[i]).to(device)
            tau = (times[:4] - times[3])[None]
            with torch.no_grad():
                z, theta, nu = model.encode(geom, fields[:4][None], tau)
            curve = {}
            for steps in CHECKS:
                zz, tt, nn = adapt(model, geom, coord, fields[:4][None], tau, mask[None], scales, z, theta, nu, steps, policy)
                err = future_error(model, geom, coord, mask, fields, times, zz, tt, nn)
                mean = err.mean(dim=0)
                end = err[-1]
                curve[str(steps)] = {
                    "ux": float(mean[0]),
                    "uy": float(mean[1]),
                    "p": float(mean[2]),
                    "end_uy": float(end[1]),
                    "R": float(torch.maximum(mean / TARGET, end / TARGET).max()),
                }
            rows.append({"id": cid, "curve": curve})
            print(f"{policy} {cid} R0 {curve['0']['R']:.3f} R50 {curve['50']['R']:.3f} uy50 {curve['50']['uy']:.3f}", flush=True)
        report["policies"][policy] = rows
    dest = OUT / "adapt_abc_val.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
