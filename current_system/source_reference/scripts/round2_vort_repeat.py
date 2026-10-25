#!/usr/bin/env python3
"""三对涡量复验。抽样序列预先写好，对照和涡量读同一份。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, case_score, per_frame_rel  # noqa: E402
from round2_spatial import (  # noqa: E402
    DX,
    DY,
    frame_loss,
    fluid_vorticity,
    step_forward,
    vorticity_loss,
)

BASE = ROOT / "outputs" / "checkpoints" / "fno_k8_base" / "base.pt"
OMEGA_RMS = 0.23727951508476705
SEEDS = (11, 22, 33)
VAL_AT = (0, 250, 500, 750, 1000)
UPDATES = 1000
N_TRAIN = 80


def weight_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_vorticity_sign() -> None:
    """Ux=-y、Uy=x 时，内部格点涡量应为 2。"""
    y = torch.linspace(-8.0, 8.0, 64)
    x = torch.linspace(-8.0, 20.0, 64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mask = torch.ones(1, 64, 64)
    omega, valid = fluid_vorticity((-yy)[None], xx[None], mask)
    err = float((omega[valid] - 2).abs().max())
    if err > 1e-4 or float(y[0]) != -8.0 or float(y[-1]) != 8.0:
        raise SystemExit(f"涡量符号检查失败，最大偏差 {err}")
    if abs(DX - 28 / 63) > 1e-12 or abs(DY - 16 / 63) > 1e-12:
        raise SystemExit("网格间距和 28/63、16/63 不一致")


def task_path(seed: int) -> Path:
    return OUT / f"rep_tasks_s{seed}.json"


def prepare() -> None:
    check_vorticity_sign()
    store = CaseStore("train")
    if len(store.ids) != N_TRAIN:
        raise SystemExit(f"训练集是 {len(store.ids)} 组，不是 80")
    saved = torch.load(BASE, map_location="cpu", weights_only=False)
    digest = weight_sha256(saved["model"])
    probe_rng = np.random.default_rng(9001)
    probe = []
    for _ in range(2):
        probe.append(
            {
                "cases": probe_rng.choice(N_TRAIN, size=4, replace=False).tolist(),
                "start": int(probe_rng.integers(0, 161 - 4 - 8)),
            }
        )
    (OUT / "rep_probe.json").write_text(
        json.dumps({"seed": 9001, "purpose": "只记梯度，不参与优化", "batches": probe}, indent=2),
        encoding="utf-8",
    )
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tasks = []
        for step in range(1, UPDATES + 1):
            tasks.append(
                {
                    "step": step,
                    "cases": rng.choice(N_TRAIN, size=4, replace=False).tolist(),
                    "start": int(rng.integers(0, 161 - 4 - 8)),
                }
            )
        payload = {
            "seed": seed,
            "n_train": N_TRAIN,
            "ids": store.ids,
            "augmentation": "none",
            "base_sha256": digest,
            "omega_rms": OMEGA_RMS,
            "tasks": tasks,
        }
        path = task_path(seed)
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(seed, file_sha256(path), "steps", len(tasks))
    gate = {
        "registered_before_runs": True,
        "base_mean_R": 2.54,
        "base_max_Ri": 4.38,
        "need_pairs_vort_better_than_control": 2,
        "need_vort_runs_below_base": 2,
        "selected_mean_R_below": 2.54,
        "selected_max_Ri_below": 4.38,
        "val_updates": list(VAL_AT),
        "update_0_is_candidate": True,
        "base_sha256": digest,
    }
    (OUT / "rep_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print("base", digest)
    print("vorticity sign ok", "row0", -8.0, "row increases y")


def pair_losses(model, geom, fields, mask, scales, start: int):
    window = [fields[:, start + j] for j in range(4)]
    field_terms = []
    vort_terms = []
    for k in range(8):
        pred = step_forward(model, None, geom, window)
        target = fields[:, start + 4 + k]
        field_terms.append(frame_loss(pred, target, mask, scales))
        vort_terms.append(vorticity_loss(pred, target, mask, OMEGA_RMS))
        window.append(pred)
    field_stack = torch.stack(field_terms)
    vort_stack = torch.stack(vort_terms)
    lf = 0.5 * field_stack.mean() + 0.5 * field_stack[-1]
    lw = 0.5 * vort_stack.mean() + 0.5 * vort_stack[-1]
    return lf, lw


def grad_norm(params) -> float:
    total = torch.zeros((), device="cuda")
    for param in params:
        if param.grad is not None:
            total = total + param.grad.detach().square().sum()
    return float(torch.sqrt(total))


@torch.no_grad()
def evaluate(model, store: CaseStore, device) -> dict:
    model.eval()
    rows = []
    for i, cid in enumerate(store.ids):
        geom = torch.from_numpy(store.geom[i])[None].to(device)
        fields = torch.from_numpy(store.fields[i]).to(device)
        mask = torch.from_numpy(store.mask[i])
        window = [fields[j][None] for j in range(4)]
        preds = []
        for _ in range(4, fields.shape[0]):
            pred = step_forward(model, None, geom, window)
            preds.append(pred[0].cpu())
            window.append(pred)
        err = per_frame_rel(torch.stack(preds), fields[4:].cpu(), mask)
        score = case_score(err)
        ratio = torch.maximum(err.mean(0) / TARGET, err[-1] / TARGET)
        channel_i = int(ratio.argmax())
        channel = ["ux", "uy", "p"][channel_i]
        which = "末端" if float(err[-1][channel_i]) >= float(err.mean(0)[channel_i]) else "整段"
        rows.append(
            {
                "id": cid,
                "mean": score["mean"],
                "end": score["end"],
                "max": score["max"],
                "R": score["R"],
                "channel": channel,
                "which": which,
            }
        )
    rs = [row["R"] for row in rows]
    model.train()
    return {
        "mean_R": float(np.mean(rs)),
        "max_R": float(np.max(rs)),
        "n_pass": int(sum(value < 1 for value in rs)),
        "mean_channels": [float(np.mean([row["mean"][c] for row in rows])) for c in range(3)],
        "end_channels": [float(np.mean([row["end"][c] for row in rows])) for c in range(3)],
        "cases": rows,
    }


def probe_grads(model, train_set, scales, device, batches) -> dict:
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    ratios = []
    model.train()
    for batch in batches:
        pick = batch["cases"]
        geom = torch.from_numpy(np.stack([train_set.geom[i] for i in pick])).to(device)
        fields = torch.from_numpy(np.stack([train_set.fields[i] for i in pick])).to(device)
        mask = torch.from_numpy(np.stack([train_set.mask[i] for i in pick])).to(device)
        lf, lw = pair_losses(model, geom, fields, mask, scales, batch["start"])
        model.zero_grad(set_to_none=True)
        lf.backward(retain_graph=True)
        field_norm = grad_norm(model.parameters())
        model.zero_grad(set_to_none=True)
        lw.backward()
        vort_norm = grad_norm(model.parameters())
        model.zero_grad(set_to_none=True)
        ratios.append({"field": field_norm, "vorticity": vort_norm, "ratio": vort_norm / max(field_norm, 1e-12)})
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "field": float(np.mean([row["field"] for row in ratios])),
        "vorticity": float(np.mean([row["vorticity"] for row in ratios])),
        "ratio": float(np.mean([row["ratio"] for row in ratios])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--line", choices=["control", "vorticity"])
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.seed is None or args.line is None:
        raise SystemExit("需要 --seed 和 --line，或 --prepare")
    check_vorticity_sign()
    lam = 0.0 if args.line == "control" else 1.0
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sampling = np.random.default_rng(args.seed)
    tasks_file = task_path(args.seed)
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    if tasks["seed"] != args.seed or tasks["augmentation"] != "none":
        raise SystemExit("任务序列和种子不一致")
    saved = torch.load(BASE, map_location=device, weights_only=False)
    digest = weight_sha256(saved["model"])
    if digest != tasks["base_sha256"]:
        raise SystemExit("底座权重哈希变了")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
    model.load_state_dict(saved["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    train_set = CaseStore("train")
    val_set = CaseStore("val")
    if train_set.ids != tasks["ids"]:
        raise SystemExit("训练集顺序和任务序列不一致")
    scales = torch.tensor(train_set.channel_scales(), device=device)
    probe = json.loads((OUT / "rep_probe.json").read_text(encoding="utf-8"))["batches"]
    tag = f"round2_rep_{args.line}_s{args.seed}"
    out = ROOT / "outputs" / "checkpoints" / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ckpt_0.pt").exists():
        raise SystemExit(f"{out} 已有第 0 次权重")
    group = opt.param_groups[0]
    config = {
        "tag": tag,
        "line": args.line,
        "lambda_omega": lam,
        "seed": args.seed,
        "updates_are_new": True,
        "not_resumed_from_vorticity_500": True,
        "base": str(BASE),
        "base_sha256": digest,
        "task_sequence": str(tasks_file),
        "task_sha256": file_sha256(tasks_file),
        "augmentation": "none",
        "K": 8,
        "batch": 4,
        "lr": group["lr"],
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "weight_decay": group["weight_decay"],
        "amsgrad": group["amsgrad"],
        "precision": "float32",
        "amp": False,
        "grad_clip": 1.0,
        "omega_rms": OMEGA_RMS,
        "omega_rms_recomputed": False,
        "dx": DX,
        "dy": DY,
        "velocity_scale": "raw_physical",
        "channel_scale_used_by": "field_loss_only",
        "y_row0": -8.0,
        "y_increases_with_row": True,
        "mask": "center and four neighbors, mask>0.5",
        "torch_seed_set": True,
        "numpy_sampling_from_file": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    block = []
    best = float("inf")
    best_update = None
    started = time.perf_counter()

    def save_ckpt(update: int, metrics: dict, grads: dict) -> None:
        nonlocal best, best_update
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "model_name": saved["model_name"],
            "in_channels": saved["in_channels"],
            "out_channels": saved["out_channels"],
            "update": update,
            "update_meaning": "new_updates_from_frozen_base",
            "lambda_omega": lam,
            "seed": args.seed,
            "numpy_state": sampling.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "config": config,
            "metrics": {k: v for k, v in metrics.items() if k != "cases"},
            "probe_grads": grads,
        }
        torch.save(payload, out / f"ckpt_{update}.pt")
        (out / f"eval_{update}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if metrics["mean_R"] < best:
            best = metrics["mean_R"]
            best_update = update
            torch.save(payload, out / "best.pt")

    def checkpoint(update: int) -> None:
        metrics = evaluate(model, val_set, device)
        grads = probe_grads(model, train_set, scales, device, probe)
        preceding = block[-250:]
        row = {
            "update": update,
            "minutes": (time.perf_counter() - started) / 60,
            "mean_R": metrics["mean_R"],
            "max_R": metrics["max_R"],
            "n_pass": metrics["n_pass"],
            "mean_uy": metrics["mean_channels"][1],
            "end_uy": metrics["end_channels"][1],
            "probe_grad_field": grads["field"],
            "probe_grad_vorticity": grads["vorticity"],
            "probe_grad_ratio": grads["ratio"],
        }
        if preceding:
            row["block_lf"] = float(np.mean([item["lf"] for item in preceding]))
            row["block_lw"] = float(np.mean([item["lw"] for item in preceding]))
            row["block_preclip"] = float(np.mean([item["preclip"] for item in preceding]))
            row["clip_hit_rate"] = float(np.mean([item["clipped"] for item in preceding]))
        history.append(row)
        save_ckpt(update, metrics, grads)
        print(
            f"[{tag}] update {update} meanR {metrics['mean_R']:.3f} maxR {metrics['max_R']:.3f} "
            f"pass {metrics['n_pass']} endUy {metrics['end_channels'][1]:.3f} "
            f"grad ratio {grads['ratio']:.3f}",
            flush=True,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[{tag}] lambda {lam} base {digest[:12]}", flush=True)
    checkpoint(0)
    model.train()
    for task in tasks["tasks"]:
        pick = task["cases"]
        geom = torch.from_numpy(np.stack([train_set.geom[i] for i in pick])).to(device)
        fields = torch.from_numpy(np.stack([train_set.fields[i] for i in pick])).to(device)
        mask = torch.from_numpy(np.stack([train_set.mask[i] for i in pick])).to(device)
        lf, lw = pair_losses(model, geom, fields, mask, scales, task["start"])
        loss = lf if lam == 0 else lf + lam * lw
        opt.zero_grad(set_to_none=True)
        loss.backward()
        preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        block.append(
            {
                "lf": float(lf.detach()),
                "lw": float(lw.detach()),
                "preclip": float(preclip),
                "clipped": float(preclip) > 1.0,
            }
        )
        if task["step"] in VAL_AT:
            checkpoint(task["step"])
    (out / "selection.json").write_text(
        json.dumps(
            {
                "best_update": best_update,
                "best_mean_R": best,
                "beat_frozen_base": best_update != 0 and best < 2.54,
                "selected_is_frozen_base": best_update == 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{tag}] selected update {best_update} meanR {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
