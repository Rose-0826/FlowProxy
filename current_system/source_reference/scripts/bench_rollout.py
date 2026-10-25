#!/usr/bin/env python3
"""同一台机器上的求解耗时，以及上一帧模型自己连着猜的误差。"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.splits import _meta_from_id, held_out_geometry  # noqa: E402

CASES = ROOT / "data" / "raw" / "cases"
CARS = ROOT / "data" / "raw" / "cars"
H5_BODY = ROOT / "data" / "processed" / "dataset_120.h5"
CKPT = ROOT / "outputs" / "checkpoints" / "unetex_120_prev1" / "best.pt"
HORIZON_S = 20.0


def last_solver_time(log_path: Path) -> dict | None:
    exec_s = clock_s = None
    if not log_path.is_file():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"ExecutionTime = ([0-9.]+) s\s+ClockTime = ([0-9.]+) s", line)
        if match:
            exec_s = float(match.group(1))
            clock_s = float(match.group(2))
    if exec_s is None:
        return None
    return {"execution_s": exec_s, "clock_s": clock_s}


def mesh_time(case_dir: Path) -> float | None:
    log_path = case_dir / "log.snappy"
    if not log_path.is_file():
        log_path = case_dir / "log.Allrun"
    if not log_path.is_file():
        return None
    found = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"Finished meshing in = ([0-9.]+) s", line)
        if match:
            found = float(match.group(1))
    return found


def sample_cfd(root: Path, limit: int = 8) -> list[dict]:
    rows = []
    if not root.is_dir():
        return rows
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        solved = last_solver_time(case_dir / "log.icoFoam")
        if solved is None:
            continue
        rows.append({"id": case_dir.name, "mesh_s": mesh_time(case_dir), **solved})
        if len(rows) >= limit:
            break
    return rows


def kind_of(cy: float, diameter: float) -> str:
    if held_out_geometry(cy, diameter) and abs(abs(cy) - 0.8) < 1e-6:
        return "far"
    if held_out_geometry(cy, diameter):
        return "extreme_d"
    return "other"


def rel_one(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred * mask
    target = target * mask
    num = torch.linalg.vector_norm(pred - target, dim=(1, 2))
    den = torch.linalg.vector_norm(target, dim=(1, 2)).clamp_min(1e-8)
    return num / den


def load_model(path: Path, cfg: dict, device: torch.device):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model


def time_case(model, geom, fields, device: torch.device) -> dict:
    geom_t = torch.from_numpy(geom).to(device)
    frames = torch.from_numpy(fields).to(device)
    prev = frames[0]
    x = torch.cat([geom_t, prev], dim=0).unsqueeze(0)
    with torch.no_grad():
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for index in range(1, frames.shape[0]):
            prev = frames[index - 1]
            x = torch.cat([geom_t, prev], dim=0).unsqueeze(0)
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    steps = frames.shape[0] - 1
    return {"steps": steps, "seconds": elapsed, "seconds_per_step": elapsed / steps}


def rollout(model, geom, fields, times, mask, device: torch.device) -> list[torch.Tensor]:
    geom_t = torch.from_numpy(geom).to(device)
    truth = torch.from_numpy(fields).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    prev = truth[0]
    end = float(times[0]) + HORIZON_S
    rows = []
    with torch.no_grad():
        for index in range(1, truth.shape[0]):
            if float(times[index]) > end:
                break
            x = torch.cat([geom_t, prev], dim=0).unsqueeze(0)
            pred = model(x)[0, :3]
            rows.append(rel_one(pred, truth[index], mask_t).cpu())
            prev = pred
    return rows


def summarize(buckets: dict[str, list[torch.Tensor]]) -> dict:
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


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT, cfg, device)
    body = sample_cfd(CASES)
    cars = sample_cfd(CARS)
    solver = [row["clock_s"] for row in body + cars]
    timing = {"device": str(device), "cfd_body": body, "cfd_car": cars, "cfd_clock_mean_s": float(np.mean(solver))}

    with h5py.File(H5_BODY, "r") as handle:
        test_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["splits"]["test"][()]]
        demo_id = "cylinder_Re100_cy0_D1"
        group = handle["cases"][demo_id]
        timing["model_one_case"] = time_case(
            model,
            np.asarray(group["geom"][()], dtype=np.float32),
            np.asarray(group["fields"][()], dtype=np.float32),
            device,
        )
        timing["speedup_vs_solver_clock"] = timing["cfd_clock_mean_s"] / timing["model_one_case"]["seconds"]
        buckets: dict[str, list[torch.Tensor]] = {}
        last_rows: dict[str, list[torch.Tensor]] = {}
        for cid in test_ids:
            case = handle["cases"][cid]
            series = rollout(
                model,
                np.asarray(case["geom"][()], dtype=np.float32),
                np.asarray(case["fields"][()], dtype=np.float32),
                np.asarray(case["times"][()], dtype=np.float64),
                np.asarray(case["mask"][()], dtype=np.float32),
                device,
            )
            meta = _meta_from_id(cid)
            keys = ["all", f"shape:{meta['shape']}", f"kind:{kind_of(meta['cy'], meta['D'])}"]
            for key in keys:
                buckets.setdefault(key, []).extend(series)
                last_rows.setdefault(key, []).append(series[-1])
    dest = ROOT / "outputs" / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "speedup.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    report = {
        "model": "unetex_120_prev1",
        "horizon_s": HORIZON_S,
        "note": "从 t=40 的真值出发，之后每步用自己的预测当上一帧，连着猜 20 秒。",
        "mean_over_steps": summarize(buckets),
        "last_step": summarize(last_rows),
    }
    (dest / "rollout_120.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    step = timing["model_one_case"]
    print(
        f"cfd_clock_mean {timing['cfd_clock_mean_s']:.1f}s  "
        f"model {step['seconds']:.3f}s for {step['steps']} steps  "
        f"speedup {timing['speedup_vs_solver_clock']:.0f}x"
    )
    all_mean = report["mean_over_steps"]["all"]
    all_last = report["last_step"]["all"]
    print(
        f"rollout mean ux {all_mean['rel_l2_ux']*100:.1f}% uy {all_mean['rel_l2_uy']*100:.1f}% p {all_mean['rel_l2_p']*100:.1f}%"
    )
    print(
        f"rollout last ux {all_last['rel_l2_ux']*100:.1f}% uy {all_last['rel_l2_uy']*100:.1f}% p {all_last['rel_l2_p']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
