#!/usr/bin/env python3
"""F0 与 K=8 底座的同时域比较。Ri 只用整段平均和末端，时域最大误差另列。"""

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

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import TARGET, per_frame_rel  # noqa: E402

NAMES = ("ux", "uy", "p")
TAUS = TARGET.numpy()
MODELS = {
    "F0": ROOT / "outputs" / "checkpoints" / "fno_long" / "best.pt",
    "K8": ROOT / "outputs" / "checkpoints" / "fno_k8_base" / "base.pt",
}


def weight_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def decide(mean: np.ndarray, end: np.ndarray) -> dict:
    ratios = []
    for i, name in enumerate(NAMES):
        ratios.append((float(mean[i] / TAUS[i]), name, "整段平均"))
        ratios.append((float(end[i] / TAUS[i]), name, "末端"))
    value, channel, which = max(ratios)
    return {"R": value, "channel": channel, "which": which}


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "steps": int(rows[0]["steps"]),
        "seconds": float(rows[0]["seconds"]),
        "complete": bool(rows[0]["complete"]),
        "mean": np.mean([row["mean"] for row in rows], axis=0).tolist(),
        "end": np.mean([row["end"] for row in rows], axis=0).tolist(),
        "max_over_time_mean": np.mean([row["max"] for row in rows], axis=0).tolist(),
        "mean_R": float(np.mean([row["R"] for row in rows])),
        "max_R": float(np.max([row["R"] for row in rows])),
        "n_pass": int(sum(row["R"] < 1 for row in rows)),
    }


@torch.no_grad()
def rollout(model, store: CaseStore, index: int, device) -> tuple[np.ndarray, np.ndarray, float]:
    geom = torch.from_numpy(store.geom[index])[None].to(device)
    fields = torch.from_numpy(store.fields[index]).to(device)
    mask = torch.from_numpy(store.mask[index]).to(device)
    window = [fields[j] for j in range(4)]
    preds = []
    for _ in range(4, fields.shape[0]):
        hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
        pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
        preds.append(pred)
        window.append(pred)
    err = per_frame_rel(torch.stack(preds), fields[4:], mask).cpu().numpy()
    origin = float(store.times[index][3])
    future = np.asarray(store.times[index][4:], dtype=np.float64)
    return err, future, origin


def slices(err: np.ndarray, future: np.ndarray, origin: float) -> dict:
    specs = [("2s", 2.0), ("8s", 8.0), ("20s", 20.0), ("40s", 40.0), ("available", None)]
    out = {}
    for name, horizon in specs:
        if horizon is None:
            take = np.ones(future.shape[0], dtype=bool)
            complete = True
            seconds = float(future[-1] - origin)
        else:
            limit = origin + horizon
            take = future <= limit + 1e-4
            complete = bool(future[-1] + 1e-4 >= limit - 0.3) and bool(take.any())
        if not take.any():
            continue
        part = err[take]
        mean = part.mean(axis=0)
        end = part[-1]
        choice = decide(mean, end)
        out[name] = {
            "steps": int(take.sum()),
            "seconds": float(future[take][-1] - origin) if horizon is None else float(horizon),
            "t_end": float(future[take][-1]),
            "complete": complete,
            "mean": mean.tolist(),
            "end": end.tolist(),
            "max": part.max(axis=0).tolist(),
            "R": choice["R"],
            "channel": choice["channel"],
            "which": choice["which"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    store = CaseStore(args.split)
    report = {"split": args.split, "models": {}}
    loaded = {}
    for name, path in MODELS.items():
        saved = torch.load(path, map_location=device, weights_only=False)
        model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        loaded[name] = model
        cases = []
        for i, cid in enumerate(store.ids):
            err, future, origin = rollout(model, store, i, device)
            cases.append({"id": cid, "origin_t": origin, **{"horizons": slices(err, future, origin)}})
            print(name, cid, "available R", f"{cases[-1]['horizons']['available']['R']:.3f}", flush=True)
        flat = {}
        for horizon in cases[0]["horizons"]:
            rows = []
            for case in cases:
                row = {"id": case["id"], **case["horizons"][horizon]}
                rows.append(row)
            flat[horizon] = {"summary": summarize(rows), "cases": rows}
        report["models"][name] = {
            "ckpt": str(path.relative_to(ROOT)),
            "sha256": weight_sha256(saved["model"]),
            "horizons": flat,
        }
        for horizon, block in flat.items():
            s = block["summary"]
            print(
                f"{args.split} {name} {horizon} steps {s['steps']} complete {s['complete']} "
                f"uy {s['mean'][1]*100:.2f}% endUy {s['end'][1]*100:.2f}% "
                f"R {s['mean_R']:.3f} maxR {s['max_R']:.3f} pass {s['n_pass']}/{s['n']}",
                flush=True,
            )
    if args.time:
        model = loaded["K8"]
        torch.cuda.synchronize()
        for _ in range(3):
            rollout(model, store, 0, device)
        torch.cuda.synchronize()
        def time_steps(steps: int) -> float:
            geom = torch.from_numpy(store.geom[0])[None].to(device)
            fields = torch.from_numpy(store.fields[0]).to(device)
            def once():
                window = [fields[j] for j in range(4)]
                for _ in range(steps):
                    hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
                    pred = model(torch.cat([geom, hist[None]], dim=1))[0, :3]
                    window.append(pred)
            for _ in range(3):
                once()
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                once()
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)
            return float(np.median(samples))
        report["k8_autonomous_seconds"] = {
            "case": store.ids[0],
            "steps_32_about_8s": time_steps(32),
            "steps_157_available_window": time_steps(157),
            "includes": "GPU sync and network only, no disk",
        }
        print("K8 seconds", report["k8_autonomous_seconds"], flush=True)
    dest = OUT / f"horizon_{args.split}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
