#!/usr/bin/env python3
"""两天决策实验：基线、FNO 自主推演、周期统计量。

只读 data/processed/dataset_long.h5 和 outputs/checkpoints/fno_long/best.pt。
不改已有权重，不把测试集拿来选模型。

场误差只在训练集和 8 组验证集上比较。FNO 用四帧初始化，之后不再读真值帧。
复制和线性外推与 FNO 对齐同一个预测起点，并单独报它们自己的一帧/两帧预算。
统计量来自 HDF5 里保存的 CFD 壁面力（time, Cd, Cl），不是 64x64 场积分。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402

H5 = ROOT / "data" / "processed" / "dataset_long.h5"
CKPT = ROOT / "outputs" / "checkpoints" / "fno_long" / "best.pt"
OUT = ROOT / "outputs" / "decision_20260930"
HORIZONS_S = (1.0, 2.0, 4.0, 8.0, 40.0)
HISTORY = 4
CHANNELS = ("ux", "uy", "p")


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def load_split(handle: h5py.File) -> dict[str, list[str]]:
    return {
        name: [decode(item) for item in handle["splits"][name][()]]
        for name in ("train", "val")
    }


def steps_for(dt: float, seconds: float) -> int:
    return max(1, int(round(seconds / dt)))


def masked_rel(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    weight = mask.astype(np.float64)
    diff = (pred - truth) * weight
    base = truth * weight
    num = np.linalg.norm(diff)
    den = max(np.linalg.norm(base), 1e-8)
    return float(num / den)


def per_channel(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    out = {}
    for index, name in enumerate(CHANNELS):
        out[name] = masked_rel(pred[index], truth[index], mask)
    out["mean"] = float(np.mean([out[name] for name in CHANNELS]))
    return out


def copy_frame(history: np.ndarray, steps: int) -> np.ndarray:
    last = history[-1]
    return np.repeat(last[None], steps, axis=0)


def linear_extrap(history: np.ndarray, steps: int) -> np.ndarray:
    last = history[-1]
    prev = history[-2]
    slope = last - prev
    return np.stack([last + (i + 1) * slope for i in range(steps)], axis=0)


def load_fno(device: torch.device):
    pack = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = pack["config"]
    model = build_model(pack["model_name"], pack["in_channels"], pack["out_channels"], cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    return model


@torch.no_grad()
def fno_rollout(model, geom: np.ndarray, history: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    window = [torch.from_numpy(frame) for frame in history]
    preds = []
    geom_t = torch.from_numpy(geom).to(device)
    for _ in range(steps):
        hist = torch.cat(window[-HISTORY:], dim=0)
        x = torch.cat([geom_t, hist.to(device)], dim=0).unsqueeze(0)
        y = model(x)[0].detach().cpu()
        preds.append(y.numpy())
        window.append(y)
    return np.stack(preds, axis=0)


def evaluate_fields(device: torch.device) -> tuple[list[dict], dict]:
    model = load_fno(device)
    rows = []
    with h5py.File(H5, "r") as handle:
        splits = load_split(handle)
        for split, ids in splits.items():
            for cid in ids:
                group = handle["cases"][cid]
                fields = np.asarray(group["fields"], dtype=np.float32)
                times = np.asarray(group["times"], dtype=np.float64)
                geom = np.asarray(group["geom"], dtype=np.float32)
                mask = np.asarray(group["mask"], dtype=np.float32)
                dt = float(np.median(np.diff(times)))
                start = HISTORY
                truth = fields[start:]
                history = fields[:HISTORY]
                budgets = {
                    "persist_1frame": copy_frame(history[-1:], truth.shape[0]),
                    "linear_2frame": linear_extrap(history[-2:], truth.shape[0]),
                    "fno_4frame": fno_rollout(model, geom, history, truth.shape[0], device),
                }
                # 对齐四帧起点后，复制和线性也从同一帧出发。
                budgets["persist_aligned"] = copy_frame(fields[start - 1 : start], truth.shape[0])
                budgets["linear_aligned"] = linear_extrap(fields[start - 2 : start], truth.shape[0])
                for name, pred in budgets.items():
                    for seconds in HORIZONS_S:
                        step = steps_for(dt, seconds)
                        if step > pred.shape[0]:
                            continue
                        end = per_channel(pred[step - 1], truth[step - 1], mask)
                        window = [
                            per_channel(pred[i], truth[i], mask)["mean"]
                            for i in range(step)
                        ]
                        rows.append(
                            {
                                "split": split,
                                "case_id": cid,
                                "method": name,
                                "horizon_s": seconds,
                                "steps": step,
                                "dt": dt,
                                "init_frames": 4 if name.startswith("fno") else (2 if "linear" in name else 1),
                                "end_mean": end["mean"],
                                "end_ux": end["ux"],
                                "end_uy": end["uy"],
                                "end_p": end["p"],
                                "window_mean": float(np.mean(window)),
                            }
                        )
                print(f"field {split} {cid}", flush=True)
    summary = summarize_fields(rows)
    return rows, summary


def summarize_fields(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        key = (row["split"], row["method"], row["horizon_s"])
        bucket = out.setdefault(key, [])
        bucket.append(row)
    table = []
    for (split, method, horizon), items in sorted(out.items()):
        ends = np.array([item["end_mean"] for item in items])
        windows = np.array([item["window_mean"] for item in items])
        worst = max(items, key=lambda item: item["end_mean"])
        table.append(
            {
                "split": split,
                "method": method,
                "horizon_s": horizon,
                "n": len(items),
                "end_mean": float(ends.mean()),
                "end_median": float(np.median(ends)),
                "end_worst": float(ends.max()),
                "worst_case": worst["case_id"],
                "window_mean": float(windows.mean()),
            }
        )
    return {"rows": table}


def read_forces(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forces = np.asarray(group["forces"], dtype=np.float64)
    time = forces[:, 0]
    cd = forces[:, 1]
    cl = forces[:, 2]
    order = np.argsort(time)
    time, unique = np.unique(time[order], return_index=True)
    return time, cd[order][unique], cl[order][unique]


def strouhal(time: np.ndarray, cl: np.ndarray, diameter: float, u_inf: float = 1.0) -> float:
    signal = cl - cl.mean()
    if signal.size < 16 or np.allclose(signal, 0):
        return float("nan")
    dt = float(np.median(np.diff(time)))
    spectrum = np.fft.rfft(signal * np.hanning(signal.size))
    freq = np.fft.rfftfreq(signal.size, d=dt)
    if freq.size < 3:
        return float("nan")
    peak = 1 + int(np.argmax(np.abs(spectrum[1:])))
    return float(freq[peak] * diameter / u_inf)


def window_stats(time: np.ndarray, cd: np.ndarray, cl: np.ndarray, diameter: float) -> dict:
    return {
        "Cd_mean": float(cd.mean()),
        "Cl_rms": float(np.sqrt(np.mean(cl ** 2))),
        "St": strouhal(time, cl, diameter),
    }


def extract_stats() -> list[dict]:
    rows = []
    with h5py.File(H5, "r") as handle:
        splits = load_split(handle)
        role = {cid: name for name, ids in splits.items() for cid in ids}
        for cid in handle["cases"]:
            group = handle["cases"][cid]
            time, cd, cl = read_forces(group)
            diameter = float(group.attrs["D"])
            full = window_stats(time, cd, cl, diameter)
            # 后半段与整段比较，看统计量会不会被窗口晃动。
            half = time >= np.median(time)
            late = window_stats(time[half], cd[half], cl[half], diameter)
            rows.append(
                {
                    "split": role.get(cid, "test"),
                    "case_id": cid,
                    "shape": str(group.attrs["shape"]),
                    "Re": float(group.attrs["Re"]),
                    "cy": float(group.attrs["cy"]),
                    "D": diameter,
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"late_{k}": v for k, v in late.items()},
                    "d_Cd": abs(full["Cd_mean"] - late["Cd_mean"]),
                    "d_Cl_rms": abs(full["Cl_rms"] - late["Cl_rms"]),
                    "d_St": abs(full["St"] - late["St"]) if np.isfinite(full["St"]) and np.isfinite(late["St"]) else float("nan"),
                }
            )
    return rows


def features_of(row: dict) -> np.ndarray:
    shape = 1.0 if row["shape"] == "square" else 0.0
    return np.array([shape, row["Re"] / 200.0, row["cy"], row["D"]], dtype=np.float64)


TARGETS = ("full_Cd_mean", "full_Cl_rms", "full_St")


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std


def fit_mlp(x: np.ndarray, y: np.ndarray, seed: int = 42) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, y.shape[1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    for _ in range(800):
        opt.zero_grad()
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    return model


def rbf_interp(train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray) -> np.ndarray:
    # 简单径向基插值。点数少，直接解线性系统。
    dist = np.linalg.norm(train_x[:, None, :] - train_x[None, :, :], axis=-1)
    scale = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    kernel = np.exp(-(dist ** 2) / (2 * scale ** 2))
    kernel = kernel + np.eye(len(train_x)) * 1e-6
    weight = np.linalg.solve(kernel, train_y)
    qdist = np.linalg.norm(query[:, None, :] - train_x[None, :, :], axis=-1)
    qk = np.exp(-(qdist ** 2) / (2 * scale ** 2))
    return qk @ weight


def gaussian_process(train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray) -> np.ndarray:
    # 各目标独立的平方指数 GP，噪声在训练集上用留一挑一个尺度。
    preds = []
    dist = np.linalg.norm(train_x[:, None, :] - train_x[None, :, :], axis=-1)
    base = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    for column in range(train_y.shape[1]):
        y = train_y[:, column]
        best = None
        for length in (0.5 * base, base, 2 * base):
            for noise in (1e-4, 1e-3, 1e-2):
                kernel = np.exp(-(dist ** 2) / (2 * length ** 2)) + np.eye(len(y)) * noise
                try:
                    chol = np.linalg.cholesky(kernel)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
                # 近似留一：用拟合残差挑超参，避免把验证集用进调参。
                fitted = kernel @ alpha
                score = float(np.mean((fitted - y) ** 2))
                if best is None or score < best[0]:
                    best = (score, length, noise, alpha)
        _, length, noise, alpha = best
        qdist = np.linalg.norm(query[:, None, :] - train_x[None, :, :], axis=-1)
        qk = np.exp(-(qdist ** 2) / (2 * length ** 2))
        preds.append(qk @ alpha)
    return np.stack(preds, axis=1)


def metric_block(pred: np.ndarray, truth: np.ndarray) -> dict:
    out = {}
    for index, name in enumerate(TARGETS):
        err = np.abs(pred[:, index] - truth[:, index])
        scale = np.maximum(np.abs(truth[:, index]), 1e-8)
        out[name] = {
            "mae": float(err.mean()),
            "medae": float(np.median(err)),
            "mape": float(np.mean(err / scale)),
            "worst": float(err.max()),
        }
    return out


def fit_stats(rows: list[dict]) -> dict:
    train = [row for row in rows if row["split"] == "train" and np.isfinite(row["full_St"])]
    val = [row for row in rows if row["split"] == "val" and np.isfinite(row["full_St"])]
    x_train = np.stack([features_of(row) for row in train])
    y_train = np.stack([[row[name] for name in TARGETS] for row in train])
    x_val = np.stack([features_of(row) for row in val])
    y_val = np.stack([[row[name] for name in TARGETS] for row in val])
    x_train_n, x_val_n = standardize(x_train, x_val)
    y_mean = y_train.mean(axis=0)
    y_std = np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0))
    y_train_n = (y_train - y_mean) / y_std

    methods = {}
    mean_pred = np.repeat(y_train.mean(axis=0)[None], len(val), axis=0)
    methods["train_mean"] = metric_block(mean_pred, y_val)

    gp = gaussian_process(x_train_n, y_train, x_val_n)
    methods["gaussian_process"] = metric_block(gp, y_val)

    interp = rbf_interp(x_train_n, y_train, x_val_n)
    methods["rbf"] = metric_block(interp, y_val)

    mlp = fit_mlp(x_train_n, y_train_n)
    with torch.no_grad():
        pred_n = mlp(torch.from_numpy(x_val_n.astype(np.float32))).numpy()
    methods["mlp"] = metric_block(pred_n * y_std + y_mean, y_val)

    # 窗口波动相对验证集上候选差异。差异小于窗口波动时，不能声称能排序。
    spreads = {}
    for name in TARGETS:
        values = np.array([row[name] for row in val])
        short = name.removeprefix("full_")
        delta_key = {"Cd_mean": "d_Cd", "Cl_rms": "d_Cl_rms", "St": "d_St"}[short]
        windows = np.array([row[delta_key] for row in val])
        spreads[name] = {
            "val_std": float(values.std()),
            "val_range": float(values.max() - values.min()),
            "window_mae": float(np.mean(windows)),
            "window_max": float(np.max(windows)),
        }
    return {"n_train": len(train), "n_val": len(val), "methods": methods, "window_vs_spread": spreads}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if (OUT / "force_stats.csv").exists() and (OUT / "field_summary.json").exists():
        with (OUT / "force_stats.csv").open(encoding="utf-8") as handle:
            stats_rows = list(csv.DictReader(handle))
        for row in stats_rows:
            for key, value in list(row.items()):
                if key not in {"split", "case_id", "shape"}:
                    row[key] = float(value)
        field_summary = json.loads((OUT / "field_summary.json").read_text(encoding="utf-8"))
    else:
        field_rows, field_summary = evaluate_fields(device)
        write_csv(OUT / "field_rollout.csv", field_rows)
        (OUT / "field_summary.json").write_text(
            json.dumps(field_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        stats_rows = extract_stats()
        write_csv(OUT / "force_stats.csv", stats_rows)
    stats_fit = fit_stats(stats_rows)
    (OUT / "stats_fit.json").write_text(
        json.dumps(stats_fit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"field": field_summary["rows"], "stats": stats_fit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
