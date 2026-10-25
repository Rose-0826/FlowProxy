#!/usr/bin/env python3
"""延长试算的上限：同一个算例拿升力相位做谐波最小二乘，看最好能到多少相对 L2。

每个算例自己拟合自己的模态，所以这是任何“形状给模态、升力给相位”的模型的上限，不是考试分数。
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.build_dataset import case_to_arrays  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402


def lift_all(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    for path in sorted(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient.dat")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 5:
                times.append(float(cols[0]))
                values.append(float(cols[4]))
    t, index = np.unique(np.asarray(times), return_index=True)
    return t, np.asarray(values)[index]


def lift_angle(t: np.ndarray, cl: np.ndarray, t_from: float) -> tuple[np.ndarray, np.ndarray]:
    sel = t >= t_from
    tt = t[sel]
    x = cl[sel] - cl[sel].mean()
    n = x.shape[0]
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    return tt, np.unwrap(np.angle(np.fft.ifft(np.fft.fft(x) * h)))


def rel(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[None, None]
    num = np.sqrt((((pred - target) * m) ** 2).sum(axis=(2, 3)))
    den = np.sqrt(((target * m) ** 2).sum(axis=(2, 3))) + 1e-8
    return (num / den).mean(axis=0)


def oracle(fields: np.ndarray, times: np.ndarray, mask: np.ndarray, t: np.ndarray, cl: np.ndarray, t_from: float) -> dict:
    tt, angle = lift_angle(t, cl, t_from)
    theta = np.interp(times, tt, angle)
    count = fields.shape[0]
    flat = fields.reshape(count, -1).astype(np.float64)
    out = {}
    for harmonics in (0, 2, 4):
        cols = [np.ones(count)] + [fn(k * theta) for k in range(1, harmonics + 1) for fn in (np.cos, np.sin)]
        design = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(design, flat, rcond=None)
        out[harmonics] = rel((design @ coef).reshape(fields.shape), fields, mask)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--from-time", type=float, default=160.0)
    args = parser.parse_args()
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
    cfg["cfd"]["skip_time"] = float(args.from_time)
    for case_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        arrays = case_to_arrays(cfg, case_dir)
        t, cl = lift_all(case_dir)
        late = oracle(arrays["fields"], arrays["times"].astype(np.float64), arrays["mask"], t, cl, args.from_time - 60.0)
        cid = case_dir.name
        source = ROOT / "data" / "processed" / ("dataset_car.h5" if cid.startswith("car_") else "dataset_120.h5")
        raw_dir = ROOT / "data" / "raw" / ("cars" if cid.startswith("car_") else "cases") / cid
        with h5py.File(source, "r") as handle:
            group = handle["cases"][cid]
            early = oracle(
                np.asarray(group["fields"][()]),
                np.asarray(group["times"][()], dtype=np.float64),
                np.asarray(group["mask"][()]),
                *lift_all(raw_dir),
                30.0,
            )
        span = arrays["times"]
        print(f"== {cid}  frames {len(span)} t {span[0]:.2f}-{span[-1]:.2f}  Cl std {cl[t >= args.from_time].std():.3f}")
        for label, table in (("40-80 s  ", early), (f"{args.from_time:g}-{span[-1]:g} s", late)):
            row = "  ".join(
                f"h{k}: ux {v[0]*100:4.1f}% uy {v[1]*100:5.1f}% p {v[2]*100:5.1f}%" for k, v in table.items()
            )
            print(f"   {label}  {row}")


if __name__ == "__main__":
    main()
