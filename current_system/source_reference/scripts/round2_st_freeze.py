#!/usr/bin/env python3
"""用训练集上的径向基模型，先把 St 筛选预测写盘。已有 CFD 只放在核验文件里。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("decision_baselines", ROOT / "scripts" / "decision_baselines.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = ROOT / "outputs" / "round2"
BAND = (0.15, 0.17)
CANDIDATES = [
    {"shape": shape, "Re": 100.0, "cy": cy, "D": 1.0}
    for shape in ("cylinder", "square")
    for cy in (-0.4, 0.0, 0.4)
]


def distance(st: float) -> dict:
    low, high = BAND
    if st < low:
        gap = low - st
        side = "below"
    elif st > high:
        gap = st - high
        side = "above"
    else:
        gap = 0.0
        side = "inside"
    return {"gap_to_band": float(gap), "side": side, "near_edge": bool(side != "inside" and gap < 0.01)}


def period_scatter(time: np.ndarray, cl: np.ndarray, diameter: float) -> dict:
    signal = cl - np.mean(cl)
    sign = np.sign(signal)
    sign[sign == 0] = 1
    flips = np.where(np.diff(sign) > 0)[0]
    if flips.size < 4:
        return {"n_periods": int(max(flips.size - 1, 0)), "st_std": None}
    periods = np.diff(time[flips])
    st = diameter / np.maximum(periods, 1e-8)
    return {
        "n_periods": int(st.size),
        "st_mean": float(st.mean()),
        "st_std": float(st.std()),
        "st_min": float(st.min()),
        "st_max": float(st.max()),
    }


def main() -> None:
    rows = base.extract_stats()
    train = [row for row in rows if row["split"] == "train" and np.isfinite(row["full_St"])]
    x_train = np.stack([base.features_of(row) for row in train])
    y_train = np.stack([[row[name] for name in base.TARGETS] for row in train])
    queries = []
    for item in CANDIDATES:
        cid = f"{item['shape']}_Re100_cy{item['cy']:g}_D1"
        queries.append({**item, "id": cid})
    x_query = np.stack([base.features_of(item) for item in queries])
    _, x_query_n = base.standardize(x_train, x_query)
    x_train_n, _ = base.standardize(x_train, x_train)
    pred = base.rbf_interp(x_train_n, y_train, x_query_n)
    frozen = []
    for item, values in zip(queries, pred):
        st, cd, cl = float(values[2]), float(values[0]), float(values[1])
        frozen.append(
            {
                "id": item["id"],
                "model": "rbf_train80",
                "features": ["shape", "Re/200", "cy", "D"],
                "Cd_mean": cd,
                "Cl_rms": cl,
                "St": st,
                **distance(st),
            }
        )
    ranked = sorted(frozen, key=lambda row: (row["side"] == "inside", -row["gap_to_band"], row["Cl_rms"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank_avoid_then_small_cl"] = rank
    payload = {
        "frozen_before_cfd_check": True,
        "band": list(BAND),
        "note": "排序只用这个文件里的预测。贴着 0.15 或 0.17 的，要等周期波动和 CFD 再判断是否真的避开。",
        "candidates": ranked,
    }
    path = OUT / "st_screen_predictions.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path)
    known = {row["case_id"]: row for row in rows}
    check = []
    with h5py.File(base.H5, "r") as handle:
        for row in ranked:
            measured = known.get(row["id"])
            if measured is None:
                check.append({"id": row["id"], "existing_cfd": False})
                continue
            group = handle["cases"][row["id"]]
            time, _cd, cl = base.read_forces(group)
            late = time >= 160.0
            scatter = period_scatter(time[late], cl[late], 1.0)
            pred_gap = distance(row["St"])
            meas_gap = distance(float(measured["full_St"]))
            check.append(
                {
                    "id": row["id"],
                    "split": measured["split"],
                    "predicted_St": row["St"],
                    "measured_full_St": measured["full_St"],
                    "measured_late_St": measured["late_St"],
                    "predicted_side": pred_gap["side"],
                    "measured_side": meas_gap["side"],
                    "same_side": pred_gap["side"] == meas_gap["side"],
                    "period_scatter_t160": scatter,
                }
            )
    check_path = OUT / "st_screen_existing_cfd.json"
    check_path.write_text(json.dumps({"uses_existing_120_cases": True, "cases": check}, indent=2), encoding="utf-8")
    print("wrote", check_path)
    for row in check:
        print(row["id"], "pred", f"{row.get('predicted_St', float('nan')):.4f}", "meas", f"{row.get('measured_full_St', float('nan')):.4f}", row.get("measured_side"), "split", row.get("split"))


if __name__ == "__main__":
    main()
