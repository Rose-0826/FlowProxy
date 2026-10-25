#!/usr/bin/env python3
"""形状均值基线，以及 120 组之外的 St 候选。先登记，再写预测。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decision_baselines", ROOT / "scripts" / "decision_baselines.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = ROOT / "outputs" / "round2"
BAND = (0.15, 0.17)
# 不使用已经预登记的 12 组确认例。cy=±0.6、Re=100、D=1 不在原 120 组里。
HOLDOUT = [
    {"shape": shape, "Re": 100.0, "cy": cy, "D": 1.0}
    for shape in ("cylinder", "square")
    for cy in (-0.6, 0.6)
]
EXISTING = [
    {"shape": shape, "Re": 100.0, "cy": cy, "D": 1.0}
    for shape in ("cylinder", "square")
    for cy in (-0.4, 0.0, 0.4)
]


def case_id(item: dict) -> str:
    return f"{item['shape']}_Re100_cy{item['cy']:g}_D1"


def side_of(st: float) -> str:
    if st < BAND[0]:
        return "below"
    if st > BAND[1]:
        return "above"
    return "inside"


def main() -> None:
    rows = base.extract_stats()
    known = {row["case_id"] for row in rows}
    for item in HOLDOUT:
        if case_id(item) in known:
            raise SystemExit(f"{case_id(item)} 已经在 120 组里")
    register = {
        "registered_before_prediction": True,
        "band": list(BAND),
        "not_the_12_confirm_cases": True,
        "cases": [
            {**item, "id": case_id(item), "role": "st_holdout_outside_120"}
            for item in HOLDOUT
        ],
    }
    (OUT / "st_holdout_preregister.json").write_text(json.dumps(register, indent=2), encoding="utf-8")
    train = [row for row in rows if row["split"] == "train" and np.isfinite(row["full_St"])]
    shape_mean = {}
    for shape in ("cylinder", "square"):
        values = [row["full_St"] for row in train if row["shape"] == shape]
        shape_mean[shape] = float(np.mean(values))
    x_train = np.stack([base.features_of(row) for row in train])
    y_train = np.stack([[row[name] for name in base.TARGETS] for row in train])
    x_train_n, _ = base.standardize(x_train, x_train)

    def predict(items):
        query = []
        for item in items:
            query.append({**item, "id": case_id(item)})
        x_query = np.stack([base.features_of(item) for item in query])
        _, x_query_n = base.standardize(x_train, x_query)
        rbf = base.rbf_interp(x_train_n, y_train, x_query_n)
        out = []
        for item, values in zip(query, rbf):
            st = float(values[2])
            rule = shape_mean[item["shape"]]
            out.append(
                {
                    "id": item["id"],
                    "rbf_St": st,
                    "rbf_Cd": float(values[0]),
                    "rbf_Cl_rms": float(values[1]),
                    "rbf_side": side_of(st),
                    "shape_mean_St": rule,
                    "shape_mean_side": side_of(rule),
                    "same_side_as_shape_rule": side_of(st) == side_of(rule),
                }
            )
        return out

    existing = predict(EXISTING)
    holdout = predict(HOLDOUT)
    payload = {
        "frozen_before_cfd": True,
        "band": list(BAND),
        "shape_mean_St": shape_mean,
        "existing_train_points": existing,
        "holdout_outside_120": holdout,
        "note": "形状均值只看圆或方。它和径向基在类别上一致时，模型没有额外的筛选信息。",
    }
    (OUT / "st_holdout_predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("shape mean", shape_mean)
    for row in existing + holdout:
        print(row["id"], "rbf", f"{row['rbf_St']:.4f}", row["rbf_side"], "rule", f"{row['shape_mean_St']:.4f}", row["shape_mean_side"], "same", row["same_side_as_shape_rule"])


if __name__ == "__main__":
    main()
