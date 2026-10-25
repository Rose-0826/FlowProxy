from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def default_split(case_ids: list[str], cfg: dict, leftover_to: str = "test") -> dict[str, list[str]]:
    holdout_re = {float(x) for x in cfg["data"].get("holdout_re", [])}
    hold_geo = set(cfg["data"].get("holdout_geometry", []))
    test, rest = [], []
    for cid in case_ids:
        meta = _meta_from_id(cid)
        geo_key = f"{meta['shape']}_cy{meta['cy']}_D{meta['D']}"
        if meta["Re"] in holdout_re or geo_key in hold_geo:
            test.append(cid)
        else:
            rest.append(cid)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(rest)
    n_train = int(len(rest) * float(cfg["data"]["train_ratio"]))
    n_val = int(len(rest) * float(cfg["data"]["val_ratio"]))
    train = rest[:n_train]
    val = rest[n_train : n_train + n_val]
    leftover = rest[n_train + n_val :]
    if leftover_to == "train":
        train.extend(leftover)
    else:
        test.extend(leftover)
    if not val and train:
        val = [train.pop()]
    if not test and val:
        test = [val[-1]]
    return {"train": train, "val": val, "test": test}


def _meta_from_id(cid: str) -> dict:
    meta = {"shape": "cylinder", "Re": 100.0, "cy": 0.0, "D": 1.0}
    for p in cid.split("_"):
        if p.startswith("Re"):
            try:
                meta["Re"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("cy"):
            try:
                meta["cy"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("D"):
            try:
                meta["D"] = float(p[1:])
            except ValueError:
                pass
        elif p in {"cylinder", "square"}:
            meta["shape"] = p
    return meta


# 验证集是点名的 8 组，每个流速一根圆、一根方。新图形不会撞上这些名字。
VAL_CASE_IDS = (
    "cylinder_Re80_cy0_D1.1",
    "square_Re80_cy-0.4_D0.9",
    "cylinder_Re100_cy0.4_D0.9",
    "square_Re100_cy0_D1.1",
    "cylinder_Re150_cy-0.4_D1",
    "square_Re150_cy0.4_D1",
    "cylinder_Re200_cy0_D0.9",
    "square_Re200_cy0.4_D1.1",
)


def _near(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-6


def held_out_geometry(cy: float, D: float) -> bool:
    """没见过的几何：离中线最远，或最细、最粗。形状不限，以后的新图形沿用同一条。"""
    far = (_near(cy, 0.8) or _near(cy, -0.8)) and _near(D, 1.0)
    extreme_d = _near(cy, 0.0) and (_near(D, 0.8) or _near(D, 1.2))
    return far or extreme_d


def assign_role(case_id: str) -> str:
    """整组进训练、验证或测试。验证优先于外推测试，其余进训练。"""
    if case_id in VAL_CASE_IDS:
        return "val"
    meta = _meta_from_id(case_id)
    if held_out_geometry(meta["cy"], meta["D"]):
        return "test"
    return "train"


def split_flow_cases(case_ids: list[str]) -> dict[str, list[str]]:
    """按位置和直径划分。同一规则适用于圆柱、方柱以及以后同样命名的新图形。"""
    out = {"train": [], "val": [], "test": []}
    for cid in sorted(case_ids):
        out[assign_role(cid)].append(cid)
    missing = [cid for cid in VAL_CASE_IDS if cid not in out["val"]]
    if missing:
        raise RuntimeError("验证集缺组: " + ", ".join(missing))
    return out


def write_split(path: str | Path, split: dict[str, list[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")
