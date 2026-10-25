#!/usr/bin/env python3
"""把一个训练算例自己的 64×64 真值写回它的 OpenFOAM 网格，并短跑 icoFoam。

只检查接口。不读其他算例，也不把它算成初始化成绩。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASE_ID = "cylinder_Re100_cy0_D1"
SRC = ROOT / "data" / "raw" / "cases_long" / CASE_ID
DST = ROOT / "data" / "raw" / "warmstart_iface" / CASE_ID
H5 = ROOT / "data" / "processed" / "dataset_long.h5"
OUT = ROOT / "outputs" / "round2" / "warmstart_iface.json"
FRAME_T = 180.0
END_T = 182.0
DX = 28.0 / 63.0
DY = 16.0 / 63.0
NEAR_LIMIT = 1.5 * float(np.hypot(DX, DY))


def foam_run(case: Path, command: str) -> None:
    script = f"source {ROOT / 'cfd' / 'env.sh'} && cd {case} && {command}"
    subprocess.run(["bash", "-lc", script], check=True)


def copy_case() -> None:
    if DST.exists():
        return
    DST.mkdir(parents=True)
    shutil.copytree(SRC / "system", DST / "system")
    shutil.copytree(SRC / "0.orig", DST / "0.orig")
    shutil.copytree(SRC / "constant", DST / "constant", ignore=shutil.ignore_patterns("polyMesh"))
    text = (DST / "system" / "controlDict").read_text(encoding="utf-8")
    text = re.sub(r"startFrom\s+\w+;", "startFrom       startTime;", text)
    text = re.sub(r"startTime\s+[^;]+;", f"startTime       {FRAME_T:g};", text)
    text = re.sub(r"endTime\s+[^;]+;", f"endTime         {END_T:g};", text)
    text = re.sub(r"writeInterval\s+[^;]+;", "writeInterval   1;", text)
    (DST / "system" / "controlDict").write_text(text, encoding="utf-8")


def read_vectors(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"nonuniform\s+List<vector>\s+(\d+)\s*\(", text)
    if not match:
        raise RuntimeError(f"{path} 没有 nonuniform List<vector>")
    count = int(match.group(1))
    start = match.end()
    body = text[start:].split("\n)", 1)[0].replace("(", " ").replace(")", " ")
    nums = np.fromstring(body, sep=" ")
    if nums.size != count * 3:
        raise RuntimeError(f"{path} 向量数 {nums.size}，期望 {count * 3}")
    return nums.reshape(count, 3)


def load_frame() -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(H5, "r") as handle:
        group = handle["cases"][CASE_ID]
        times = np.asarray(group["times"], dtype=np.float64)
        index = int(np.argmin(np.abs(times - FRAME_T)))
        fields = np.asarray(group["fields"][index], dtype=np.float64)
        mask = np.asarray(group["mask"], dtype=np.float64)
    return fields, mask, float(times[index])


def map_field(centers: np.ndarray, fields: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    xs = np.linspace(-8.0, 20.0, fields.shape[2])
    ys = np.linspace(-8.0, 8.0, fields.shape[1])
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    fluid = mask > 0.5
    src = np.column_stack([xx[fluid], yy[fluid]])
    vals = np.column_stack([fields[c][fluid] for c in range(3)])
    linear = LinearNDInterpolator(src, vals, fill_value=np.nan)
    pred = linear(centers[:, 0], centers[:, 1])
    tree = cKDTree(src)
    dist, index = tree.query(centers[:, :2], k=1)
    missing = ~np.isfinite(pred).all(axis=1)
    pred[missing] = vals[index[missing]]
    weak = dist > NEAR_LIMIT
    return pred, {
        "source_fluid_nodes": int(fluid.sum()),
        "source_solid_nodes_excluded": int((~fluid).sum()),
        "cells": int(centers.shape[0]),
        "linear": int((~missing).sum()),
        "nearest_because_outside_hull": int(missing.sum()),
        "nearest_farther_than_1p5_cells": int(weak.sum()),
        "rule": "只使用 mask>0.5 的网格点。凸包内线性插值；包外用最近流体点，不用固体占位 0。",
    }


def write_fields(mapped: np.ndarray) -> None:
    time_dir = DST / f"{FRAME_T:g}"
    time_dir.mkdir(parents=True)
    u_tmpl = (DST / "0.orig" / "U").read_text(encoding="utf-8")
    p_tmpl = (DST / "0.orig" / "p").read_text(encoding="utf-8")
    lines = "\n".join(f"({row[0]:.8g} {row[1]:.8g} 0)" for row in mapped)
    u_body = f"internalField   nonuniform List<vector>\n{mapped.shape[0]}\n(\n{lines}\n);\n"
    p_lines = "\n".join(f"{row[2]:.8g}" for row in mapped)
    p_body = f"internalField   nonuniform List<scalar>\n{mapped.shape[0]}\n(\n{p_lines}\n);\n"
    (time_dir / "U").write_text(replace_internal(u_tmpl, u_body), encoding="utf-8")
    (time_dir / "p").write_text(replace_internal(p_tmpl, p_body), encoding="utf-8")


def replace_internal(template: str, body: str) -> str:
    start = template.find("internalField")
    end = template.find("boundaryField")
    if start < 0 or end < 0:
        raise RuntimeError("原边界文件缺少 internalField 或 boundaryField")
    return template[:start] + body + "\n" + template[end:]


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fatal = "FOAM FATAL" in text or "FOAM exiting" in text
    cont = re.findall(r"sum local = ([+\-0-9.eE]+), global = ([+\-0-9.eE]+)", text)
    courant = re.findall(r"Courant Number mean: ([+\-0-9.eE]+) max: ([+\-0-9.eE]+)", text)
    ended = text.rstrip().endswith("End")
    forces = sorted((DST / "postProcessing").glob("forceCoeffs*/**/coefficient.dat"))
    coeffs = []
    if forces:
        for line in forces[-1].read_text(encoding="utf-8", errors="ignore").splitlines():
            if line and not line.startswith("#"):
                cols = line.split()
                if len(cols) >= 5:
                    coeffs.append([float(cols[0]), float(cols[1]), float(cols[4])])
    coeff = np.asarray(coeffs, dtype=np.float64) if coeffs else np.zeros((0, 3))
    return {
        "fatal": fatal,
        "ended": ended,
        "n_steps": len(cont),
        "continuity_sum_local_first": float(cont[0][0]) if cont else None,
        "continuity_sum_local_last": float(cont[-1][0]) if cont else None,
        "continuity_global_last": float(cont[-1][1]) if cont else None,
        "courant_max_last": float(courant[-1][1]) if courant else None,
        "n_force_rows": int(coeff.shape[0]),
        "cd_last": float(coeff[-1, 1]) if coeff.size else None,
        "cl_last": float(coeff[-1, 2]) if coeff.size else None,
        "force_finite": bool(coeff.size and np.isfinite(coeff).all()),
    }


def main() -> None:
    copy_case()
    centers_path = DST / "constant" / "C"
    if not centers_path.is_file():
        foam_run(DST, "blockMesh && snappyHexMesh -overwrite && postProcess -func writeCellCentres")
    if not centers_path.is_file():
        found = list(DST.glob("**/C"))
        raise SystemExit(f"没有写到体心，看到 {found[:5]}")
    centers = read_vectors(centers_path)
    fields, mask, used_t = load_frame()
    mapped, stats = map_field(centers, fields, mask)
    if not np.isfinite(mapped).all():
        raise SystemExit("映射结果含有非有限值")
    write_fields(mapped)
    foam_run(DST, "icoFoam > log.icoFoam 2>&1")
    report = {
        "purpose": "接口检查。用训练算例自己的真值回写，不是 AI 初始化成绩。",
        "case_id": CASE_ID,
        "split": "train",
        "copy": str(DST.relative_to(ROOT)),
        "source_case_untouched": str(SRC.relative_to(ROOT)),
        "frame_t": used_t,
        "run_to": END_T,
        "mapping": stats,
        "solver": parse_log(DST / "log.icoFoam"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
