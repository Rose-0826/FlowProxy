#!/usr/bin/env python3
"""四组热启动。monitor 只读已有力历史；是否停求解由本脚本决定。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "outputs"))

import warmstart_round1_setup as ws  # noqa: E402
from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from warmstart_iface import map_field, read_vectors, replace_internal  # noqa: E402

CFG_PATH = ROOT / "outputs" / "warmstart_round1" / "preregister.json"
H5 = ROOT / "data" / "processed" / "dataset_long.h5"
RUNS = ROOT / "data" / "raw" / "warmstart_pilot"
LOGS = ROOT / "outputs" / "warmstart_round1" / "runs"


def foam(case: Path, command: str) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PWD"] = str(case)
    script = f"source {ROOT / 'cfd' / 'env.sh'} && cd {case} && {command}"
    subprocess.run(["bash", "-lc", script], check=True, cwd=str(case), env=env)


def source_frame(donor: str) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(H5, "r") as handle:
        group = handle["cases"][donor]
        times = np.asarray(group["times"], dtype=np.float64)
        gap = np.abs(times - 180.0)
        picks = np.flatnonzero(np.isclose(gap, gap.min()))
        index = int(picks[np.argmin(times[picks])])
        fields = np.asarray(group["fields"][index], dtype=np.float64)
        mask = np.asarray(group["mask"], dtype=np.float64)
        return fields, mask, float(times[index])


def prepare_case(case_id: str) -> Path:
    src = ROOT / "data" / "raw" / "cases_long" / case_id
    dst = RUNS / case_id / "_mesh"
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(dst.parent / ".mesh.lock", "w", encoding="utf-8")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    if (dst / "constant" / "C").is_file():
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "system", dst / "system")
    shutil.copytree(src / "0.orig", dst / "0.orig")
    shutil.copytree(src / "constant", dst / "constant", ignore=shutil.ignore_patterns("polyMesh"))
    text = (dst / "system" / "controlDict").read_text(encoding="utf-8")
    text = re.sub(r"startFrom\s+\w+;", "startFrom       startTime;", text)
    text = re.sub(r"startTime\s+[^;]+;", "startTime       0;", text)
    text = re.sub(r"endTime\s+[^;]+;", "endTime         201;", text)
    text = re.sub(r"writeInterval\s+1;", "writeInterval   1;", text)
    text = text.replace("writeInterval   10;", "writeInterval   1;", 1)
    (dst / "system" / "controlDict").write_text(text, encoding="utf-8")
    foam(dst, "blockMesh && snappyHexMesh -overwrite && postProcess -func writeCellCentres")
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()
    for child in list(dst.iterdir()):
        if child.is_dir() and child.name not in {"0", "0.orig", "constant", "system"} and _time_name(child.name):
            shutil.rmtree(child)
    return dst


def _time_name(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def centers_of(mesh_case: Path) -> np.ndarray:
    return read_vectors(mesh_case / "constant" / "C")


def write_state(case: Path, mapped: np.ndarray) -> None:
    time_dir = case / "0"
    time_dir.mkdir(parents=True, exist_ok=True)
    u_tmpl = (case / "0.orig" / "U").read_text(encoding="utf-8")
    p_tmpl = (case / "0.orig" / "p").read_text(encoding="utf-8")
    lines = "\n".join(f"({row[0]:.8g} {row[1]:.8g} 0)" for row in mapped)
    u_body = f"internalField   nonuniform List<vector>\n{len(mapped)}\n(\n{lines}\n);\n"
    p_lines = "\n".join(f"{value:.8g}" for value in mapped[:, 2])
    p_body = f"internalField   nonuniform List<scalar>\n{len(mapped)}\n(\n{p_lines}\n);\n"
    (time_dir / "U").write_text(replace_internal(u_tmpl, u_body), encoding="utf-8")
    (time_dir / "p").write_text(replace_internal(p_tmpl, p_body), encoding="utf-8")


def uniform_field(case: Path, count: int, u_inf: float, pressure: float = 0.0) -> np.ndarray:
    out = np.zeros((count, 3), dtype=np.float64)
    out[:, 0] = u_inf
    out[:, 2] = pressure
    return out


def add_perturbation(values: np.ndarray, centers: np.ndarray, meta: dict) -> np.ndarray:
    dux, duy = ws.perturb_velocity(centers[:, 0], centers[:, 1], meta["U_inf"], meta["D"], meta["obstacle_center_x"], meta["cy"])
    out = values.copy()
    out[:, 0] += dux
    out[:, 1] += duy
    return out


def ai_snapshot(case_id: str, cfg: dict) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(cfg["initializers"]["ai"]["checkpoint_path"], map_location=device, weights_only=False)
    model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], load_config(ROOT / "configs" / "default.yaml")).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with h5py.File(H5, "r") as handle:
        geom = np.asarray(handle["cases"][case_id]["geom"], dtype=np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(geom)[None].to(device))[0, :3].cpu().numpy()
    raw = LOGS / case_id / "ai_raw_snapshot.npy"
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw, pred)
    return pred


def initial_field(case_id: str, method: str, centers: np.ndarray, cfg: dict) -> np.ndarray:
    meta = cfg["target_metadata"][case_id]
    with h5py.File(H5, "r") as handle:
        mask = np.asarray(handle["cases"][case_id]["mask"], dtype=np.float64)
    if method == "cold":
        return uniform_field(None, len(centers), meta["U_inf"])
    if method == "perturbed":
        return add_perturbation(uniform_field(None, len(centers), meta["U_inf"]), centers, meta)
    if method == "nearest":
        donor = cfg["initializers"]["nearest"]["selected_donors"][case_id]["donor_case_id"]
        fields, _mask, _t = source_frame(donor)
        mapped, _stats = map_field(centers, fields, _mask)
        return mapped
    if method == "ai":
        fields = ai_snapshot(case_id, cfg)
        mapped, _stats = map_field(centers, fields, mask)
        return add_perturbation(mapped, centers, meta)
    raise ValueError(method)


def read_forces(case: Path) -> np.ndarray:
    files = sorted((case / "postProcessing").glob("forceCoeffs*/**/coefficient.dat"))
    rows = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line and not line.startswith("#"):
                cols = line.split()
                if len(cols) >= 5:
                    rows.append((float(cols[0]), float(cols[1]), float(cols[4])))
    if not rows:
        return np.zeros((0, 3))
    data = np.asarray(rows, dtype=np.float64)
    _, index = np.unique(data[:, 0], return_index=True)
    return data[index]


def monitor_case(case_id: str, method: str, case: Path, cfg: dict) -> dict:
    forces = read_forces(case)
    if len(forces) == 0:
        return {"status": "collecting", "reason": "no_forces_yet"}
    tau = forces[:, 0] - forces[0, 0]
    payload = {
        "case_id": case_id,
        "parameters": {k: cfg["target_metadata"][case_id][k] for k in ("shape", "Re", "cy", "D", "U_inf")},
        "time": tau.tolist(),
        "Cd": forces[:, 1].tolist(),
        "Cl": forces[:, 2].tolist(),
    }
    dest = LOGS / case_id / method
    dest.mkdir(parents=True, exist_ok=True)
    history_path = dest / "force_prefix.json"
    decision_path = dest / "decision.json"
    history_path.write_text(json.dumps(payload), encoding="utf-8")
    decision = ws.monitor(cfg, payload)
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return decision


def run_solver(case: Path, case_id: str, method: str, cfg: dict, end_time: float, watch: bool) -> dict:
    text = (case / "system" / "controlDict").read_text(encoding="utf-8")
    text = re.sub(r"endTime\s+[^;]+;", f"endTime         {end_time:g};", text)
    (case / "system" / "controlDict").write_text(text, encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    log_path = case / "log.icoFoam"
    started = time.perf_counter()
    env["PWD"] = str(case)
    proc = subprocess.Popen(
        ["bash", "-lc", f"source {ROOT / 'cfd' / 'env.sh'} && cd {case} && icoFoam > log.icoFoam 2>&1"],
        cwd=str(case),
        env=env,
    )
    decision = {"status": "collecting"}
    while proc.poll() is None:
        time.sleep(2)
        if not watch:
            continue
        decision = monitor_case(case_id, method, case, cfg)
        if decision.get("status") == "online_confirmed":
            proc.terminate()
            break
    proc.wait()
    wall = time.perf_counter() - started
    log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    return {
        "returncode": proc.returncode,
        "fatal": "FOAM FATAL" in log,
        "ended": log.rstrip().endswith("End") or decision.get("status") == "online_confirmed",
        "wall_seconds": wall,
        "decision": decision,
        "finite_forces": bool(np.isfinite(read_forces(case)).all()) if len(read_forces(case)) else False,
    }


def stage_method(case_id: str, method: str, cfg: dict) -> Path:
    mesh = prepare_case(case_id)
    dst = RUNS / case_id / method
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(mesh, dst, ignore=shutil.ignore_patterns("log.*", "postProcessing", "VTK"))
    for child in list(dst.iterdir()):
        if child.is_dir() and _time_name(child.name) and child.name != "0":
            shutil.rmtree(child)
    centers = centers_of(dst)
    values = initial_field(case_id, method, centers, cfg)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{case_id} {method} 初值含非有限数")
    write_state(dst, values)
    return dst


def main() -> None:
    command = sys.argv[1]
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if command == "smoke":
        case_id = "square_Re80_cy-0.4_D0.9"
        report = {}
        for method in ("ai", "nearest"):
            case = stage_method(case_id, method, cfg)
            result = run_solver(case, case_id, method, cfg, end_time=2.0, watch=True)
            report[method] = result
            print(method, json.dumps(result), flush=True)
        (LOGS / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif command == "job":
        case_id, method = sys.argv[2], sys.argv[3]
        t0 = time.perf_counter()
        case = stage_method(case_id, method, cfg)
        prep = time.perf_counter() - t0
        result = run_solver(case, case_id, method, cfg, end_time=201.0, watch=True)
        result["prep_seconds"] = prep
        dest = LOGS / case_id / method / "result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(case_id, method, result.get("decision", {}).get("status"), flush=True)
    else:
        raise SystemExit("use smoke or job CASE METHOD")


if __name__ == "__main__":
    main()
