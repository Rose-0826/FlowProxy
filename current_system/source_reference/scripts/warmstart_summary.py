#!/usr/bin/env python3
"""从已有日志和场文件整理 16 行热启动明细。不重新求解。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "outputs"))

import warmstart_round1_setup as ws  # noqa: E402
from warmstart_iface import read_vectors  # noqa: E402

RUNS = ROOT / "data" / "raw" / "warmstart_pilot"
OUT_DIR = ROOT / "outputs" / "warmstart_round1"
H5 = ROOT / "data" / "processed" / "dataset_long.h5"
CHANNELS = ("Ux", "Uy", "p")


def read_scalar(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "nonuniform List<scalar>"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(path)
    rest = text[start + len(marker):]
    count = int(rest.split("(", 1)[0])
    body = rest.split("(", 1)[1].split("\n)", 1)[0]
    values = np.fromstring(body, sep=" ")
    if values.size != count:
        raise RuntimeError(f"{path} {values.size} != {count}")
    return values


def sample_times(case: Path) -> np.ndarray:
    times = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            times.append(float(child.name))
        except ValueError:
            continue
    return np.asarray(sorted(times), dtype=np.float64)


def to_grid(centers: np.ndarray, values: np.ndarray, xx: np.ndarray, yy: np.ndarray, tree: cKDTree) -> np.ndarray:
    linear = LinearNDInterpolator(centers, values, fill_value=np.nan)
    sampled = linear(xx, yy)
    missing = ~np.isfinite(sampled)
    if missing.any():
        _, index = tree.query(np.column_stack([xx[missing], yy[missing]]), k=1)
        sampled[missing] = values[index]
    return sampled


def load_run_frame(folder: Path, centers: np.ndarray, xx, yy, tree) -> np.ndarray:
    velocity = read_vectors(folder / "U")
    pressure = read_scalar(folder / "p")
    raw = np.column_stack([velocity[:, 0], velocity[:, 1], pressure])
    out = np.empty((3, xx.shape[0], xx.shape[1]), dtype=np.float64)
    for channel in range(3):
        out[channel] = to_grid(centers, raw[:, channel], xx, yy, tree)
    return out


def bracket(times: np.ndarray, instant: float) -> tuple[float, float, float]:
    if instant <= times[0]:
        return times[0], times[0], 0.0
    if instant >= times[-1]:
        return times[-1], times[-1], 0.0
    right = int(np.searchsorted(times, instant, side="left"))
    left = right - 1
    span = times[right] - times[left]
    weight = 0.0 if span == 0 else (instant - times[left]) / span
    return float(times[left]), float(times[right]), float(weight)


def frame_at(folders, times, instant, cache, centers, xx, yy, tree):
    left, right, weight = bracket(times, instant)
    for stamp in (left, right):
        if stamp not in cache:
            cache[stamp] = load_run_frame(folders[stamp], centers, xx, yy, tree)
    return (1.0 - weight) * cache[left] + weight * cache[right]


def h5_frame_at(times, fields, instant):
    left, right, weight = bracket(times, instant)
    ileft = int(np.argmin(np.abs(times - left)))
    iright = int(np.argmin(np.abs(times - right)))
    return (1.0 - weight) * fields[ileft] + weight * fields[iright]


def integrate_stats(samples: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = sorted(samples, key=lambda item: item[0])
    t = np.asarray([item[0] for item in samples], dtype=np.float64)
    q = np.stack([item[1] for item in samples], axis=0)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise RuntimeError("field samples are not increasing")
    trap = 0.5 * (q[1:] + q[:-1])
    duration = t[-1] - t[0]
    mean = np.sum(trap * dt[:, None, None, None], axis=0) / duration
    q2 = 0.5 * (q[1:] ** 2 + q[:-1] ** 2)
    second = np.sum(q2 * dt[:, None, None, None], axis=0) / duration
    fluct = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    rms = np.sqrt(np.maximum(second, 0.0))
    return mean, fluct, rms


def masked_rel(difference: np.ndarray, denom: np.ndarray, mask: np.ndarray) -> float | None:
    den = np.linalg.norm(denom[mask])
    if den < 1e-8:
        return None
    return float(np.linalg.norm(difference[mask]) / den)


def confirmation_window(prefix: dict, config: dict) -> tuple[float, float]:
    meta = ws.metadata(prefix)
    t, _cd, cl = ws.history(prefix)
    keep = t >= config["time"]["ignore_first_seconds_in_cycle_analysis"]
    t, cl = t[keep], cl[keep]
    filt = config["stability"]["resolved_filter"]
    factor = meta["D"] / meta["U_inf"]
    pts = ws.peaks(
        t, cl,
        filt["peak_min_separation_nd"] * factor,
        filt["peak_prominence_half_window_nd"] * factor,
        filt["peak_prominence_floor"],
    )
    if len(pts) < 5:
        raise RuntimeError("saved force prefix does not contain the confirmation block")
    return float(pts[-3]), float(pts[-1])


def reference_window(case_id: str, config: dict) -> tuple[float, float, dict]:
    filt = config["stability"]["resolved_filter"]
    with h5py.File(H5, "r") as handle:
        group = handle["cases"][case_id]
        force = np.asarray(group["forces"], dtype=np.float64)
        diameter = float(group.attrs["D"])
    order = np.argsort(force[:, 0])
    force = force[order]
    _, unique = np.unique(force[:, 0], return_index=True)
    force = force[unique]
    t, cd, cl = force[:, 0], force[:, 1], force[:, 2]
    keep = (t >= 160.0) & (t <= 200.0)
    t, cd, cl = t[keep], cd[keep], cl[keep]
    factor = diameter / 1.0
    pts = ws.peaks(
        t, cl,
        filt["peak_min_separation_nd"] * factor,
        filt["peak_prominence_half_window_nd"] * factor,
        filt["peak_prominence_floor"],
    )
    start, end = float(pts[-3]), float(pts[-1])
    block = ws.block_metrics(t, cd, cl, pts, len(pts) - 3, cycles=2)
    block["period"] /= factor
    block["St"] = 1.0 / block["period"]
    return start, end, block


def main() -> None:
    config = json.loads((OUT_DIR / "preregister.json").read_text(encoding="utf-8"))
    xs = np.linspace(-8.0, 20.0, 64)
    ys = np.linspace(-8.0, 8.0, 64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    rows = []
    references = {case_id: reference_window(case_id, config) for case_id in config["pilot_case_ids"]}
    with h5py.File(H5, "r") as handle:
        for case_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
            case_id = case_dir.name
            if case_id not in references:
                continue
            ref_start, ref_end, ref_block = references[case_id]
            group = handle["cases"][case_id]
            mask = np.asarray(group["mask"], dtype=np.float64) > 0.5
            ref_times = np.asarray(group["times"], dtype=np.float64)
            ref_fields = np.asarray(group["fields"], dtype=np.float64)
            inside = (ref_times >= ref_start) & (ref_times <= ref_end)
            ref_samples = [(float(ref_start), h5_frame_at(ref_times, ref_fields, ref_start))]
            ref_samples.extend((float(t), ref_fields[i]) for i, t in enumerate(ref_times) if inside[i] and not np.isclose(t, ref_start) and not np.isclose(t, ref_end))
            ref_samples.append((float(ref_end), h5_frame_at(ref_times, ref_fields, ref_end)))
            ref_mean, ref_fluct, ref_rms = integrate_stats(ref_samples)
            centers = read_vectors(case_dir / "_mesh" / "constant" / "C")[:, :2]
            tree = cKDTree(centers)
            for method in ("cold", "perturbed", "nearest", "ai"):
                result = json.loads((OUT_DIR / "runs" / case_id / method / "result.json").read_text(encoding="utf-8"))
                prefix = json.loads((OUT_DIR / "runs" / case_id / method / "force_prefix.json").read_text(encoding="utf-8"))
                decision = result["decision"]
                block = decision["blocks"][-1]
                start, end = confirmation_window(prefix, config)
                donor = ""
                source_time = ""
                if method == "nearest":
                    donor = config["initializers"]["nearest"]["selected_donors"][case_id]["donor_case_id"]
                    source_time = config["execution"]["actual_source_fields_and_timestamps"][case_id]["actual_source_time"]
                elif method == "ai":
                    donor = "unetex_long"
                folders = {}
                for child in (case_dir / method).iterdir():
                    if not child.is_dir():
                        continue
                    try:
                        folders[float(child.name)] = child
                    except ValueError:
                        continue
                saved = np.asarray(sorted(folders), dtype=np.float64)
                covered = bool(saved.size) and saved[0] <= start + 1e-8 and saved[-1] + 1e-8 >= end
                record = {
                    "case_id": case_id,
                    "initializer": method,
                    "checkpoint_or_donor": donor,
                    "source_time": source_time,
                    "frozen_config": "outputs/warmstart_round1/preregister.json",
                    "run_id": f"{case_id}/{method}",
                    "tau_stop": decision.get("evaluated_at_tau", ""),
                    "confirm_start": start,
                    "confirm_end": end,
                    "confirm_cycles": 2,
                    "init_wall_s": result.get("prep_seconds", ""),
                    "solver_wall_s": result.get("wall_seconds", ""),
                    "total_wall_s": (result.get("prep_seconds") or 0) + result.get("wall_seconds"),
                    "ref_Cd_mean": ref_block["Cd_mean"],
                    "run_Cd_mean": block["Cd_mean"],
                    "rel_Cd_mean": abs(block["Cd_mean"] - ref_block["Cd_mean"]) / abs(ref_block["Cd_mean"]),
                    "ref_Cl_fluctuation_RMS": ref_block["Cl_fluctuation_RMS"],
                    "run_Cl_fluctuation_RMS": block["Cl_fluctuation_RMS"],
                    "rel_Cl_fluctuation_RMS": abs(block["Cl_fluctuation_RMS"] - ref_block["Cl_fluctuation_RMS"]) / abs(ref_block["Cl_fluctuation_RMS"]),
                    "ref_St": ref_block["St"],
                    "run_St": 1.0 / block["period"],
                    "rel_St": abs(1.0 / block["period"] - ref_block["St"]) / abs(ref_block["St"]),
                    "field_status": "未核验",
                    "field_frames": "",
                }
                for name in CHANNELS:
                    record[f"mean_{name}"] = ""
                    record[f"fluct_{name}"] = ""
                if covered:
                    cache = {}
                    chosen = saved[(saved >= start) & (saved <= end)]
                    samples = [(start, frame_at(folders, saved, start, cache, centers, xx, yy, tree))]
                    for stamp in chosen:
                        if np.isclose(stamp, start) or np.isclose(stamp, end):
                            continue
                        samples.append((float(stamp), frame_at(folders, saved, float(stamp), cache, centers, xx, yy, tree)))
                    samples.append((end, frame_at(folders, saved, end, cache, centers, xx, yy, tree)))
                    mean, fluct, _rms = integrate_stats(samples)
                    unresolved = False
                    for index, name in enumerate(CHANNELS):
                        mean_err = masked_rel(mean[index] - ref_mean[index], ref_rms[index], mask)
                        fluct_err = masked_rel(fluct[index] - ref_fluct[index], ref_fluct[index], mask)
                        if mean_err is None or fluct_err is None:
                            unresolved = True
                            record[f"mean_{name}"] = ""
                            record[f"fluct_{name}"] = ""
                        else:
                            record[f"mean_{name}"] = mean_err
                            record[f"fluct_{name}"] = fluct_err
                    record["field_status"] = "分母过小未分辨" if unresolved else "已核验"
                    record["field_frames"] = len(samples)
                else:
                    record["field_status"] = f"未核验：保存场未覆盖 {start:.3f}–{end:.3f}"
                rows.append(record)
                print(case_id, method, record["field_status"], record["field_frames"], flush=True)
    dest = OUT_DIR / "warmstart_round1_summary.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", dest)


if __name__ == "__main__":
    main()
