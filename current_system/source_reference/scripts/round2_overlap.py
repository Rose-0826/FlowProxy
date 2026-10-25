#!/usr/bin/env python3
"""比较 cases_long 与 cases_long40 在 160–200 秒的重叠段。不拼接。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flowproxy.build_dataset import _read_force_coeffs, case_to_arrays  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from decision_baselines import strouhal  # noqa: E402

OLD = ROOT / "data" / "processed" / "dataset_long.h5"
NEW = ROOT / "data" / "raw" / "cases_long40"
OUT = ROOT / "outputs" / "round2" / "overlap_160_200.json"


def rel(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> list[float]:
    out = []
    fluid = mask > 0.5
    for c in range(3):
        num = np.linalg.norm((a[c] - b[c])[fluid])
        den = np.linalg.norm(b[c][fluid]) + 1e-8
        out.append(float(num / den))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rest", action="store_true")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["cfd"]["skip_time"] = 160.0
    done = []
    if NEW.is_dir():
        for case in sorted(NEW.iterdir()):
            series = case / "VTK" / f"{case.name}.vtm.series"
            log = case / "log.foamToVTK"
            if series.is_file() and log.is_file() and "End" in log.read_text(encoding="utf-8", errors="ignore")[-400:]:
                done.append(case.name)
    rows = []
    with h5py.File(OLD, "r") as handle:
        role = {}
        for name in ("train", "val", "test"):
            for item in handle["splits"][name][()]:
                role[item.decode() if isinstance(item, bytes) else str(item)] = name
        already = set()
        if args.rest and OUT.is_file():
            old_doc = json.loads(OUT.read_text(encoding="utf-8"))
            rows.extend(old_doc.get("compared", []))
            already = {row["id"] for row in rows}
            if old_doc.get("val_case"):
                already.add(old_doc["val_case"]["id"])
        picked = [cid for cid in done if cid not in already and role.get(cid) in ("train", "val", "test")]
        if not args.rest:
            picked = picked[:4]
        for cid in picked:
            group = handle["cases"][cid]
            old_t = np.asarray(group["times"], dtype=np.float64)
            old_f = np.asarray(group["fields"], dtype=np.float32)
            mask = np.asarray(group["mask"], dtype=np.float32)
            new = case_to_arrays(cfg, NEW / cid)
            new_t = np.asarray(new["times"], dtype=np.float64)
            overlap = (new_t >= 160.0 - 1e-3) & (new_t <= 200.0 + 1e-3)
            nt = new_t[overlap]
            nf = new["fields"][overlap]
            matched = []
            for i, t in enumerate(old_t):
                j = int(np.argmin(np.abs(nt - t)))
                if abs(nt[j] - t) > 1e-3:
                    continue
                matched.append((i, j))
            first = []
            for i, j in matched[:4]:
                first.append({"t_old": float(old_t[i]), "t_new": float(nt[j]), "rel": rel(nf[j], old_f[i], mask)})
            field = np.mean([rel(nf[j], old_f[i], mask) for i, j in matched], axis=0)
            old_force = np.asarray(group["forces"], dtype=np.float64)
            new_force = _read_force_coeffs(NEW / cid)
            def window_force(force):
                keep = (force[:, 0] >= 160.0) & (force[:, 0] <= 200.0)
                return force[keep]
            old_w = window_force(old_force)
            new_w = window_force(np.asarray(new_force, dtype=np.float64))
            rows.append(
                {
                    "id": cid,
                    "split": role[cid],
                    "n_matched_frames": len(matched),
                    "old_t0": float(old_t[0]),
                    "new_t0": float(new_t[0]),
                    "new_t_last": float(new_t[-1]),
                    "dt_old": float(np.median(np.diff(old_t))),
                    "dt_new_overlap": float(np.median(np.diff(nt))) if nt.size > 2 else None,
                    "first_four": first,
                    "mean_rel_ux_uy_p": [float(x) for x in field],
                    "old_St": strouhal(old_w[:, 0], old_w[:, 2], float(group.attrs["D"])),
                    "new_St": strouhal(new_w[:, 0], new_w[:, 2], float(new["D"])),
                    "old_Cd": float(old_w[:, 1].mean()),
                    "new_Cd": float(new_w[:, 1].mean()),
                    "old_Cl_rms": float(np.sqrt(np.mean(old_w[:, 2] ** 2))),
                    "new_Cl_rms": float(np.sqrt(np.mean(new_w[:, 2] ** 2))),
                    "n_new_frames": int(new["times"].shape[0]),
                    "fields_sha256": hashlib.sha256(np.ascontiguousarray(new["fields"]).tobytes()).hexdigest(),
                    "times_sha256": hashlib.sha256(np.ascontiguousarray(new["times"]).tobytes()).hexdigest(),
                }
            )
            print(cid, rows[-1]["mean_rel_ux_uy_p"], "St", rows[-1]["old_St"], rows[-1]["new_St"], flush=True)
    OUT.write_text(json.dumps({"finished_cases_seen": len(done), "compared": rows, "spliced": False}, indent=2), encoding="utf-8")
    print("wrote", OUT, "finished", len(done))


if __name__ == "__main__":
    main()
