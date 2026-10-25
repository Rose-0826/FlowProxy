#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.build_dataset import assemble_with_split, build_car_h5, build_from_runs, retime_h5  # noqa: E402
from flowproxy.dataset import save_cases_h5  # noqa: E402
from flowproxy.paths import load_config, resolve_under_root  # noqa: E402
from flowproxy.splits import default_split  # noqa: E402
from flowproxy.synthetic import make_synthetic_case  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--from-vtk", type=Path, default=None, help="OpenFOAM 算例根目录（含 VTK）")
    p.add_argument("--retime", type=Path, default=None, help="按 VTK 物理时间重排已有 h5，写出新文件")
    p.add_argument("--assemble", type=Path, default=None, help="已排好时间的圆柱 h5，其余算例从 VTK 补入")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--cars", type=Path, default=None, help="只把车形 VTK 收成单独的 h5")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.retime:
        out = args.out or (resolve_under_root(cfg, "data_processed") / "dataset_timefixed.h5")
        runs = resolve_under_root(cfg, "data_raw") / "cases"
        retime_h5(cfg, args.retime, out, runs)
        return
    if args.assemble:
        out = args.out or (resolve_under_root(cfg, "data_processed") / "dataset_120.h5")
        runs = resolve_under_root(cfg, "data_raw") / "cases"
        assemble_with_split(cfg, runs, args.assemble, out)
        return
    if args.cars:
        import yaml

        out = args.out or (resolve_under_root(cfg, "data_processed") / "dataset_car.h5")
        spec = yaml.safe_load((ROOT / "configs" / "car_cases.yaml").read_text(encoding="utf-8"))
        val_ids = [
            "car_Re80_yb0.2_H0.8_a45",
            "car_Re100_yb-0.4_H1_a70",
            "car_Re150_yb0.8_H1.2_a45",
            "car_Re200_yb0.2_H1_a30",
        ]
        build_car_h5(cfg, args.cars, out, list(spec["holdout"]), val_ids)
        return
    out = args.out or (resolve_under_root(cfg, "data_processed") / "dataset.h5")
    if args.synthetic:
        phys = cfg["physics"]
        times = np.linspace(float(cfg["cfd"]["skip_time"]), float(cfg["cfd"]["end_time"]), 32)
        cases = []
        for Re in phys["Re_list"]:
            for cy in phys["cy_offsets"]:
                for scale in phys["D_scales"]:
                    D = phys["D"] * scale
                    cid = f"{phys['shape']}_Re{Re:g}_cy{cy:g}_D{D:g}"
                    cases.append(
                        make_synthetic_case(cfg, cid, Re=float(Re), times=times, cx=phys["cx"], cy=float(cy), D=float(D), shape=phys["shape"])
                    )
        split = default_split([c["case_id"] for c in cases], cfg)
        save_cases_h5(out, cases, split)
        print(f"synthetic {len(cases)} -> {out}")
        return
    vtk_root = args.from_vtk or (resolve_under_root(cfg, "data_raw") / "cases")
    build_from_runs(cfg, vtk_root, out)


if __name__ == "__main__":
    main()
