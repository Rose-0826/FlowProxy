from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .dataset import save_cases_h5
from .geometry import geometry_channels
from .interpolate import find_vtk_files, interpolate_points_to_grid, read_vtk_internal


def parse_case_meta(case_dir: Path) -> dict:
    meta_path = case_dir / "meta.json"
    if meta_path.exists():
        import json

        return json.loads(meta_path.read_text(encoding="utf-8"))
    name = case_dir.name
    # cylinder_Re100_cy0.0_D1.0
    parts = name.split("_")
    meta = {"shape": "cylinder", "Re": 100.0, "cx": 0.0, "cy": 0.0, "D": 1.0, "case_id": name}
    for p in parts:
        if p.startswith("Re"):
            meta["Re"] = float(p[2:])
        elif p.startswith("cy"):
            meta["cy"] = float(p[2:])
        elif p.startswith("D") and p[1:2].isdigit():
            meta["D"] = float(p[1:])
        elif p in {"cylinder", "square"}:
            meta["shape"] = p
    return meta


def _suffix(vtk: Path) -> int:
    match = re.search(r"_(\d+)$", vtk.parent.name)
    if not match:
        raise ValueError(f"VTK 上级目录没有步数编号: {vtk}")
    return int(match.group(1))


def time_map_from_series(case_dir: Path) -> dict[int, float]:
    """foamToVTK 的 *.vtm.series：目录编号 -> 物理时间（秒）。"""
    vtk_dir = case_dir / "VTK"
    if not vtk_dir.is_dir():
        return {}
    hits = sorted(vtk_dir.glob("*.vtm.series"))
    if not hits:
        return {}
    series = json.loads(hits[0].read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for item in series.get("files", []):
        match = re.search(r"_(\d+)\.vtm$", item["name"])
        if match:
            out[int(match.group(1))] = float(item["time"])
    return out


def time_map_from_log(case_dir: Path) -> dict[int, float]:
    """log.foamToVTK 里先写 Time，再写 internal.vtu 路径。VTK 目录缺失时用这个。"""
    log_path = case_dir / "log.foamToVTK"
    if not log_path.is_file():
        return {}
    out: dict[int, float] = {}
    pending: float | None = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Time:"):
            pending = float(line.split()[1])
            continue
        if pending is None or "internal.vtu" not in line:
            continue
        match = re.search(r"_(\d+)/internal\.vtu", line)
        if match:
            out[int(match.group(1))] = pending
        pending = None
    return out


def load_time_map(case_dir: Path) -> dict[int, float]:
    series = time_map_from_series(case_dir)
    logged = time_map_from_log(case_dir)
    if series and logged and series != logged:
        raise RuntimeError(f"{case_dir.name}: vtm.series 与 log.foamToVTK 的时间对不上")
    chosen = series or logged
    if not chosen:
        raise FileNotFoundError(f"{case_dir} 没有 vtm.series，也没有 log.foamToVTK")
    return chosen


def lex_internal_vtk(case_dir: Path, time_map: dict[int, float]) -> list[Path]:
    """与旧组包相同的路径字符串序。VTK 已删除时按日志编号把路径补回来。"""
    found = find_vtk_files(case_dir)
    internal = [path for path in found if "internal" in path.name.lower()]
    if internal:
        return internal
    rebuilt = [
        case_dir / "VTK" / f"{case_dir.name}_{suffix}" / "internal.vtu" for suffix in time_map
    ]
    return sorted(rebuilt)


def physical_time(vtk: Path, time_map: dict[int, float]) -> float:
    suffix = _suffix(vtk)
    if suffix not in time_map:
        raise KeyError(f"{vtk} 的编号 {suffix} 不在时间表里")
    return float(time_map[suffix])


def case_to_arrays(cfg: dict, case_dir: str | Path) -> dict:
    case_dir = Path(case_dir)
    meta = parse_case_meta(case_dir)
    files = find_vtk_files(case_dir)
    if not files:
        raise FileNotFoundError(f"no VTK in {case_dir}, run foamToVTK first")
    time_map = load_time_map(case_dir)
    # internal.vtu 的文件名里没有时间。秒数在上级目录编号对应的 vtm.series 里。
    timed = sorted((physical_time(vtk, time_map), vtk) for vtk in files if "internal" in vtk.name.lower())
    if not timed:
        raise FileNotFoundError(f"no internal VTK in {case_dir}")
    skip = float(cfg["cfd"].get("skip_time", 0.0))
    timed = [(t, vtk) for t, vtk in timed if t >= skip]
    if not timed:
        raise RuntimeError(f"{case_dir.name}: 没有 t>={skip} 的帧")
    geo = geometry_channels(
        cfg,
        shape=meta["shape"],
        cx=meta.get("cx", 0.0),
        cy=meta["cy"],
        D=meta["D"],
        Re=meta["Re"],
        backlight=meta.get("backlight"),
        length=meta.get("length"),
    )
    frames = []
    times = []
    for t, vtk in timed:
        pts, values = read_vtk_internal(vtk)
        field = interpolate_points_to_grid(pts, values, cfg, fill=0.0)
        field = field * geo["mask"][None, ...] if field.ndim == 3 else field
        # field: [3, ny, nx] but interpolate returns [C, ny, nx]
        frames.append(field)
        times.append(t)
    fields = np.stack(frames, axis=0).astype(np.float32)
    times_arr = np.asarray(times, dtype=np.float32)
    forces = _read_force_coeffs(case_dir)
    return {
        "case_id": meta.get("case_id", case_dir.name),
        "shape": meta["shape"],
        "Re": float(meta["Re"]),
        "cx": float(meta.get("cx", 0.0)),
        "cy": float(meta["cy"]),
        "D": float(meta["D"]),
        "times": times_arr,
        "geom": geo["geom"],
        "fields": fields,
        "mask": geo["mask"],
        **({"forces": forces} if forces is not None else {}),
    }


def _read_force_coeffs(case_dir: Path) -> np.ndarray | None:
    """time, Cd, Cl。coefficient.dat 的列是 time Cd Cd(f) Cd(r) Cl；续算的多段拼起来。"""
    hits = sorted(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient.dat"))
    if not hits:
        return None
    rows: dict[float, list[float]] = {}
    for hit in hits:
        for line in hit.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = [float(x) for x in line.split()]
            if len(cols) >= 5:
                rows[cols[0]] = [cols[0], cols[1], cols[4]]
    if not rows:
        return None
    return np.asarray([rows[t] for t in sorted(rows)], dtype=np.float32)


def frame_order_from_index_clock(
    case_dir: Path, old_times: np.ndarray, skip: float
) -> tuple[np.ndarray, np.ndarray]:
    """旧 h5 的 times 是路径排序后的序号。返回按物理秒排好、且 t>=skip 的帧下标。"""
    old_times = np.asarray(old_times, dtype=np.float64)
    expected = np.arange(float(old_times[0]), float(old_times[0]) + len(old_times), dtype=np.float64)
    if old_times.ndim != 1 or len(old_times) < 2 or not np.allclose(old_times, expected, atol=1e-4):
        raise RuntimeError(f"{case_dir.name}: times 不是连续序号，拒绝按旧顺序重排")
    time_map = load_time_map(case_dir)
    files = lex_internal_vtk(case_dir, time_map)
    indices = np.rint(old_times).astype(np.int64)
    if int(indices.min()) < 0 or int(indices.max()) >= len(files):
        raise RuntimeError(f"{case_dir.name}: 序号 {indices.min()}..{indices.max()} 超出 VTK 帧数 {len(files)}")
    phys = np.asarray([physical_time(files[int(i)], time_map) for i in indices], dtype=np.float64)
    all_phys = np.asarray([physical_time(path, time_map) for path in files], dtype=np.float64)
    keep = np.flatnonzero(phys >= skip)
    n_late = int(np.sum(all_phys >= skip))
    if n_late != len(keep):
        raise RuntimeError(f"{case_dir.name}: 旧序号丢掉了 t>={skip:g} 的帧（时间表 {n_late}，h5 里 {len(keep)}）")
    order = keep[np.argsort(phys[keep], kind="mergesort")]
    return order.astype(np.int64), phys[order]


def _check_physical_clock(case_id: str, times: np.ndarray, skip: float, end_time: float) -> None:
    times = np.asarray(times, dtype=np.float64)
    step = np.diff(times)
    if len(times) < 2 or np.any(step <= 1e-6):
        raise RuntimeError(f"{case_id}: 时间不是严格递增")
    if float(times[0]) + 1e-3 < skip:
        raise RuntimeError(f"{case_id}: 仍含 t<{skip:g} 的帧，首帧 {times[0]}")
    if abs(float(times[-1]) - end_time) > 0.3:
        raise RuntimeError(f"{case_id}: 末帧 {times[-1]} 不是结束时间 {end_time}")
    if abs(float(np.median(step)) - 0.25) > 0.05:
        raise RuntimeError(f"{case_id}: 步长中位数 {float(np.median(step)):.4f} 不是约 0.25 秒")


def retime_h5(cfg: dict, src_h5: str | Path, dst_h5: str | Path, runs_dir: str | Path) -> None:
    """用现有场，按 VTK 物理时间重排并丢掉启动段。不覆盖 dataset.h5。"""
    import h5py

    src_h5 = Path(src_h5)
    dst_h5 = Path(dst_h5)
    runs_dir = Path(runs_dir)
    if not src_h5.is_file():
        raise FileNotFoundError(src_h5)
    if dst_h5.resolve() == src_h5.resolve():
        raise RuntimeError(f"拒绝覆盖源文件 {src_h5}")
    if dst_h5.name in {"dataset.h5", "dataset.h5.bak"}:
        raise RuntimeError(f"拒绝写入 {dst_h5.name}，请换一个新文件名")
    skip = float(cfg["cfd"].get("skip_time", 0.0))
    end_time = float(cfg["cfd"].get("end_time", 80.0))
    cases = []
    with h5py.File(src_h5, "r") as handle:
        for cid in handle["cases"].keys():
            group = handle["cases"][cid]
            case_dir = runs_dir / cid
            order, times = frame_order_from_index_clock(
                case_dir, np.asarray(group["times"][()], dtype=np.float64), skip
            )
            _check_physical_clock(cid, times, skip, end_time)
            fields = np.asarray(group["fields"][()], dtype=np.float32)[order]
            case = {
                "case_id": cid,
                "shape": str(group.attrs["shape"]),
                "Re": float(group.attrs["Re"]),
                "cx": float(group.attrs["cx"]),
                "cy": float(group.attrs["cy"]),
                "D": float(group.attrs["D"]),
                "times": times.astype(np.float32),
                "geom": np.asarray(group["geom"][()], dtype=np.float32),
                "fields": fields,
                "mask": np.asarray(group["mask"][()], dtype=np.float32),
            }
            if "forces" in group:
                case["forces"] = np.asarray(group["forces"][()], dtype=np.float32)
            cases.append(case)
            print(
                f"retimed {cid}: {len(times)} frames, t={float(times[0]):.2f}..{float(times[-1]):.2f}",
                flush=True,
            )
    save_cases_h5(dst_h5, cases, None)
    print(f"wrote {len(cases)} cases -> {dst_h5}")


def cases_from_h5(path: str | Path) -> list[dict]:
    import h5py

    cases = []
    with h5py.File(path, "r") as handle:
        for cid in handle["cases"].keys():
            group = handle["cases"][cid]
            case = {
                "case_id": cid,
                "shape": str(group.attrs["shape"]),
                "Re": float(group.attrs["Re"]),
                "cx": float(group.attrs["cx"]),
                "cy": float(group.attrs["cy"]),
                "D": float(group.attrs["D"]),
                "times": np.asarray(group["times"][()], dtype=np.float32),
                "geom": np.asarray(group["geom"][()], dtype=np.float32),
                "fields": np.asarray(group["fields"][()], dtype=np.float32),
                "mask": np.asarray(group["mask"][()], dtype=np.float32),
            }
            if "forces" in group:
                case["forces"] = np.asarray(group["forces"][()], dtype=np.float32)
            cases.append(case)
    return cases


def assemble_with_split(
    cfg: dict,
    runs_dir: str | Path,
    base_h5: str | Path,
    out_h5: str | Path,
) -> dict[str, list[str]]:
    """旧圆柱沿用已经排好时间的场，新算例再插值。划分写进新文件，不覆盖 dataset.h5。"""
    from .splits import split_flow_cases, write_split

    runs_dir = Path(runs_dir)
    base_h5 = Path(base_h5)
    out_h5 = Path(out_h5)
    if out_h5.name in {"dataset.h5", "dataset.h5.bak"} or out_h5.resolve() == base_h5.resolve():
        raise RuntimeError(f"拒绝写入 {out_h5}")
    skip = float(cfg["cfd"].get("skip_time", 0.0))
    end_time = float(cfg["cfd"].get("end_time", 80.0))
    cases = cases_from_h5(base_h5)
    have = {case["case_id"] for case in cases}
    for case in cases:
        _check_physical_clock(case["case_id"], case["times"], skip, end_time)
    print(f"kept {len(cases)} retimed cases from {base_h5.name}", flush=True)
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or child.name in have:
            continue
        case = case_to_arrays(cfg, child)
        _check_physical_clock(case["case_id"], case["times"], skip, end_time)
        cases.append(case)
        print(
            f"ingested {case['case_id']}: {len(case['times'])} frames, "
            f"t={float(case['times'][0]):.2f}..{float(case['times'][-1]):.2f}",
            flush=True,
        )
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("算例编号重复")
    split = split_flow_cases(ids)
    save_cases_h5(out_h5, cases, split)
    write_split(out_h5.with_suffix(".splits.json"), split)
    counts = {name: len(rows) for name, rows in split.items()}
    print(f"wrote {len(cases)} cases {counts} -> {out_h5}", flush=True)
    return split


def build_car_h5(
    cfg: dict,
    cars_dir: str | Path,
    out_h5: str | Path,
    holdout: list[str],
    val_ids: list[str],
) -> dict[str, list[str]]:
    """只收车形。圆和方的 h5 不动。点名的车进测试，另外几辆进验证。"""
    from .splits import write_split

    cars_dir = Path(cars_dir)
    out_h5 = Path(out_h5)
    blocked = {"dataset.h5", "dataset.h5.bak", "dataset_timefixed.h5", "dataset_120.h5"}
    if out_h5.name in blocked:
        raise RuntimeError(f"拒绝写入 {out_h5}")
    skip = float(cfg["cfd"].get("skip_time", 0.0))
    end_time = float(cfg["cfd"].get("end_time", 80.0))
    cases = []
    for child in sorted(p for p in cars_dir.iterdir() if p.is_dir() and p.name.startswith("car_")):
        case = case_to_arrays(cfg, child)
        _check_physical_clock(case["case_id"], case["times"], skip, end_time)
        cases.append(case)
        print(
            f"ingested {case['case_id']}: {len(case['times'])} frames, "
            f"t={float(case['times'][0]):.2f}..{float(case['times'][-1]):.2f}",
            flush=True,
        )
    ids = [case["case_id"] for case in cases]
    missing = [cid for cid in list(holdout) + list(val_ids) if cid not in ids]
    if missing:
        raise RuntimeError("车形划分缺组: " + ", ".join(missing))
    hold = set(holdout)
    val = set(val_ids)
    if hold & val:
        raise RuntimeError("同一辆车不能既考试又验证")
    split = {"train": [], "val": [], "test": []}
    for cid in sorted(ids):
        if cid in hold:
            split["test"].append(cid)
        elif cid in val:
            split["val"].append(cid)
        else:
            split["train"].append(cid)
    save_cases_h5(out_h5, cases, split)
    write_split(out_h5.with_suffix(".splits.json"), split)
    counts = {name: len(rows) for name, rows in split.items()}
    print(f"wrote {len(cases)} cars {counts} -> {out_h5}", flush=True)
    return split


def build_from_runs(cfg: dict, runs_dir: str | Path, out_h5: str | Path, split: dict[str, list[str]] | None = None) -> None:
    runs_dir = Path(runs_dir)
    cases = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "VTK").exists() and not list(child.glob("VTK/**/*")):
            continue
        try:
            cases.append(case_to_arrays(cfg, child))
            print(f"ingested {child.name}", flush=True)
        except Exception as exc:
            print(f"skip {child.name}: {exc}", flush=True)
    if not cases:
        raise RuntimeError(f"no valid VTK cases in {runs_dir}")
    save_cases_h5(out_h5, cases, split)
    print(f"wrote {len(cases)} cases -> {out_h5}")
