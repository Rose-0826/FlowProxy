"""第二轮数据协议：原 80/8/32、160–200 秒、通道尺度只由训练集拟合。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from flowproxy.dataset import read_lift
from flowproxy.lines.cycle import HARMONICS, sine_fit

ROOT = Path(__file__).resolve().parents[3]
H5 = ROOT / "data" / "processed" / "dataset_long.h5"
CASES = ROOT / "data" / "raw" / "cases_long"
SPLIT = ROOT / "configs" / "splits" / "body_120.json"
OUT = ROOT / "outputs" / "round2"


def load_split() -> dict[str, list[str]]:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def canonical_phase(times: np.ndarray, case_dir: Path) -> tuple[np.ndarray, float, float]:
    """升力基频的上升过零为相位 0。返回每帧相位、频率 Hz、拟合优度。"""
    t, cl = read_lift(case_dir)
    sel = (t >= float(times[0]) - 1e-6) & (t <= float(times[-1]) + 1e-6)
    freq, delta, r2 = sine_fit(t[sel], cl[sel])
    # 拟合是 A cos(2π f t − δ)。cos 在相位 −π/2 上升过零，所以规范相位再加 π/2。
    phase = 2.0 * math.pi * freq * times - delta + 0.5 * math.pi
    return phase.astype(np.float32), float(freq), float(r2)


def coordinate_channels(ny: int, nx: int, cy: float, diameter: float, cx: float = 0.0) -> np.ndarray:
    x = np.linspace(-8.0, 20.0, nx, dtype=np.float32)
    y = np.linspace(-8.0, 8.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xi = (xx - cx) / diameter
    eta = (yy - cy) / diameter
    wall = np.minimum(yy - (-8.0), 8.0 - yy) / diameter
    return np.stack([xi, eta, wall], axis=0).astype(np.float32)


def fit_harmonics(fields: np.ndarray, phase: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fields [T,3,H,W]。返回系数 [27,H,W] 和逐通道重建相对 L2。"""
    count = fields.shape[0]
    cols = [np.ones(count, dtype=np.float64)]
    theta = phase.astype(np.float64)
    for k in range(1, HARMONICS + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    design = np.stack(cols, axis=1)
    flat = fields.reshape(count, -1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, flat, rcond=None)
    recon = (design @ coef).reshape(fields.shape)
    coeff = coef.reshape(3 * design.shape[1], *fields.shape[-2:]).astype(np.float32)
    fluid = mask > 0.5
    rel = []
    for c in range(3):
        a = (recon[:, c] - fields[:, c])[:, fluid]
        b = fields[:, c][:, fluid]
        rel.append(float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-8)))
    return coeff, np.asarray(rel, dtype=np.float32)


class CaseStore:
    """训练集放在内存。测试划分不在这里加载，避免选型时碰到。"""

    def __init__(self, split: str):
        import h5py

        ids = load_split()[split]
        self.ids: list[str] = []
        self.geom = []
        self.coord = []
        self.mask = []
        self.fields = []
        self.times = []
        self.phase = []
        self.freq = []
        self.omega = []
        self.r2 = []
        self.oracle = []
        with h5py.File(H5, "r") as handle:
            for cid in ids:
                group = handle["cases"][cid]
                times = np.asarray(group["times"][()], dtype=np.float64)
                fields = np.asarray(group["fields"][()], dtype=np.float32)
                mask = np.asarray(group["mask"][()], dtype=np.float32)
                phase, freq, r2 = canonical_phase(times, CASES / cid)
                if split == "train":
                    coeff, _rel = fit_harmonics(fields, phase, mask)
                else:
                    coeff = None
                cy = float(group.attrs["cy"])
                diameter = float(group.attrs["D"])
                self.ids.append(cid)
                self.geom.append(np.asarray(group["geom"][()], dtype=np.float32))
                self.coord.append(coordinate_channels(mask.shape[0], mask.shape[1], cy, diameter, float(group.attrs.get("cx", 0.0))))
                self.mask.append(mask)
                self.fields.append(fields)
                self.times.append(times.astype(np.float32))
                self.phase.append(phase)
                self.freq.append(freq)
                self.omega.append(float(2.0 * math.pi * freq))
                self.r2.append(r2)
                self.oracle.append(coeff)

    def channel_scales(self) -> np.ndarray:
        if not self.ids:
            raise RuntimeError("通道尺度只能用训练集拟合")
        acc = np.zeros(3, dtype=np.float64)
        count = 0.0
        for fields, mask in zip(self.fields, self.mask):
            fluid = mask > 0.5
            for c in range(3):
                acc[c] += float(np.square(fields[:, c][:, fluid]).sum())
            count += float(fluid.sum() * fields.shape[0])
        return np.sqrt(acc / max(count, 1.0) + 1e-8).astype(np.float32)


def write_manifests() -> None:
    import h5py

    OUT.mkdir(parents=True, exist_ok=True)
    split = load_split()
    rows = []
    with h5py.File(H5, "r") as handle:
        for role, ids in split.items():
            for cid in ids:
                group = handle["cases"][cid]
                times = np.asarray(group["times"][()], dtype=np.float64)
                rows.append(
                    {
                        "id": cid,
                        "split": role,
                        "shape": str(group.attrs["shape"]),
                        "Re": float(group.attrs["Re"]),
                        "cy": float(group.attrs["cy"]),
                        "D": float(group.attrs["D"]),
                        "t0": float(times[0]),
                        "t1": float(times[-1]),
                        "n_frames": int(times.shape[0]),
                    }
                )
    (OUT / "split_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    train = CaseStore("train")
    scales = train.channel_scales()
    (OUT / "normalization.json").write_text(
        json.dumps({"channel_rms": scales.tolist(), "fit_on": "train_80", "frequency_beta": 0.3}, indent=2),
        encoding="utf-8",
    )
    audit = []
    for i, cid in enumerate(train.ids):
        _coeff, rel = fit_harmonics(train.fields[i], train.phase[i], train.mask[i])
        audit.append({"id": cid, "split": "train", "oracle_rel": rel.tolist(), "lift_r2": train.r2[i], "freq_hz": train.freq[i]})
    val = CaseStore("val")
    for i, cid in enumerate(val.ids):
        _coeff, rel = fit_harmonics(val.fields[i], val.phase[i], val.mask[i])
        audit.append({"id": cid, "split": "val", "oracle_rel": rel.tolist(), "lift_r2": val.r2[i], "freq_hz": val.freq[i]})
    (OUT / "representation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    timing = {
        "primary_support_frames": 4,
        "prediction_origin": "last_support_frame",
        "primary_start": "first_four_frames",
        "horizons_seconds": [1, 2, 4, 8, 20],
        "full_40_seconds": False,
        "reason": "序列只到 200 秒。首四帧末端约 160.75 秒，最长未来约 39 秒。",
    }
    (OUT / "timing_manifest.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"manifests -> {OUT}", flush=True)
