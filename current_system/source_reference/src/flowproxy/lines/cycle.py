"""涡街稳定周期的共用计算。只有函数，不存状态，两条线各自训练自己的周期网络。

周期网络吃 4 个几何通道，输出 N_OUT 个通道：
平均流 3，第 k 次谐波余弦 3、正弦 3（k = 1..HARMONICS），最后 1 个通道是频率。
相位约定：升力约等于 c + A cos(相位)，相位 = 2πf t − δ。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from flowproxy.dataset import read_lift
from flowproxy.splits import _meta_from_id, held_out_geometry

HARMONICS = 4
N_FIELD = 3 * (1 + 2 * HARMONICS)
N_OUT = N_FIELD + 1
F_REF = 0.16
LOGF_MIN = -1.6
LOGF_MAX = 1.2
FREQ_GRID = np.arange(0.05, 0.40, 2.5e-4)


def _fixed_freq_fit(t: np.ndarray, y: np.ndarray, f: float) -> tuple[float, float, float]:
    w = 2.0 * math.pi * f
    basis = np.stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    resid = float(np.square(basis @ coef - y).sum())
    total = float(np.square(y - y.mean()).sum()) + 1e-12
    return float(f), float(math.atan2(coef[1], coef[0])), 1.0 - resid / total


def sine_fit(t: np.ndarray, y: np.ndarray, freq: float | None = None) -> tuple[float, float, float]:
    """拟合 c + A cos(2πf t − δ)。给了 freq 就只解相位和振幅。返回 f、δ、拟合优度。"""
    if freq is not None:
        return _fixed_freq_fit(t, y, freq)
    coarse = max((_fixed_freq_fit(t, y, float(f)) for f in FREQ_GRID), key=lambda fit: fit[2])
    step = float(FREQ_GRID[1] - FREQ_GRID[0])
    fine = np.linspace(coarse[0] - step, coarse[0] + step, 81)
    return max((_fixed_freq_fit(t, y, float(f)) for f in fine), key=lambda fit: fit[2])


def lift_clock(case_dir: Path, field_times: np.ndarray) -> tuple[np.ndarray, float, float]:
    """训练标签：用整段窗口的升力拟合频率和相位。返回每帧相位、频率、拟合优度。"""
    t, cl = read_lift(case_dir)
    sel = (t >= float(field_times[0]) - 1e-6) & (t <= float(field_times[-1]) + 1e-6)
    f, delta, r2 = sine_fit(t[sel], cl[sel])
    phase = 2.0 * math.pi * f * np.asarray(field_times, dtype=np.float64) - delta
    return phase.astype(np.float32), f, r2


def window_phase(case_dir: Path, t0: float, width: float, freq: float | None) -> tuple[float, float]:
    """考试时只看开头 width 秒的升力。freq 为空时连频率一起拟合。返回 f、δ。"""
    t, cl = read_lift(case_dir)
    sel = (t >= t0 - 1e-6) & (t <= t0 + width + 1e-6)
    if int(sel.sum()) < 20:
        raise RuntimeError(f"{case_dir.name} 开头的升力太短")
    f, delta, _ = sine_fit(t[sel], cl[sel], freq)
    return f, delta


def cycle_field(raw: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    p = phase.reshape(-1, 1, 1, 1)
    out = raw[:, 0:3]
    for k in range(1, HARMONICS + 1):
        base = 3 + 6 * (k - 1)
        out = out + torch.cos(k * p) * raw[:, base : base + 3] + torch.sin(k * p) * raw[:, base + 3 : base + 6]
    return out


def _freq_logit(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fluid = mask.unsqueeze(1) if mask.ndim == 3 else mask
    head = raw[:, N_FIELD : N_FIELD + 1]
    return ((head * fluid).sum(dim=(2, 3)) / fluid.sum(dim=(2, 3)).clamp_min(1.0)).reshape(-1)


def cycle_freq(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F_REF * torch.exp(_freq_logit(raw, mask).clamp(LOGF_MIN, LOGF_MAX))


def freq_loss(raw: torch.Tensor, mask: torch.Tensor, freq_true: torch.Tensor) -> torch.Tensor:
    target = torch.log(freq_true.clamp_min(1e-3) / F_REF)
    return torch.square(_freq_logit(raw, mask) - target).mean()


def read_phase(raw: torch.Tensor, frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """用一帧的上下流速，在一次谐波上做最小二乘，读出相位。"""
    fluid = mask if mask.ndim == 3 else mask.squeeze(1)
    fluct = (frame[:, 1] - raw[:, 1]) * fluid
    c = raw[:, 4] * fluid
    s = raw[:, 7] * fluid
    cc = (c * c).sum(dim=(1, 2))
    ss = (s * s).sum(dim=(1, 2))
    cs = (c * s).sum(dim=(1, 2))
    fc = (fluct * c).sum(dim=(1, 2))
    fs = (fluct * s).sum(dim=(1, 2))
    det = cc * ss - cs * cs + 1e-6
    a = (ss * fc - cs * fs) / det
    b = (cc * fs - cs * fc) / det
    return torch.atan2(b, a)


def rel_frame(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target [N,C,H,W] 或 [C,H,W]，返回每帧每通道的相对 L2 [N,3]。"""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    fluid = mask.reshape(1, 1, *mask.shape[-2:])
    num = torch.linalg.vector_norm((pred - target) * fluid, dim=(2, 3))
    den = torch.linalg.vector_norm(target * fluid, dim=(2, 3)).clamp_min(1e-8)
    return num / den


def kind_of(cy: float, diameter: float) -> str:
    if held_out_geometry(cy, diameter) and abs(abs(cy) - 0.8) < 1e-6:
        return "far"
    if held_out_geometry(cy, diameter):
        return "extreme_d"
    return "other"


def group_keys(cid: str) -> list[str]:
    meta = _meta_from_id(cid)
    return ["all", f"shape:{meta['shape']}", f"Re:{int(meta['Re'])}", f"kind:{kind_of(meta['cy'], meta['D'])}"]


def summarize(buckets: dict[str, list[torch.Tensor]]) -> dict:
    out = {}
    for key, rows in buckets.items():
        stacked = torch.cat([r.reshape(-1, 3) for r in rows], dim=0)
        mean = stacked.mean(dim=0)
        out[key] = {
            "n": int(stacked.shape[0]),
            "rel_l2_ux": float(mean[0]),
            "rel_l2_uy": float(mean[1]),
            "rel_l2_p": float(mean[2]),
            "rel_l2_mean": float(mean.mean()),
        }
    return out


class CaseBank:
    """把一个划分的所有算例整段放进显存。模型很小，数据只有几百 MB。"""

    def __init__(self, h5_path: Path, split: str, cases_root: Path, device, with_labels: bool = True):
        import h5py

        from flowproxy.dataset import load_case_ids

        self.ids = load_case_ids(h5_path, split)
        geoms, masks, fields, phases, freqs, times, r2s = [], [], [], [], [], [], []
        with h5py.File(h5_path, "r") as handle:
            for cid in self.ids:
                group = handle["cases"][cid]
                t = np.asarray(group["times"][()], dtype=np.float64)
                geoms.append(np.asarray(group["geom"][()], dtype=np.float32))
                masks.append(np.asarray(group["mask"][()], dtype=np.float32))
                fields.append(np.asarray(group["fields"][()], dtype=np.float32))
                times.append(t)
                if with_labels:
                    ph, f, r2 = lift_clock(cases_root / cid, t)
                    phases.append(ph)
                    freqs.append(f)
                    r2s.append(r2)
        self.geom = torch.from_numpy(np.stack(geoms)).to(device)
        self.mask = torch.from_numpy(np.stack(masks)).to(device)
        self.fields = torch.from_numpy(np.stack(fields)).to(device)
        self.times = torch.from_numpy(np.stack(times)).to(device)
        if with_labels:
            self.phase = torch.from_numpy(np.stack(phases)).to(device)
            self.freq = torch.tensor(freqs, dtype=torch.float32, device=device)
            self.r2 = r2s

    def subset(self, index: list[int]) -> "CaseBank":
        """同一批显存张量里挑出几个算例，不重新读盘。"""
        part = object.__new__(CaseBank)
        pick = torch.as_tensor(index, device=self.geom.device)
        part.ids = [self.ids[i] for i in index]
        for name in ("geom", "mask", "fields", "times", "phase", "freq"):
            if hasattr(self, name):
                setattr(part, name, getattr(self, name)[pick])
        if hasattr(self, "r2"):
            part.r2 = [self.r2[i] for i in index]
        return part
