"""第二轮共用指标。逐通道相对 L2，以及按 5%/5%/8% 归一化的算例分数 R。"""

from __future__ import annotations

import torch

TARGETS = {"ux": 0.05, "uy": 0.05, "p": 0.08}
TARGET = torch.tensor([0.05, 0.05, 0.08])


def per_frame_rel(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/true [T,3,H,W] 或 [3,H,W]，mask [H,W]。返回 [T,3]。"""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)
    fluid = mask.reshape(1, 1, *mask.shape[-2:])
    num = torch.linalg.vector_norm((pred - true) * fluid, dim=(2, 3))
    den = torch.linalg.vector_norm(true * fluid, dim=(2, 3)).clamp_min(1e-8)
    return num / den


def case_score(frame_err: torch.Tensor, end_index: int | None = None) -> dict:
    """frame_err [T,3]，已经是未来段。R 用整段均值和末端，对 0.05/0.05/0.08 取最大。"""
    end = frame_err[-1] if end_index is None else frame_err[end_index]
    mean = frame_err.mean(dim=0)
    ratio = torch.maximum(mean / TARGET.to(frame_err.device), end / TARGET.to(frame_err.device))
    return {
        "mean": [float(v) for v in mean],
        "end": [float(v) for v in end],
        "max": [float(v) for v in frame_err.max(dim=0).values],
        "R": float(ratio.max()),
    }


def horizon_index(times: torch.Tensor, origin: float, seconds: float) -> int | None:
    """最后一个不超过 origin+seconds 的下标。覆盖不到则返回 None。"""
    limit = origin + seconds
    ok = torch.nonzero(times <= limit + 1e-6, as_tuple=False)
    if ok.numel() == 0 or float(times[ok[-1]]) + 1e-6 < limit - 0.3 and seconds >= 20:
        # 20 秒以上必须真有这么长；短时域允许落到最近一帧，由调用方再判断覆盖
        pass
    if ok.numel() == 0:
        return None
    return int(ok[-1])
