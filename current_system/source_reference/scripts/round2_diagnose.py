#!/usr/bin/env python3
"""周期路线的四项诊断。这些数不是预测成绩。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.round2.data import OUT, CaseStore, fit_harmonics  # noqa: E402
from flowproxy.round2.metrics import per_frame_rel  # noqa: E402
from flowproxy.round2.periodic import PeriodicModel, generate  # noqa: E402


def rel_np(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> list[float]:
    out = []
    fluid = mask > 0.5
    for c in range(3):
        a = (pred[:, c] - true[:, c])[:, fluid]
        b = true[:, c][:, fluid]
        out.append(float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-8)))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "outputs" / "checkpoints" / "round2_A_seed0" / "best.pt"
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model = PeriodicModel().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    rows = []
    for split in ("train", "val"):
        store = CaseStore(split)
        for i, cid in enumerate(store.ids):
            fields = store.fields[i]
            phase = store.phase[i]
            mask = store.mask[i]
            times = store.times[i]
            coeff, oracle_rel = fit_harmonics(fields, phase, mask)
            geom = torch.from_numpy(store.geom[i])[None].to(device)
            coord = torch.from_numpy(store.coord[i])[None].to(device)
            tau = torch.from_numpy((times[:4] - times[3])[None])
            frames = torch.from_numpy(fields[:4])[None].to(device)
            with torch.no_grad():
                z, theta, nu = model.encode(geom, frames, tau.to(device))
                pred_coeff = model.coefficients(geom, coord, z)[0].cpu().numpy()
                omega = float(model.omega(geom, nu)[0])
            # 预测模态配真实未来相位
            theta_true = torch.from_numpy(phase).to(device)
            with torch.no_grad():
                recon = generate(torch.from_numpy(pred_coeff)[None].to(device), theta_true.unsqueeze(0))[0].cpu().numpy()
            mode_rel = rel_np(recon, fields, mask)
            # 真实模态配启动估计的相位和频率。θ(t)=θs+ω(t-ts)
            theta_hat = float(theta[0]) + omega * (times - times[3])
            design_phase = theta_hat.astype(np.float32)
            count = fields.shape[0]
            cols = [np.ones(count)]
            for k in range(1, 5):
                cols.append(np.cos(k * design_phase))
                cols.append(np.sin(k * design_phase))
            # 用真实系数沿估计相位重建：把 oracle 系数按估计相位生成
            theta_hat_t = torch.from_numpy(design_phase).to(device)
            with torch.no_grad():
                clocked = generate(torch.from_numpy(coeff)[None].to(device), theta_hat_t.unsqueeze(0))[0].cpu().numpy()
            clock_rel = rel_np(clocked, fields, mask)
            rows.append(
                {
                    "split": split,
                    "id": cid,
                    "lift_r2": store.r2[i],
                    "freq_hz": store.freq[i],
                    "freq_hat_hz": omega / (2 * np.pi),
                    "freq_err": abs(omega / (2 * np.pi) / store.freq[i] - 1),
                    "oracle_rel": oracle_rel.tolist(),
                    "pred_modes_true_phase": mode_rel,
                    "true_modes_startup_clock": clock_rel,
                }
            )
    def pack(split: str, key: str) -> dict:
        picked = [r for r in rows if r["split"] == split]
        arr = np.array([r[key] if key != "freq_err" else [r["freq_err"]] * 3 for r in picked])
        if key == "freq_err":
            vals = np.array([r["freq_err"] for r in picked])
            return {"median": float(np.median(vals)), "mean": float(vals.mean()), "max": float(vals.max())}
        mean = arr.mean(axis=0)
        return {"ux": float(mean[0]), "uy": float(mean[1]), "p": float(mean[2])}

    summary = {
        "note": "诊断，不是预测成绩。oracle 是本例完整真值拟合的四谐波。",
        "phase_gauge": "升力基频 cos(2πft-δ) 的上升过零定义为相位 0，实现为拟合相位加 π/2。",
        "time_unit": "秒，来自 h5 的 times。角频率 ω=2πf。",
        "pressure": "使用 h5 里的压力，没有再减未来压力均值。",
        "gradient_path": "D 的冒烟测试里，内循环之后编码器梯度非零。本脚本不重新训练。",
        "train_oracle": pack("train", "oracle_rel"),
        "val_oracle": pack("val", "oracle_rel"),
        "val_pred_modes_true_phase": pack("val", "pred_modes_true_phase"),
        "val_true_modes_startup_clock": pack("val", "true_modes_startup_clock"),
        "val_freq_err": pack("val", "freq_err"),
    }
    dest = OUT / "cycle_diagnosis.json"
    dest.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", dest)


if __name__ == "__main__":
    main()
