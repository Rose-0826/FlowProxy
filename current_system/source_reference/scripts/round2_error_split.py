#!/usr/bin/env python3
"""把 E1750、F1750 的误差拆开，并说明诊断口径和正式评测为什么不一样。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowproxy.models import build_model  # noqa: E402
from flowproxy.paths import load_config  # noqa: E402
from flowproxy.round2.data import OUT, CaseStore  # noqa: E402
from flowproxy.round2.metrics import per_frame_rel  # noqa: E402


def stacked_rel(pred, true, mask) -> float:
    fluid = mask > 0.5
    a = (pred[:, 1] - true[:, 1])[:, fluid]
    b = true[:, 1][:, fluid]
    return float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-8))


def harmonic_parts(frames: np.ndarray, phase: np.ndarray) -> dict[str, np.ndarray]:
    """用同一套相位，把场拆成平均、一次谐波、二次及以上。"""
    theta = phase.astype(np.float64)
    cols = [np.ones(frames.shape[0], dtype=np.float64)]
    for k in range(1, 5):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    design = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, frames.reshape(frames.shape[0], -1).astype(np.float64), rcond=None)
    mean = (design[:, :1] @ coef[:1]).reshape(frames.shape)
    fund = (design[:, 1:3] @ coef[1:3]).reshape(frames.shape)
    higher = (design[:, 3:] @ coef[3:]).reshape(frames.shape)
    return {"mean": mean, "fund": fund, "higher": higher}


def against_full(part_pred, part_true, full_true, mask) -> float:
    """分量误差除以整段 Uy，不再除以该分量自己的大小。"""
    fluid = mask > 0.5
    num = np.linalg.norm((part_pred[:, 1] - part_true[:, 1])[:, fluid])
    den = np.linalg.norm(full_true[:, 1][:, fluid]) + 1e-8
    return float(num / den)


def energy_share(part, full, mask) -> float:
    fluid = mask > 0.5
    return float(np.sum(part[:, 1][:, fluid] ** 2) / (np.sum(full[:, 1][:, fluid] ** 2) + 1e-12))


def frame_mean_uy(pred_t, true_t, mask) -> float:
    err = per_frame_rel(pred_t, true_t, mask)
    return float(err[:, 1].mean())


def main() -> None:
    device = torch.device("cuda")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    store = CaseStore("val")
    report = {"note": "诊断。一次时间平移最多 24 帧，约一个周期。"}
    for name, ckpt in [
        ("E1750", ROOT / "outputs/checkpoints/round2_E_seed0/best.pt"),
        ("F1750", ROOT / "outputs/checkpoints/round2_F_seed0/best.pt"),
    ]:
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model = build_model(saved["model_name"], saved["in_channels"], saved["out_channels"], cfg).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        rows = []
        with torch.no_grad():
            for i, cid in enumerate(store.ids):
                geom = torch.from_numpy(store.geom[i])[None].to(device)
                fields = torch.from_numpy(store.fields[i]).to(device)
                mask_t = torch.from_numpy(store.mask[i]).to(device)
                window = [fields[j] for j in range(4)]
                preds = []
                for _ in range(4, fields.shape[0]):
                    hist = torch.stack(window[-4:]).reshape(-1, *fields.shape[-2:])
                    pred = model(torch.cat([geom, hist[None]], 1))[0, :3]
                    preds.append(pred)
                    window.append(pred)
                pred = torch.stack(preds).cpu().numpy()
                true = fields[4:].cpu().numpy()
                mask = mask_t.cpu().numpy()
                official = frame_mean_uy(torch.from_numpy(pred), torch.from_numpy(true), mask_t.cpu())
                stacked = stacked_rel(pred, true, mask)
                best = stacked
                best_lag = 0
                for lag in range(-24, 25):
                    if lag == 0:
                        continue
                    a, b = (pred[lag:], true[:-lag]) if lag > 0 else (pred[:lag], true[-lag:])
                    value = stacked_rel(a, b, mask)
                    if value < best:
                        best, best_lag = value, lag
                mean_p = pred.mean(axis=0)
                mean_t = true.mean(axis=0)
                mean_err = stacked_rel(mean_p[None], mean_t[None], mask)
                fluct = stacked_rel(pred - mean_p, true - mean_t, mask)
                # 涡核：流体里 |涡量| 最大点，只看 x>2 的尾迹
                def core(frames):
                    u, v = frames[:, 0], frames[:, 1]
                    dvdx = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * ((20 - (-8)) / 63))
                    dudy = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * (16 / 63))
                    omega = np.abs(dvdx - dudy)
                    xs = np.linspace(-8, 20, mask.shape[1])
                    wake = xs[1:-1] > 2
                    omega[:, :, ~wake] = 0
                    pos = []
                    for t in range(omega.shape[0]):
                        j = np.unravel_index(np.argmax(omega[t]), omega[t].shape)
                        pos.append((xs[j[1] + 1], np.linspace(-8, 8, mask.shape[0])[j[0] + 1]))
                    return np.array(pos)
                cp, ct = core(pred), core(true)
                phase = store.phase[i][4:]
                parts_p = harmonic_parts(pred, phase)
                parts_t = harmonic_parts(true, phase)
                higher = stacked_rel(parts_p["higher"], parts_t["higher"], mask)
                fund = stacked_rel(parts_p["fund"], parts_t["fund"], mask)
                rows.append(
                    {
                        "id": cid,
                        "official_frame_mean_uy": official,
                        "stacked_uy_before_shift": stacked,
                        "stacked_uy_after_shift": best,
                        "lag_frames": best_lag,
                        "mean_field_uy": mean_err,
                        "fluctuation_uy": fluct,
                        "fundamental_uy": fund,
                        "higher_harmonic_uy": higher,
                        "mean_over_full_uy": against_full(parts_p["mean"], parts_t["mean"], true, mask),
                        "fundamental_over_full_uy": against_full(parts_p["fund"], parts_t["fund"], true, mask),
                        "higher_over_full_uy": against_full(parts_p["higher"], parts_t["higher"], true, mask),
                        "energy_mean": energy_share(parts_t["mean"], true, mask),
                        "energy_fundamental": energy_share(parts_t["fund"], true, mask),
                        "energy_higher": energy_share(parts_t["higher"], true, mask),
                        "core_shift_mean_length": float(np.linalg.norm(cp - ct, axis=1).mean()),
                    }
                )
        def avg(key):
            return float(np.mean([r[key] for r in rows]))
        report_one = {
            "official_frame_mean_uy": avg("official_frame_mean_uy"),
            "stacked_before": avg("stacked_uy_before_shift"),
            "stacked_after": avg("stacked_uy_after_shift"),
            "mean_field_uy": avg("mean_field_uy"),
            "fluctuation_uy": avg("fluctuation_uy"),
            "fundamental_uy": avg("fundamental_uy"),
            "higher_harmonic_uy": avg("higher_harmonic_uy"),
            "mean_over_full_uy": avg("mean_over_full_uy"),
            "fundamental_over_full_uy": avg("fundamental_over_full_uy"),
            "higher_over_full_uy": avg("higher_over_full_uy"),
            "energy_mean": avg("energy_mean"),
            "energy_fundamental": avg("energy_fundamental"),
            "energy_higher": avg("energy_higher"),
            "note": "相对自身真值的百分比不能相加。统一分母是整段 Uy；能量占比来自真值，交叉项存在时三者之和不必为 1。",
            "core_shift": avg("core_shift_mean_length"),
            "cases": rows,
        }
        report = report_one
        print(name if False else ckpt.name, {k: report[k] for k in report if k != "cases"})
        OUT.joinpath(f"error_split_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(name, "official", f"{report['official_frame_mean_uy']*100:.2f}%",
              "over full mean/fund/higher",
              f"{report['mean_over_full_uy']*100:.2f}%",
              f"{report['fundamental_over_full_uy']*100:.2f}%",
              f"{report['higher_over_full_uy']*100:.2f}%",
              "energy",
              f"{report['energy_mean']*100:.1f}%",
              f"{report['energy_fundamental']*100:.1f}%",
              f"{report['energy_higher']*100:.1f}%")


if __name__ == "__main__":
    main()
