"""相位锚定。

平均流加一对涡街模态，频率由形状和雷诺数给出。
相位起点用第一帧的上下流速在这对模态上的投影，之后自己往前走。
机制来自 Noack 等，J. Fluid Mech. 497, 2003。
网络只吃几何，不吃整帧流场。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from flowproxy.dataset import close_h5, load_case_ids
from flowproxy.lines.isolate import claim_line, write_eval, write_status
from flowproxy.losses import channel_weights, physics_bundle, report_errors, supervised_loss
from flowproxy.metrics import relative_l2
from flowproxy.models import build_model
from flowproxy.splits import _meta_from_id, held_out_geometry
from flowproxy.train import _checkpoint, _format_bar, loader_kwargs, pick_device

TAG = "unetex_120_mode"
RIVALS = frozenset({"unetex_120_roll4"})
EVAL_NAME = "test_120_mode.json"
H5_NAME = "dataset_120.h5"


class ModePhaseDataset(Dataset):
    """几何、第一帧、目标帧和时间差。第一帧只用来算相位，不进网络。"""

    def __init__(self, h5_path: str | Path, split: str):
        self.h5_path = str(h5_path)
        self.ids = load_case_ids(h5_path, split)
        self._file = None
        self.geom: dict[str, np.ndarray] = {}
        self.mask: dict[str, np.ndarray] = {}
        self.anchor: dict[str, np.ndarray] = {}
        self.times: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as handle:
            for cid in self.ids:
                group = handle["cases"][cid]
                fields = group["fields"]
                times = np.asarray(group["times"], dtype=np.float64)
                if fields.shape[0] != times.shape[0] or fields.shape[0] < 2:
                    raise RuntimeError(f"{cid} 的帧数和时间对不上")
                self.geom[cid] = np.asarray(group["geom"][()], dtype=np.float32)
                self.mask[cid] = np.asarray(group["mask"][()], dtype=np.float32)
                self.anchor[cid] = np.asarray(fields[0], dtype=np.float32)
                self.times[cid] = times
                self.samples.extend((cid, t) for t in range(fields.shape[0]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _h5(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        cid, frame = self.samples[idx]
        target = np.asarray(self._h5()["cases"][cid]["fields"][frame], dtype=np.float32)
        dt = float(self.times[cid][frame] - self.times[cid][0])
        return {
            "geom": torch.from_numpy(self.geom[cid]),
            "anchor": torch.from_numpy(self.anchor[cid]),
            "y": torch.from_numpy(target),
            "dt": torch.tensor(dt, dtype=torch.float32),
            "mask": torch.from_numpy(self.mask[cid]),
            "case_id": cid,
        }


def reconstruct(raw: torch.Tensor, anchor: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """raw 的 10 个通道：平均流 3、余弦模态 3、正弦模态 3、频率 1。"""
    mean = raw[:, 0:3]
    mode_cos = raw[:, 3:6]
    mode_sin = raw[:, 6:9]
    fluid = mask.unsqueeze(1) if mask.ndim == 3 else mask
    weight = fluid.sum(dim=(2, 3)).clamp_min(1.0)
    logit = (raw[:, 9:10] * fluid).sum(dim=(2, 3)) / weight
    freq_hz = 0.16 * torch.exp(logit.clamp(-1.5, 1.1))
    omega = freq_hz * (2.0 * math.pi)
    fluct = (anchor[:, 1:2] - mean[:, 1:2]) * fluid
    proj_cos = (fluct * mode_cos[:, 1:2]).sum(dim=(1, 2, 3))
    proj_sin = (fluct * mode_sin[:, 1:2]).sum(dim=(1, 2, 3))
    radius = torch.sqrt(proj_cos.square() + proj_sin.square())
    phase_live = torch.atan2(proj_sin, proj_cos)
    phase0 = torch.where(radius >= 1.0, phase_live, phase_live.detach())
    phase = phase0.view(-1, 1, 1, 1) + omega.view(-1, 1, 1, 1) * dt.view(-1, 1, 1, 1)
    return mean + torch.cos(phase) * mode_cos + torch.sin(phase) * mode_sin


def _kind(cy: float, diameter: float) -> str:
    if held_out_geometry(cy, diameter) and abs(abs(cy) - 0.8) < 1e-6:
        return "far"
    if held_out_geometry(cy, diameter):
        return "extreme_d"
    return "other"


def _per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fluid = mask.unsqueeze(1) if mask.ndim == 3 else mask
    pred = pred * fluid
    target = target * fluid
    num = torch.linalg.vector_norm(pred - target, dim=(2, 3))
    den = torch.linalg.vector_norm(target, dim=(2, 3)).clamp_min(1e-8)
    return num / den


@torch.no_grad()
def _evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        geom = batch["geom"].to(device)
        pred = reconstruct(
            model(geom),
            batch["anchor"].to(device),
            batch["dt"].to(device),
            batch["mask"].to(device),
        )
        err = report_errors(pred, batch["y"].to(device), batch["mask"].to(device))
        rel = relative_l2(pred, batch["y"].to(device), batch["mask"].to(device))
        err["rel_l2_mean"] = float(rel.mean())
        for key, value in err.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _loss_backward(model, opt, scaler, batch, weights, device, dx, dy, cfg) -> torch.Tensor:
    geom = batch["geom"].to(device)
    if geom.shape[1] != 4:
        raise RuntimeError(f"相位锚定只吃 4 个几何通道，拿到了 {geom.shape[1]}")
    anchor = batch["anchor"].to(device)
    target = batch["y"].to(device)
    dt = batch["dt"].to(device)
    mask = batch["mask"].to(device)
    opt.zero_grad(set_to_none=True)
    use_amp = scaler is not None
    if use_amp:
        with torch.autocast(device_type="cuda", enabled=True):
            raw = model(geom)
        pred = reconstruct(raw.float(), anchor.float(), dt.float(), mask.float())
        loss = supervised_loss(pred, target.float(), weights)
        loss = loss + physics_bundle(
            pred, mask.float(), dx, dy, float(cfg["train"]["lambda_div"]), float(cfg["train"]["lambda_wall"])
        )["phys"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        pred = reconstruct(model(geom), anchor, dt, mask)
        loss = supervised_loss(pred, target, weights)
        loss = loss + physics_bundle(
            pred, mask, dx, dy, float(cfg["train"]["lambda_div"]), float(cfg["train"]["lambda_wall"])
        )["phys"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.detach()


def _exam(cfg, ckpt: Path, root: Path, device) -> None:
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pack.get("line") != "mode_phase":
        raise SystemExit("这份权重不是相位锚定，拒绝考试")
    h5 = root / "data" / "processed" / H5_NAME
    dataset = ModePhaseDataset(h5, "test")
    model = build_model("unetex", 4, 10, cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    buckets: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            pred = reconstruct(
                model(batch["geom"].to(device)),
                batch["anchor"].to(device),
                batch["dt"].to(device),
                batch["mask"].to(device),
            )
            rel = _per_sample(pred, batch["y"].to(device), batch["mask"].to(device)).cpu()
            for index, cid in enumerate(batch["case_id"]):
                meta = _meta_from_id(cid)
                row = rel[index]
                keys = ["all", f"shape:{meta['shape']}", f"Re:{int(meta['Re'])}", f"kind:{_kind(meta['cy'], meta['D'])}"]
                for key in keys:
                    buckets.setdefault(key, []).append(row)
    groups = {}
    for key, rows in buckets.items():
        mean = torch.stack(rows, dim=0).mean(dim=0)
        groups[key] = {
            "n": len(rows),
            "rel_l2_ux": float(mean[0]),
            "rel_l2_uy": float(mean[1]),
            "rel_l2_p": float(mean[2]),
            "rel_l2_mean": float(mean.mean()),
        }
    payload = {
        "line": "mode_phase",
        "tag": TAG,
        "ckpt": str(ckpt),
        "note": "形状和雷诺数决定平均流、一对模态和频率。相位只用每个算例第一帧的上下流速投影，之后按频率自己走。",
        "groups": groups,
    }
    dest = write_eval(root, EVAL_NAME, json.dumps(payload, indent=2), EVAL_NAME, frozenset({"rollout_120_roll4.json"}))
    close_h5(dataset)
    all_row = groups["all"]
    print(
        f"[mode] test ux {all_row['rel_l2_ux']*100:.1f}% uy {all_row['rel_l2_uy']*100:.1f}% "
        f"p {all_row['rel_l2_p']*100:.1f}% -> {dest}",
        flush=True,
    )


def train_mode(cfg: dict) -> Path:
    root = Path(cfg["_root"])
    h5 = root / "data" / "processed" / H5_NAME
    if h5.name != H5_NAME or not h5.is_file():
        raise SystemExit(f"相位锚定只读 {H5_NAME}")
    out_dir, _lock = claim_line(root, TAG, RIVALS)
    device = pick_device(str(cfg["train"]["device"]))
    train_set = ModePhaseDataset(h5, "train")
    val_set = ModePhaseDataset(h5, "val")
    sample = train_set[0]
    if tuple(sample["geom"].shape)[0] != 4 or tuple(sample["y"].shape)[0] != 3:
        raise RuntimeError("相位锚定的样本形状不对")
    close_h5(train_set)
    model = build_model("unetex", 4, 10, cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    common = loader_kwargs(cfg)
    train_loader = DataLoader(
        train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=False, **common
    )
    val_loader = DataLoader(val_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, **common)
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    probe = train_set[0]["y"].unsqueeze(0)
    weights = channel_weights(probe.to(device))
    close_h5(train_set)
    close_h5(val_set)
    cfg["line"] = {"name": "mode_phase", "tag": TAG, "rival": "unetex_120_roll4"}
    best = float("inf")
    stale = 0
    history = []
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    print(f"[mode] dir {out_dir}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(train_loader)
        started = time.perf_counter()
        shown = 0.0
        loss_value = 0.0
        for step, batch in enumerate(train_loader, 1):
            loss = _loss_backward(model, opt, scaler, batch, weights, device, dx, dy, cfg)
            running += float(loss)
            loss_value = float(loss)
            now = time.perf_counter()
            if step == 1 or step == total or now - shown >= 0.5:
                shown = now
                write_status(out_dir, _format_bar(f"mode ep{epoch}/{epochs}", step, total, now - started, loss_value))
        val_metrics = _evaluate(model, val_loader, device)
        val_score = val_metrics["rel_l2_mean"]
        sched.step(val_score)
        row = {"epoch": epoch, "train_loss": running / max(total, 1), **val_metrics}
        history.append(row)
        print(
            f"[mode] epoch {epoch:03d}  train {row['train_loss']:.4e}  "
            f"val relL2 {val_score:.4f}  ux {val_metrics.get('rel_l2_ux', 0):.4f}  "
            f"uy {val_metrics.get('rel_l2_uy', 0):.4f}  p {val_metrics.get('rel_l2_p', 0):.4f}",
            flush=True,
        )
        if val_score < best:
            best = val_score
            stale = 0
        else:
            stale += 1
        snapshot = _checkpoint(
            model, opt, sched, scaler, "unetex", "mode_phase", 4, 10, cfg,
            epoch, best, stale, history, val_metrics,
        )
        snapshot["line"] = "mode_phase"
        snapshot["tag"] = TAG
        torch.save(snapshot, out_dir / "last.pt")
        if val_score == best:
            torch.save(snapshot, out_dir / "best.pt")
        if stale >= patience:
            print(f"[mode] early stopping at epoch {epoch}", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_path = out_dir / "best.pt"
    _exam(cfg, best_path, root, device)
    return best_path
