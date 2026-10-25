"""多步推演。

连续猜 4 步。每一步的输入用上一步的预测，并且截断时间方向的梯度。
机制来自 Brandstetter 等，ICLR 2022 的 pushforward。
只从上一帧模型读取初始权重，不写回那个目录。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from flowproxy.dataset import close_h5, load_case_ids
from flowproxy.lines.isolate import claim_line, write_eval, write_status
from flowproxy.losses import channel_weights, physics_bundle, supervised_loss
from flowproxy.metrics import relative_l2
from flowproxy.models import build_model
from flowproxy.splits import _meta_from_id, held_out_geometry
from flowproxy.train import _checkpoint, _format_bar, loader_kwargs, pick_device

TAG = "unetex_120_roll4"
RIVALS = frozenset({"unetex_120_mode"})
EVAL_NAME = "rollout_120_roll4.json"
H5_NAME = "dataset_120.h5"
INIT_TAG = "unetex_120_prev1"
UNROLL = 4
HORIZON_S = 20.0


class RollWindowDataset(Dataset):
    """几何加真实当前帧，目标是后面连续 4 帧。"""

    def __init__(self, h5_path: str | Path, split: str, unroll: int = UNROLL):
        self.h5_path = str(h5_path)
        self.unroll = unroll
        self.ids = load_case_ids(h5_path, split)
        self._file = None
        self.geom: dict[str, np.ndarray] = {}
        self.mask: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as handle:
            for cid in self.ids:
                group = handle["cases"][cid]
                count = int(group["fields"].shape[0])
                self.geom[cid] = np.asarray(group["geom"][()], dtype=np.float32)
                self.mask[cid] = np.asarray(group["mask"][()], dtype=np.float32)
                last = count - 1 - unroll
                if last < 0:
                    continue
                self.samples.extend((cid, start) for start in range(last + 1))

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
        cid, start = self.samples[idx]
        fields = self._h5()["cases"][cid]["fields"]
        current = np.asarray(fields[start], dtype=np.float32)
        future = np.asarray(fields[start + 1 : start + 1 + self.unroll], dtype=np.float32)
        geom = self.geom[cid]
        return {
            "geom": torch.from_numpy(geom),
            "current": torch.from_numpy(current),
            "future": torch.from_numpy(future),
            "mask": torch.from_numpy(self.mask[cid]),
            "case_id": cid,
        }


def rollout_loss(model, geom, current, future, mask, weights, dx, dy, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """四步损失。下一步的输入是 detach 之后的预测。"""
    if geom.shape[1] != 4 or current.shape[1] != 3 or future.shape[1] != UNROLL:
        raise RuntimeError(
            f"多步推演形状不对 geom {tuple(geom.shape)} current {tuple(current.shape)} future {tuple(future.shape)}"
        )
    state = current
    total = geom.new_zeros(())
    last = current
    for step in range(UNROLL):
        pred = model(torch.cat([geom, state], dim=1))
        step_loss = supervised_loss(pred, future[:, step], weights)
        step_loss = step_loss + physics_bundle(
            pred, mask, dx, dy, float(cfg["train"]["lambda_div"]), float(cfg["train"]["lambda_wall"])
        )["phys"]
        total = total + step_loss
        last = pred
        state = pred.detach()
    return total / UNROLL, last


def _kind(cy: float, diameter: float) -> str:
    if held_out_geometry(cy, diameter) and abs(abs(cy) - 0.8) < 1e-6:
        return "far"
    if held_out_geometry(cy, diameter):
        return "extreme_d"
    return "other"


def _rel_one(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fluid = mask if mask.ndim == 3 else mask.squeeze(1)
    pred = pred * fluid
    target = target * fluid
    num = torch.linalg.vector_norm(pred - target, dim=(1, 2))
    den = torch.linalg.vector_norm(target, dim=(1, 2)).clamp_min(1e-8)
    return num / den


@torch.no_grad()
def _evaluate(model, loader, device, dx, dy, cfg) -> dict[str, float]:
    model.eval()
    last_totals: dict[str, float] = {}
    first_uy = 0.0
    batches = 0
    for batch in loader:
        geom = batch["geom"].to(device)
        current = batch["current"].to(device)
        future = batch["future"].to(device)
        mask = batch["mask"].to(device)
        state = current
        first = None
        pred = current
        for step in range(UNROLL):
            pred = model(torch.cat([geom, state], dim=1))
            if step == 0:
                first = pred
            state = pred
        rel_last = relative_l2(pred, future[:, -1], mask)
        rel_first = relative_l2(first, future[:, 0], mask)
        err = {
            "rel_l2_ux": float(rel_last[0]),
            "rel_l2_uy": float(rel_last[1]),
            "rel_l2_p": float(rel_last[2]),
            "rel_l2_mean": float(rel_last.mean()),
        }
        for key, value in err.items():
            last_totals[key] = last_totals.get(key, 0.0) + value
        first_uy += float(rel_first[1])
        batches += 1
    out = {key: value / max(batches, 1) for key, value in last_totals.items()}
    out["step1_uy"] = first_uy / max(batches, 1)
    return out


def _load_prev1(model, root: Path, device) -> None:
    path = root / "outputs" / "checkpoints" / INIT_TAG / "best.pt"
    if path.parent.name != INIT_TAG:
        raise SystemExit("初始权重路径不对")
    saved = torch.load(path, map_location=device, weights_only=False)
    if int(saved.get("in_channels", 0)) != 7 or int(saved.get("out_channels", 0)) != 3:
        raise SystemExit(f"{INIT_TAG} 的通道数不是上一帧模型")
    model.load_state_dict(saved["model"])
    print(f"[roll] read only init {path}", flush=True)


def _exam(cfg, ckpt: Path, root: Path, device) -> None:
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pack.get("line") != "pushforward":
        raise SystemExit("这份权重不是多步推演，拒绝考试")
    model = build_model("unetex", 7, 3, cfg).to(device)
    model.load_state_dict(pack["model"])
    model.eval()
    h5_path = root / "data" / "processed" / H5_NAME
    buckets: dict[str, list[torch.Tensor]] = {}
    last_rows: dict[str, list[torch.Tensor]] = {}
    with h5py.File(h5_path, "r") as handle:
        test_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["splits"]["test"][()]]
        for cid in test_ids:
            case = handle["cases"][cid]
            geom = torch.from_numpy(np.asarray(case["geom"][()], dtype=np.float32)).to(device)
            fields = torch.from_numpy(np.asarray(case["fields"][()], dtype=np.float32)).to(device)
            times = np.asarray(case["times"][()], dtype=np.float64)
            mask = torch.from_numpy(np.asarray(case["mask"][()], dtype=np.float32)).to(device)
            prev = fields[0]
            end = float(times[0]) + HORIZON_S
            series = []
            with torch.no_grad():
                for index in range(1, fields.shape[0]):
                    if float(times[index]) > end:
                        break
                    pred = model(torch.cat([geom, prev], dim=0).unsqueeze(0))[0, :3]
                    series.append(_rel_one(pred, fields[index], mask).cpu())
                    prev = pred
            meta = _meta_from_id(cid)
            keys = ["all", f"shape:{meta['shape']}", f"kind:{_kind(meta['cy'], meta['D'])}"]
            for key in keys:
                buckets.setdefault(key, []).extend(series)
                last_rows.setdefault(key, []).append(series[-1])

    def pack_rows(groups: dict[str, list[torch.Tensor]]) -> dict:
        out = {}
        for key, rows in groups.items():
            mean = torch.stack(rows, dim=0).mean(dim=0)
            out[key] = {
                "n": len(rows),
                "rel_l2_ux": float(mean[0]),
                "rel_l2_uy": float(mean[1]),
                "rel_l2_p": float(mean[2]),
                "rel_l2_mean": float(mean.mean()),
            }
        return out

    payload = {
        "line": "pushforward",
        "tag": TAG,
        "ckpt": str(ckpt),
        "horizon_s": HORIZON_S,
        "note": "训练时连续 4 步，步与步之间截断梯度。考试从 t=40 的真值出发，之后用自己的预测连着猜 20 秒。",
        "mean_over_steps": pack_rows(buckets),
        "last_step": pack_rows(last_rows),
    }
    dest = write_eval(root, EVAL_NAME, json.dumps(payload, indent=2), EVAL_NAME, frozenset({"test_120_mode.json"}))
    mean_all = payload["mean_over_steps"]["all"]
    last_all = payload["last_step"]["all"]
    print(
        f"[roll] rollout mean ux {mean_all['rel_l2_ux']*100:.1f}% uy {mean_all['rel_l2_uy']*100:.1f}% "
        f"p {mean_all['rel_l2_p']*100:.1f}%",
        flush=True,
    )
    print(
        f"[roll] rollout last ux {last_all['rel_l2_ux']*100:.1f}% uy {last_all['rel_l2_uy']*100:.1f}% "
        f"p {last_all['rel_l2_p']*100:.1f}% -> {dest}",
        flush=True,
    )


def train_roll(cfg: dict) -> Path:
    root = Path(cfg["_root"])
    h5 = root / "data" / "processed" / H5_NAME
    if h5.name != H5_NAME or not h5.is_file():
        raise SystemExit(f"多步推演只读 {H5_NAME}")
    out_dir, _lock = claim_line(root, TAG, RIVALS)
    device = pick_device(str(cfg["train"]["device"]))
    train_set = RollWindowDataset(h5, "train")
    val_set = RollWindowDataset(h5, "val")
    sample = train_set[0]
    if sample["geom"].shape[0] != 4 or sample["future"].shape[0] != UNROLL:
        raise RuntimeError("多步推演的样本形状不对")
    close_h5(train_set)
    model = build_model("unetex", 7, 3, cfg).to(device)
    _load_prev1(model, root, device)
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
    probe = train_set[0]["future"][0].unsqueeze(0)
    weights = channel_weights(probe.to(device))
    close_h5(train_set)
    close_h5(val_set)
    cfg["line"] = {"name": "pushforward", "tag": TAG, "unroll": UNROLL, "rival": "unetex_120_mode"}
    best = float("inf")
    stale = 0
    history = []
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    print(f"[roll] dir {out_dir}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        total = len(train_loader)
        started = time.perf_counter()
        shown = 0.0
        loss_value = 0.0
        for step, batch in enumerate(train_loader, 1):
            geom = batch["geom"].to(device)
            current = batch["current"].to(device)
            future = batch["future"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", enabled=True):
                    loss, _ = rollout_loss(model, geom, current, future, mask, weights, dx, dy, cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss, _ = rollout_loss(model, geom, current, future, mask, weights, dx, dy, cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running += float(loss.detach())
            loss_value = float(loss.detach())
            now = time.perf_counter()
            if step == 1 or step == total or now - shown >= 0.5:
                shown = now
                write_status(out_dir, _format_bar(f"roll ep{epoch}/{epochs}", step, total, now - started, loss_value))
        val_metrics = _evaluate(model, val_loader, device, dx, dy, cfg)
        val_score = val_metrics["rel_l2_mean"]
        sched.step(val_score)
        row = {"epoch": epoch, "train_loss": running / max(total, 1), **val_metrics}
        history.append(row)
        print(
            f"[roll] epoch {epoch:03d}  train {row['train_loss']:.4e}  "
            f"val relL2 {val_score:.4f}  ux {val_metrics['rel_l2_ux']:.4f}  "
            f"uy {val_metrics['rel_l2_uy']:.4f}  p {val_metrics['rel_l2_p']:.4f}  "
            f"step1_uy {val_metrics['step1_uy']:.4f}",
            flush=True,
        )
        if val_score < best:
            best = val_score
            stale = 0
        else:
            stale += 1
        snapshot = _checkpoint(
            model, opt, sched, scaler, "unetex", "pushforward", 7, 3, cfg,
            epoch, best, stale, history, val_metrics,
        )
        snapshot["line"] = "pushforward"
        snapshot["tag"] = TAG
        torch.save(snapshot, out_dir / "last.pt")
        if val_score == best:
            torch.save(snapshot, out_dir / "best.pt")
        if stale >= patience:
            print(f"[roll] early stopping at epoch {epoch}", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_path = out_dir / "best.pt"
    _exam(cfg, best_path, root, device)
    return best_path
