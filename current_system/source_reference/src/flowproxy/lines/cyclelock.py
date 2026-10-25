"""第一条线：极限环约束推演。

第一段：训练自己的周期网络（形状给平均流、四对谐波模态和频率）。
第二段：推演网络每一步吃几何、自己上一步的预测、周期在下一时刻的那一帧。
相位只从第一帧读一次，之后按形状给的频率往前走，所以时钟不跟着推演漂。
训练连续展开若干步，步间截断梯度（pushforward），起点加一点噪声。
考试从第一帧真值出发连着猜整段 40 秒，不看升力；拉回周期帧的比例只在验证集上选。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from flowproxy.lines.cycle import (
    HARMONICS,
    N_OUT,
    CaseBank,
    cycle_field,
    cycle_freq,
    group_keys,
    read_phase,
    rel_frame,
    summarize,
)
from flowproxy.lines.cycle_train import train_cycle_net
from flowproxy.lines.isolate import claim_line, write_eval, write_status
from flowproxy.losses import channel_weights, physics_bundle, supervised_loss
from flowproxy.models import build_model
from flowproxy.train import _format_bar, pick_device

TAG = "cyclelock_long"
RIVALS = frozenset({"liftcycle_long"})
EVAL_NAME = "rollout_long_cyclelock.json"
MAX_UNROLL = 8
BATCH = 16
STEPS_PER_EPOCH = 300
NOISE = 0.02
PULLS = (0.0, 0.02, 0.05, 0.1, 0.2)


@torch.no_grad()
def cycle_from_first_frame(cycle_net, bank: CaseBank) -> list[torch.Tensor]:
    """不推演，只用第一帧读相位、形状给频率，直接画出整段周期。"""
    raw = cycle_net(bank.geom).float()
    freq = cycle_freq(raw, bank.mask)
    phase0 = read_phase(raw, bank.fields[:, 0], bank.mask)
    rels = []
    for i in range(len(bank.ids)):
        dt = (bank.times[i] - bank.times[i, 0]).float()
        phase = phase0[i] + 2.0 * math.pi * freq[i] * dt
        pred = cycle_field(raw[i : i + 1].expand(dt.shape[0], -1, -1, -1), phase)
        rels.append(rel_frame(pred, bank.fields[i], bank.mask[i]).cpu())
    return rels


@torch.no_grad()
def rollout_bank(cycle_net, roll_net, bank: CaseBank, pull: float) -> torch.Tensor:
    """所有算例一起推演整段。返回相对 L2 [算例, 步, 3]。"""
    raw = cycle_net(bank.geom).float()
    freq = cycle_freq(raw, bank.mask)
    state = bank.fields[:, 0]
    phase0 = read_phase(raw, state, bank.mask)
    steps = bank.fields.shape[1] - 1
    rows = []
    for k in range(1, steps + 1):
        dt = (bank.times[:, k] - bank.times[:, 0]).float()
        prior = cycle_field(raw, phase0 + 2.0 * math.pi * freq * dt)
        pred = roll_net(torch.cat([bank.geom, state, prior], dim=1)).float()
        if pull > 0:
            pred = pred + pull * (prior - pred)
        rows.append(torch.stack([rel_frame(pred[i], bank.fields[i, k], bank.mask[i])[0] for i in range(pred.shape[0])]))
        state = pred
    return torch.stack(rows, dim=1).cpu()


@torch.no_grad()
def one_step_bank(cycle_net, roll_net, bank: CaseBank) -> torch.Tensor:
    """每一步都喂真实上一帧，和上一帧基线同一口径。返回 [算例, 步, 3]。"""
    raw = cycle_net(bank.geom).float()
    freq = cycle_freq(raw, bank.mask)
    phase0 = read_phase(raw, bank.fields[:, 0], bank.mask)
    rows = []
    for k in range(1, bank.fields.shape[1]):
        dt = (bank.times[:, k] - bank.times[:, 0]).float()
        prior = cycle_field(raw, phase0 + 2.0 * math.pi * freq * dt)
        pred = roll_net(torch.cat([bank.geom, bank.fields[:, k - 1], prior], dim=1)).float()
        rows.append(torch.stack([rel_frame(pred[i], bank.fields[i, k], bank.mask[i])[0] for i in range(pred.shape[0])]))
    return torch.stack(rows, dim=1).cpu()


def pack_rollout(ids: list[str], rel: torch.Tensor, times: torch.Tensor) -> dict:
    """整段平均、前 20 秒平均、最后一步，按分组汇总。"""
    mean_all: dict[str, list[torch.Tensor]] = {}
    mean_20: dict[str, list[torch.Tensor]] = {}
    last: dict[str, list[torch.Tensor]] = {}
    elapsed = (times[:, 1:] - times[:, :1]).cpu()
    for i, cid in enumerate(ids):
        first20 = elapsed[i] <= 20.0 + 1e-6
        for key in group_keys(cid):
            mean_all.setdefault(key, []).append(rel[i])
            mean_20.setdefault(key, []).append(rel[i][first20])
            last.setdefault(key, []).append(rel[i][-1:])
    return {"mean_40s": summarize(mean_all), "mean_20s": summarize(mean_20), "last_step": summarize(last)}


def run(cfg: dict, h5: Path, cases_root: Path, tag: str = TAG, eval_name: str = EVAL_NAME,
        cycle_epochs: int = 300, roll_epochs: int = 200, patience: int = 30) -> Path:
    root = Path(cfg["_root"])
    if not h5.is_file():
        raise SystemExit(f"缺数据 {h5}")
    out_dir, _lock = claim_line(root, tag, RIVALS)
    device = pick_device(str(cfg["train"]["device"]))
    train = CaseBank(h5, "train", cases_root, device)
    val = CaseBank(h5, "val", cases_root, device, with_labels=False)
    print(f"[lock] dir {out_dir}; stage 1 cycle net on {len(train.ids)} cases", flush=True)

    def validate_cycle(net) -> dict:
        buckets = {"all": cycle_from_first_frame(net, val)}
        return summarize(buckets)["all"]

    cycle_net, cycle_hist = train_cycle_net(cfg, train, validate_cycle, out_dir, "lock", cycle_epochs, 40, device)
    cycle_net.eval()
    for p in cycle_net.parameters():
        p.requires_grad_(False)
    torch.save({"line": "cyclelock_stage1", "tag": tag, "cycle": cycle_net.state_dict(), "history": cycle_hist}, out_dir / "cycle.pt")

    print("[lock] stage 2 rollout net", flush=True)
    roll_net = build_model("unetex", 10, 3, cfg).to(device)
    opt = torch.optim.AdamW(roll_net.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    lam_div = float(cfg["train"]["lambda_div"])
    lam_wall = float(cfg["train"]["lambda_wall"])
    weights = channel_weights(train.fields[:, ::20].reshape(-1, 3, *train.fields.shape[-2:]))
    n_cases, n_frames = train.fields.shape[:2]
    with torch.no_grad():
        raw_all = cycle_net(train.geom).float()
        freq_all = cycle_freq(raw_all, train.mask)
    best = float("inf")
    stale = 0
    history: list[dict] = []
    for epoch in range(1, roll_epochs + 1):
        unroll = min(MAX_UNROLL, 1 + (epoch - 1) // 3)
        roll_net.train()
        started = time.perf_counter()
        running = 0.0
        for step in range(1, STEPS_PER_EPOCH + 1):
            pick = torch.randint(0, n_cases, (BATCH,), device=device)
            start = torch.randint(0, n_frames - unroll, (BATCH,), device=device)
            geom = train.geom[pick]
            mask = train.mask[pick]
            raw = raw_all[pick]
            freq = freq_all[pick]
            current = train.fields[pick, start]
            phase0 = read_phase(raw, current, mask)
            state = current + torch.randn_like(current) * weights * NOISE * mask.unsqueeze(1)
            loss = geom.new_zeros(())
            for k in range(1, unroll + 1):
                dt = (train.times[pick, start + k] - train.times[pick, start]).float()
                prior = cycle_field(raw, phase0 + 2.0 * math.pi * freq * dt)
                pred = roll_net(torch.cat([geom, state, prior], dim=1))
                target = train.fields[pick, start + k]
                loss = loss + supervised_loss(pred, target, weights) + physics_bundle(pred, mask, dx, dy, lam_div, lam_wall)["phys"]
                state = pred.detach()
            loss = loss / unroll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(roll_net.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            if step == 1 or step % 20 == 0 or step == STEPS_PER_EPOCH:
                write_status(out_dir, _format_bar(f"lock roll ep{epoch}/{roll_epochs} unroll{unroll}", step, STEPS_PER_EPOCH, time.perf_counter() - started, float(loss.detach())))
        roll_net.eval()
        rel = rollout_bank(cycle_net, roll_net, val, 0.0)
        mean = rel.reshape(-1, 3).mean(dim=0)
        score = float(mean.mean())
        row = {"epoch": epoch, "unroll": unroll, "train_loss": running / STEPS_PER_EPOCH,
               "rel_l2_ux": float(mean[0]), "rel_l2_uy": float(mean[1]), "rel_l2_p": float(mean[2]),
               "rel_l2_mean": score, "last_uy": float(rel[:, -1, 1].mean())}
        history.append(row)
        print(
            f"[lock] roll epoch {epoch:03d} unroll {unroll}  train {row['train_loss']:.4e}  val 40s relL2 {score:.4f}  "
            f"ux {row['rel_l2_ux']:.4f}  uy {row['rel_l2_uy']:.4f}  p {row['rel_l2_p']:.4f}  last_uy {row['last_uy']:.4f}",
            flush=True,
        )
        if unroll == MAX_UNROLL:
            sched.step(score)
        if score < best:
            best = score
            stale = 0
            torch.save({"line": "cyclelock", "tag": tag, "cycle": cycle_net.state_dict(), "roll": roll_net.state_dict(),
                        "epoch": epoch, "history": history, "cycle_history": cycle_hist, "harmonics": HARMONICS,
                        "config": {k: cfg[k] for k in cfg if not k.startswith("_")}}, out_dir / "best.pt")
        elif unroll == MAX_UNROLL:
            stale += 1
        if stale >= patience:
            print(f"[lock] roll early stopping at epoch {epoch}", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps({"cycle": cycle_hist, "roll": history}, indent=2), encoding="utf-8")
    ckpt = out_dir / "best.pt"
    exam(cfg, ckpt, h5, cases_root, eval_name)
    return ckpt


def exam(cfg: dict, ckpt: Path, h5: Path, cases_root: Path, eval_name: str) -> None:
    root = Path(cfg["_root"])
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pack.get("line") != "cyclelock":
        raise SystemExit("这份权重不是第一条线，拒绝考试")
    device = pick_device(str(cfg["train"]["device"]))
    cycle_net = build_model("unetex", 4, N_OUT, cfg).to(device)
    roll_net = build_model("unetex", 10, 3, cfg).to(device)
    cycle_net.load_state_dict(pack["cycle"])
    roll_net.load_state_dict(pack["roll"])
    cycle_net.eval()
    roll_net.eval()
    val = CaseBank(h5, "val", cases_root, device, with_labels=False)
    test = CaseBank(h5, "test", cases_root, device, with_labels=False)
    scan = {}
    for pull in PULLS:
        rel = rollout_bank(cycle_net, roll_net, val, pull)
        scan[str(pull)] = float(rel.mean())
    chosen = min(PULLS, key=lambda p: scan[str(p)])
    cycle_only = cycle_from_first_frame(cycle_net, test)
    cycle_buckets: dict[str, list[torch.Tensor]] = {}
    for cid, rel in zip(test.ids, cycle_only):
        for key in group_keys(cid):
            cycle_buckets.setdefault(key, []).append(rel)
    payload = {
        "line": "cyclelock",
        "tag": pack["tag"],
        "ckpt": str(ckpt),
        "best_epoch": int(pack["epoch"]),
        "harmonics": HARMONICS,
        "note": "从 t=160 的一帧真值出发，只从这一帧读一次相位，之后用自己的预测连着猜 40 秒。周期帧和频率都由形状给出，不看升力。",
        "val_pull_scan": scan,
        "pull_selected_on_val": chosen,
        "rollout": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, chosen), test.times),
        "rollout_pull0": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, 0.0), test.times),
        "one_step_true_prev": pack_rollout(test.ids, one_step_bank(cycle_net, roll_net, test), test.times)["mean_40s"],
        "cycle_only_first_frame": summarize(cycle_buckets),
    }
    dest = write_eval(root, eval_name, json.dumps(payload, indent=2), eval_name, frozenset())
    m = payload["rollout"]["mean_40s"]["all"]
    last = payload["rollout"]["last_step"]["all"]
    print(
        f"[lock] pull {chosen} rollout 40s mean ux {m['rel_l2_ux']*100:.2f}% uy {m['rel_l2_uy']*100:.2f}% "
        f"p {m['rel_l2_p']*100:.2f}%  last uy {last['rel_l2_uy']*100:.2f}% -> {dest}",
        flush=True,
    )
