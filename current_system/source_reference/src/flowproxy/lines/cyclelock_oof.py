"""第一条线修正版：周期帧来自交叉分组，推演网络必须会看自己的上一帧。

上一版的推演网络完全靠周期帧：训练时周期帧由同一批算例拟合出的周期网络给出，几乎是准的。
这一版：
1. 训练算例分 FOLDS 组，每组的周期帧由没见过这组的周期网络给出（误差和真实使用时同类）。
   另训一个用全部训练算例的周期网络，只在验证和考试时用。
2. 30% 的样本整段不给周期帧，逼推演网络能只靠自己的上一帧。
3. 周期帧随机加相位偏移、缩放涨落，模拟外推时周期帧不准。
4. 输入多一张当前时刻的周期帧，推演网络能拿它和自己的当前状态比，判断周期帧可不可信。
输入：几何 4、自己的当前状态 3、当前时刻周期帧 3、下一时刻周期帧 3。
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
    read_phase,
    rel_frame,
    summarize,
)
from flowproxy.lines.cycle_train import train_cycle_net
from flowproxy.lines.cyclelock import cycle_from_first_frame, pack_rollout
from flowproxy.lines.isolate import claim_line, write_eval, write_status
from flowproxy.losses import channel_weights, physics_bundle, supervised_loss
from flowproxy.models import build_model
from flowproxy.train import _format_bar, pick_device

TAG = "cyclelock_oof"
RIVALS = frozenset({"cyclelock_long", "liftcycle_long"})
EVAL_NAME = "rollout_long_cyclelock_oof.json"
FOLDS = 5
IN_CH = 13
MAX_UNROLL = 8
BATCH = 16
STEPS_PER_EPOCH = 300
NOISE = 0.02
DROP_PRIOR = 0.3
JITTER_P = 0.5
PHASE_JITTER = 0.6
AMP_JITTER = 0.3
PULLS = (0.0, 0.02, 0.05, 0.1)


def priors_at(raw: torch.Tensor, phase: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    field = cycle_field(raw, phase)
    if scale is None:
        return field
    mean = raw[:, 0:3]
    return mean + scale.reshape(-1, 1, 1, 1) * (field - mean)


@torch.no_grad()
def rollout_bank(cycle_net, roll_net, bank: CaseBank, pull: float, use_prior: bool = True, true_prev: bool = False) -> torch.Tensor:
    raw = cycle_net(bank.geom).float()
    freq = cycle_freq(raw, bank.mask)
    phase0 = read_phase(raw, bank.fields[:, 0], bank.mask)
    state = bank.fields[:, 0]
    rows = []
    for k in range(1, bank.fields.shape[1]):
        t_prev = (bank.times[:, k - 1] - bank.times[:, 0]).float()
        t_next = (bank.times[:, k] - bank.times[:, 0]).float()
        prior_now = priors_at(raw, phase0 + 2.0 * math.pi * freq * t_prev)
        prior_next = priors_at(raw, phase0 + 2.0 * math.pi * freq * t_next)
        if not use_prior:
            prior_now = torch.zeros_like(prior_now)
            prior_next = torch.zeros_like(prior_next)
        prev = bank.fields[:, k - 1] if true_prev else state
        pred = roll_net(torch.cat([bank.geom, prev, prior_now, prior_next], dim=1)).float()
        if pull > 0 and use_prior:
            pred = pred + pull * (prior_next - pred)
        rows.append(torch.stack([rel_frame(pred[i], bank.fields[i, k], bank.mask[i])[0] for i in range(pred.shape[0])]))
        state = pred
    return torch.stack(rows, dim=1).cpu()


def run(cfg: dict, h5: Path, cases_root: Path, tag: str = TAG, eval_name: str = EVAL_NAME,
        cycle_epochs: int = 150, roll_epochs: int = 200, patience: int = 30) -> Path:
    root = Path(cfg["_root"])
    if not h5.is_file():
        raise SystemExit(f"缺数据 {h5}")
    out_dir, _lock = claim_line(root, tag, RIVALS)
    device = pick_device(str(cfg["train"]["device"]))
    train = CaseBank(h5, "train", cases_root, device)
    val = CaseBank(h5, "val", cases_root, device, with_labels=False)
    n_cases = len(train.ids)
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    order = torch.randperm(n_cases, generator=generator).tolist()
    folds = [sorted(order[f::FOLDS]) for f in range(FOLDS)]
    print(f"[oof] dir {out_dir}; {n_cases} train cases in {FOLDS} folds", flush=True)

    raw_oof = torch.zeros((n_cases, N_OUT, *train.geom.shape[-2:]), device=device)
    fold_scores = []
    for f, held in enumerate(folds):
        keep = [i for i in range(n_cases) if i not in set(held)]
        part = train.subset(keep)
        held_bank = train.subset(held)

        def validate_fold(net, bank=held_bank) -> dict:
            return summarize({"all": cycle_from_first_frame(net, bank)})["all"]

        net, hist = train_cycle_net(cfg, part, validate_fold, out_dir, f"oof fold{f}", cycle_epochs, patience, device)
        net.eval()
        with torch.no_grad():
            raw_oof[torch.as_tensor(held, device=device)] = net(held_bank.geom).float()
        best = min(hist, key=lambda r: r["rel_l2_mean"])
        fold_scores.append({"fold": f, "held": [train.ids[i] for i in held], **best})
        print(f"[oof] fold {f} held-out cycle-only uy {best['rel_l2_uy']*100:.1f}% p {best['rel_l2_p']*100:.1f}%", flush=True)
        del net

    def validate_full(net) -> dict:
        return summarize({"all": cycle_from_first_frame(net, val)})["all"]

    cycle_net, cycle_hist = train_cycle_net(cfg, train, validate_full, out_dir, "oof full", cycle_epochs, patience, device)
    cycle_net.eval()
    for p in cycle_net.parameters():
        p.requires_grad_(False)
    torch.save({"line": "cyclelock_oof_stage1", "tag": tag, "cycle": cycle_net.state_dict(), "raw_oof": raw_oof.cpu(),
                "folds": fold_scores, "history": cycle_hist}, out_dir / "cycle.pt")

    print("[oof] stage 2 rollout net", flush=True)
    roll_net = build_model("unetex", IN_CH, 3, cfg).to(device)
    opt = torch.optim.AdamW(roll_net.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6)
    domain = cfg["domain"]
    dx = (domain["x_max"] - domain["x_min"]) / max(cfg["grid"]["nx"] - 1, 1)
    dy = (domain["y_max"] - domain["y_min"]) / max(cfg["grid"]["ny"] - 1, 1)
    lam_div = float(cfg["train"]["lambda_div"])
    lam_wall = float(cfg["train"]["lambda_wall"])
    weights = channel_weights(train.fields[:, ::20].reshape(-1, 3, *train.fields.shape[-2:]))
    n_frames = train.fields.shape[1]
    freq_oof = cycle_freq(raw_oof, train.mask)
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
            raw = raw_oof[pick]
            freq = freq_oof[pick]
            current = train.fields[pick, start]
            phase0 = read_phase(raw, current, mask)
            jitter = torch.rand(BATCH, device=device) < JITTER_P
            phase0 = phase0 + jitter * (torch.rand(BATCH, device=device) * 2 - 1) * PHASE_JITTER
            scale = 1.0 + jitter * (torch.rand(BATCH, device=device) * 2 - 1) * AMP_JITTER
            keep = (torch.rand(BATCH, device=device) >= DROP_PRIOR).float().reshape(-1, 1, 1, 1)
            state = current + torch.randn_like(current) * weights * NOISE * mask.unsqueeze(1)
            loss = geom.new_zeros(())
            for k in range(1, unroll + 1):
                t_prev = (train.times[pick, start + k - 1] - train.times[pick, start]).float()
                t_next = (train.times[pick, start + k] - train.times[pick, start]).float()
                prior_now = priors_at(raw, phase0 + 2.0 * math.pi * freq * t_prev, scale) * keep
                prior_next = priors_at(raw, phase0 + 2.0 * math.pi * freq * t_next, scale) * keep
                pred = roll_net(torch.cat([geom, state, prior_now, prior_next], dim=1))
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
                write_status(out_dir, _format_bar(f"oof roll ep{epoch}/{roll_epochs} unroll{unroll}", step, STEPS_PER_EPOCH, time.perf_counter() - started, float(loss.detach())))
        roll_net.eval()
        rel = rollout_bank(cycle_net, roll_net, val, 0.0)
        mean = rel.reshape(-1, 3).mean(dim=0)
        score = float(mean.mean())
        row = {"epoch": epoch, "unroll": unroll, "train_loss": running / STEPS_PER_EPOCH,
               "rel_l2_ux": float(mean[0]), "rel_l2_uy": float(mean[1]), "rel_l2_p": float(mean[2]),
               "rel_l2_mean": score, "last_uy": float(rel[:, -1, 1].mean())}
        history.append(row)
        print(
            f"[oof] roll epoch {epoch:03d} unroll {unroll}  train {row['train_loss']:.4e}  val 40s relL2 {score:.4f}  "
            f"ux {row['rel_l2_ux']:.4f}  uy {row['rel_l2_uy']:.4f}  p {row['rel_l2_p']:.4f}  last_uy {row['last_uy']:.4f}",
            flush=True,
        )
        if unroll == MAX_UNROLL:
            sched.step(score)
        if score < best:
            best = score
            stale = 0
            torch.save({"line": "cyclelock_oof", "tag": tag, "cycle": cycle_net.state_dict(), "roll": roll_net.state_dict(),
                        "epoch": epoch, "history": history, "cycle_history": cycle_hist, "folds": fold_scores,
                        "harmonics": HARMONICS, "config": {k: cfg[k] for k in cfg if not k.startswith("_")}}, out_dir / "best.pt")
        elif unroll == MAX_UNROLL:
            stale += 1
        if stale >= patience:
            print(f"[oof] roll early stopping at epoch {epoch}", flush=True)
            break
    (out_dir / "history.json").write_text(json.dumps({"folds": fold_scores, "cycle": cycle_hist, "roll": history}, indent=2), encoding="utf-8")
    ckpt = out_dir / "best.pt"
    exam(cfg, ckpt, h5, cases_root, eval_name)
    return ckpt


def exam(cfg: dict, ckpt: Path, h5: Path, cases_root: Path, eval_name: str) -> None:
    root = Path(cfg["_root"])
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pack.get("line") != "cyclelock_oof":
        raise SystemExit("这份权重不是第一条线修正版，拒绝考试")
    device = pick_device(str(cfg["train"]["device"]))
    cycle_net = build_model("unetex", 4, N_OUT, cfg).to(device)
    roll_net = build_model("unetex", IN_CH, 3, cfg).to(device)
    cycle_net.load_state_dict(pack["cycle"])
    roll_net.load_state_dict(pack["roll"])
    cycle_net.eval()
    roll_net.eval()
    val = CaseBank(h5, "val", cases_root, device, with_labels=False)
    test = CaseBank(h5, "test", cases_root, device, with_labels=False)
    scan = {str(p): float(rollout_bank(cycle_net, roll_net, val, p).mean()) for p in PULLS}
    chosen = min(PULLS, key=lambda p: scan[str(p)])
    cycle_buckets: dict[str, list[torch.Tensor]] = {}
    from flowproxy.lines.cycle import group_keys

    for cid, rel in zip(test.ids, cycle_from_first_frame(cycle_net, test)):
        for key in group_keys(cid):
            cycle_buckets.setdefault(key, []).append(rel)
    payload = {
        "line": "cyclelock_oof",
        "tag": pack["tag"],
        "ckpt": str(ckpt),
        "best_epoch": int(pack["epoch"]),
        "harmonics": HARMONICS,
        "folds": pack["folds"],
        "note": "从 t=160 的一帧真值出发连着猜 40 秒，只从这一帧读一次相位，不看升力。推演网络训练时用交叉分组的周期帧，并随机遮掉、打乱周期帧。",
        "val_pull_scan": scan,
        "pull_selected_on_val": chosen,
        "rollout": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, chosen), test.times),
        "rollout_no_prior": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, 0.0, use_prior=False), test.times),
        "one_step_true_prev": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, 0.0, true_prev=True), test.times)["mean_40s"],
        "one_step_true_prev_no_prior": pack_rollout(test.ids, rollout_bank(cycle_net, roll_net, test, 0.0, use_prior=False, true_prev=True), test.times)["mean_40s"],
        "cycle_only_first_frame": summarize(cycle_buckets),
    }
    dest = write_eval(root, eval_name, json.dumps(payload, indent=2), eval_name, frozenset())
    m = payload["rollout"]["mean_40s"]["all"]
    last = payload["rollout"]["last_step"]["all"]
    one = payload["one_step_true_prev"]["all"]
    print(
        f"[oof] pull {chosen} rollout 40s mean ux {m['rel_l2_ux']*100:.2f}% uy {m['rel_l2_uy']*100:.2f}% p {m['rel_l2_p']*100:.2f}%  "
        f"last uy {last['rel_l2_uy']*100:.2f}%  one-step uy {one['rel_l2_uy']*100:.2f}% -> {dest}",
        flush=True,
    )
