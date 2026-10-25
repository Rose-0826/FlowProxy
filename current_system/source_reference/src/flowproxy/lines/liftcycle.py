"""第二条线：形状给出稳定周期和频率，开头一小段升力给出相位。

网络只吃几何。训练时每帧的相位和频率由整段升力拟合得到，只作标签。
考试时不给任何流场帧：只用每个算例开头 WINDOW_S 秒的升力定相位，
之后整段 40 秒按形状给出的频率往前走。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from flowproxy.lines.cycle import (
    HARMONICS,
    N_OUT,
    CaseBank,
    cycle_field,
    cycle_freq,
    group_keys,
    rel_frame,
    summarize,
    window_phase,
)
from flowproxy.lines.cycle_train import train_cycle_net
from flowproxy.lines.isolate import claim_line, write_eval
from flowproxy.models import build_model
from flowproxy.train import pick_device

TAG = "liftcycle_long"
RIVALS = frozenset({"cyclelock_long"})
EVAL_NAME = "test_long_liftcycle.json"
WINDOW_S = 10.0


@torch.no_grad()
def predict_bank(net, bank: CaseBank, cases_root: Path, freq_source: str) -> list[torch.Tensor]:
    raw = net(bank.geom).float()
    f_geom = cycle_freq(raw, bank.mask)
    out = []
    for i, cid in enumerate(bank.ids):
        times = bank.times[i].double().cpu().numpy()
        t0 = float(times[0])
        if freq_source == "geom":
            f = float(f_geom[i])
            _, delta = window_phase(cases_root / cid, t0, WINDOW_S, f)
        else:
            f, delta = window_phase(cases_root / cid, t0, WINDOW_S, None)
        phase = torch.from_numpy((2.0 * math.pi * f * times - delta).astype(np.float32)).to(raw.device)
        pred = cycle_field(raw[i : i + 1].expand(phase.shape[0], -1, -1, -1), phase)
        out.append(rel_frame(pred, bank.fields[i], bank.mask[i]).cpu())
    return out


def score_bank(net, bank: CaseBank, cases_root: Path, freq_source: str) -> dict:
    rels = predict_bank(net, bank, cases_root, freq_source)
    buckets: dict[str, list[torch.Tensor]] = {}
    for i, (cid, rel) in enumerate(zip(bank.ids, rels)):
        times = bank.times[i].cpu().numpy()
        after = torch.from_numpy(times > float(times[0]) + WINDOW_S)
        for key in group_keys(cid):
            buckets.setdefault(key, []).append(rel)
        buckets.setdefault("after_window", []).append(rel[after])
    return summarize(buckets)


def run(cfg: dict, h5: Path, cases_root: Path, tag: str = TAG, eval_name: str = EVAL_NAME, epochs: int = 300, patience: int = 40) -> Path:
    root = Path(cfg["_root"])
    if not h5.is_file():
        raise SystemExit(f"缺数据 {h5}")
    out_dir, _lock = claim_line(root, tag, RIVALS)
    device = pick_device(str(cfg["train"]["device"]))
    train = CaseBank(h5, "train", cases_root, device)
    val = CaseBank(h5, "val", cases_root, device, with_labels=False)
    low = [cid for cid, r2 in zip(train.ids, train.r2) if r2 < 0.95]
    print(f"[lift] dir {out_dir}; train {len(train.ids)} cases, lift sine fit R2<0.95: {low}", flush=True)

    def validate(net) -> dict:
        return score_bank(net, val, cases_root, "geom")["all"]

    net, history = train_cycle_net(cfg, train, validate, out_dir, "lift", epochs, patience, device)
    ckpt = out_dir / "best.pt"
    torch.save(
        {
            "line": "liftcycle",
            "tag": tag,
            "cycle": net.state_dict(),
            "cycle_channels": [4, N_OUT],
            "harmonics": HARMONICS,
            "window_s": WINDOW_S,
            "history": history,
            "config": {k: cfg[k] for k in cfg if not k.startswith("_")},
        },
        ckpt,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    exam(cfg, ckpt, h5, cases_root, eval_name)
    return ckpt


def exam(cfg: dict, ckpt: Path, h5: Path, cases_root: Path, eval_name: str) -> None:
    root = Path(cfg["_root"])
    pack = torch.load(ckpt, map_location="cpu", weights_only=False)
    if pack.get("line") != "liftcycle":
        raise SystemExit("这份权重不是第二条线，拒绝考试")
    device = pick_device(str(cfg["train"]["device"]))
    net = build_model("unetex", 4, N_OUT, cfg).to(device)
    net.load_state_dict(pack["cycle"])
    net.eval()
    test = CaseBank(h5, "test", cases_root, device, with_labels=False)
    best_epoch = min(pack["history"], key=lambda r: r["rel_l2_mean"])["epoch"]
    payload = {
        "line": "liftcycle",
        "tag": pack["tag"],
        "ckpt": str(ckpt),
        "best_epoch": int(best_epoch),
        "window_s": WINDOW_S,
        "harmonics": HARMONICS,
        "note": "不给任何流场帧。形状给平均流、四对谐波模态和频率；开头 10 秒升力只用来定相位，之后整段按形状给的频率走。",
        "geom_freq": score_bank(net, test, cases_root, "geom"),
        "lift_freq": score_bank(net, test, cases_root, "lift"),
    }
    dest = write_eval(root, eval_name, json.dumps(payload, indent=2), eval_name, frozenset())
    row = payload["geom_freq"]["all"]
    print(
        f"[lift] test ux {row['rel_l2_ux']*100:.2f}% uy {row['rel_l2_uy']*100:.2f}% p {row['rel_l2_p']*100:.2f}% -> {dest}",
        flush=True,
    )
