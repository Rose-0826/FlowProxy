from __future__ import annotations

import os
from pathlib import Path

# 父进程若一直握着同一个 HDF5，DataLoader 子进程再打开会被文件锁堵住。
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import time_feature_channels


def save_cases_h5(path: str | Path, cases: list[dict], split: dict[str, list[str]] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        grp = f.create_group("cases")
        for case in cases:
            g = grp.create_group(case["case_id"])
            g.create_dataset("geom", data=case["geom"], compression="gzip")
            g.create_dataset("fields", data=case["fields"], compression="gzip")
            g.create_dataset("times", data=case["times"])
            g.create_dataset("mask", data=case["mask"], compression="gzip")
            g.attrs["Re"] = case["Re"]
            g.attrs["cx"] = case["cx"]
            g.attrs["cy"] = case["cy"]
            g.attrs["D"] = case["D"]
            g.attrs["shape"] = case["shape"]
            if "forces" in case:
                g.create_dataset("forces", data=case["forces"])
        if split:
            s = f.create_group("splits")
            for name, ids in split.items():
                s.create_dataset(name, data=np.array(ids, dtype="S"))


def lift_phase(field_times: np.ndarray, cl_times: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """用升力随时间的摆动，得到每一帧涡转到哪一边。返回 sin、cos，形状 [2, T]。"""
    signal = np.interp(field_times, cl_times, cl).astype(np.float64)
    signal = signal - signal.mean()
    count = signal.shape[0]
    spectrum = np.fft.fft(signal)
    mask = np.zeros(count)
    if count % 2 == 0:
        mask[0] = mask[count // 2] = 1
        mask[1 : count // 2] = 2
    else:
        mask[0] = 1
        mask[1 : (count + 1) // 2] = 2
    angle = np.angle(np.fft.ifft(spectrum * mask))
    return np.stack([np.sin(angle), np.cos(angle)], axis=0).astype(np.float32)


def read_lift(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """续算会从新的起始时刻另开一个 coefficient.dat，这里把所有段拼起来按时间排好。"""
    hits = sorted(case_dir.glob("postProcessing/forceCoeffs*/**/coefficient.dat"))
    if not hits:
        raise FileNotFoundError(f"{case_dir} 没有升力记录")
    times: list[float] = []
    values: list[float] = []
    for hit in hits:
        for line in hit.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            times.append(float(cols[0]))
            values.append(float(cols[4]))
    if len(times) < 8:
        raise RuntimeError(f"{case_dir.name} 的升力记录太短")
    t, index = np.unique(np.asarray(times), return_index=True)
    return t, np.asarray(values)[index]


def close_h5(dataset) -> None:
    handle = getattr(dataset, "_file", None)
    if handle is not None:
        handle.close()
        dataset._file = None


def load_case_ids(path: str | Path, split: str | None = None) -> list[str]:
    with h5py.File(path, "r") as f:
        if split and "splits" in f and split in f["splits"]:
            raw = f["splits"][split][()]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
        return list(f["cases"].keys())


class FlowSnapshotDataset(Dataset):
    """几何通道 → 单帧 Ux/Uy/p。对应 UNetEx 几何分支。"""

    def __init__(
        self,
        h5_path: str | Path,
        split: str | None = "train",
        frame: str = "mid",
        with_time: bool = False,
        with_phase: bool = False,
        cases_root: str | Path | None = None,
    ):
        self.h5_path = str(h5_path)
        self.ids = load_case_ids(h5_path, split)
        self.frame = frame
        self.with_time = with_time
        self.with_phase = with_phase
        self._file = None
        self.phase: dict[str, np.ndarray] = {}
        if with_phase:
            if cases_root is None:
                raise ValueError("用相位时要给出算例目录")
            root = Path(cases_root)
            with h5py.File(self.h5_path, "r") as handle:
                for cid in self.ids:
                    field_times = np.asarray(handle["cases"][cid]["times"], dtype=np.float64)
                    cl_times, cl = read_lift(root / cid)
                    self.phase[cid] = lift_phase(field_times, cl_times, cl)
        self.samples: list[tuple[str, int | None]] = []
        if frame == "all":
            with h5py.File(self.h5_path, "r") as handle:
                for cid in self.ids:
                    n_frames = int(handle["cases"][cid]["fields"].shape[0])
                    self.samples.extend((cid, t) for t in range(n_frames))
        else:
            self.samples = [(cid, None) for cid in self.ids]

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
        cid, t_index = self.samples[idx]
        g = self._h5()["cases"][cid]
        fields = g["fields"]
        if t_index is None:
            t_index = fields.shape[0] // 2 if self.frame == "mid" else int(self.frame)
        geom = np.asarray(g["geom"][()], dtype=np.float32)
        if self.with_time and self.with_phase:
            raise RuntimeError("时刻和相位不要同时加")
        if self.with_time:
            t = float(np.asarray(g["times"][t_index]))
            extra = time_feature_channels(t, geom.shape[-2], geom.shape[-1])
            geom = np.concatenate([geom, extra], axis=0)
        if self.with_phase:
            angle = self.phase[cid][:, t_index]
            extra = np.stack(
                [
                    np.full(geom.shape[-2:], angle[0], dtype=np.float32),
                    np.full(geom.shape[-2:], angle[1], dtype=np.float32),
                ],
                axis=0,
            )
            geom = np.concatenate([geom, extra], axis=0)
        x = torch.from_numpy(geom)
        y = torch.from_numpy(np.asarray(fields[t_index], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(g["mask"][()], dtype=np.float32))
        return {"x": x, "y": y, "mask": mask, "case_id": cid, "Re": float(g.attrs["Re"])}


class FlowSequenceDataset(Dataset):
    """几何 + 历史 k 帧 → 未来 t 帧。对应 FNO / 时空 UNet 对照。"""

    def __init__(self, h5_path: str | Path, split: str | None = "train", history_k: int = 4, future_t: int = 1):
        self.h5_path = str(h5_path)
        self.history_k = history_k
        self.future_t = future_t
        self.ids = load_case_ids(h5_path, split)
        self._file = None
        self.samples: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as f:
            for cid in self.ids:
                n = f["cases"][cid]["fields"].shape[0]
                last = n - history_k - future_t
                if last < 0:
                    continue
                for start in range(last + 1):
                    self.samples.append((cid, start))

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
        g = self._h5()["cases"][cid]
        fields = g["fields"]
        _, _, height, width = fields.shape
        geom = np.asarray(g["geom"][()], dtype=np.float32)
        hist = np.asarray(fields[start : start + self.history_k], dtype=np.float32).reshape(-1, height, width)
        fut = np.asarray(
            fields[start + self.history_k : start + self.history_k + self.future_t],
            dtype=np.float32,
        )
        y = fut.reshape(-1, height, width)
        x = np.concatenate([geom, hist], axis=0)
        mask = np.asarray(g["mask"][()], dtype=np.float32)
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "case_id": cid,
            "Re": float(g.attrs["Re"]),
        }
