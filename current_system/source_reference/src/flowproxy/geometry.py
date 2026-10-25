from __future__ import annotations

import numpy as np


def make_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = cfg["domain"]
    g = cfg["grid"]
    x = np.linspace(d["x_min"], d["x_max"], g["nx"], dtype=np.float32)
    y = np.linspace(d["y_min"], d["y_max"], g["ny"], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return x, y, xx, yy


# 简化侧视车：车长固定。车头在 cx，车底在 cy，车高是 D，后背倾角从水平量起。
CAR_LENGTH = 3.6


def car_polygon(
    x0: float,
    y_bottom: float,
    height: float,
    backlight_deg: float,
    length: float = CAR_LENGTH,
) -> np.ndarray:
    """四边形侧视：竖直车头、平车底、平车顶、斜车尾。点序为逆时针。"""
    angle = np.deg2rad(float(backlight_deg))
    run = float(height) / np.tan(angle)
    if run <= 0 or run >= 0.9 * float(length):
        raise ValueError(f"后背倾角 {backlight_deg} 和车高 {height} 让车尾斜边盖住了整辆车")
    h = float(height)
    length = float(length)
    y0 = float(y_bottom)
    return np.array(
        [
            [x0, y0],
            [x0 + length, y0],
            [x0 + length - run, y0 + h],
            [x0, y0 + h],
        ],
        dtype=np.float64,
    )


def polygon_sdf(xx: np.ndarray, yy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """多边形符号距离。外部为正，内部为负。"""
    dist = np.full(xx.shape, np.inf, dtype=np.float64)
    inside = np.zeros(xx.shape, dtype=bool)
    count = len(poly)
    for i in range(count):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % count]
        edge_x = x2 - x1
        edge_y = y2 - y1
        t = np.clip(((xx - x1) * edge_x + (yy - y1) * edge_y) / (edge_x * edge_x + edge_y * edge_y), 0.0, 1.0)
        dx = xx - (x1 + t * edge_x)
        dy = yy - (y1 + t * edge_y)
        dist = np.minimum(dist, np.hypot(dx, dy))
        crosses = (y1 > yy) != (y2 > yy)
        x_hit = x1 + (yy - y1) * edge_x / (edge_y if edge_y != 0 else 1e-30)
        inside ^= crosses & (xx < x_hit)
    sign = np.where(inside, -1.0, 1.0)
    return (sign * dist).astype(np.float32)


def obstacle_sdf(
    xx: np.ndarray,
    yy: np.ndarray,
    shape: str,
    cx: float,
    cy: float,
    D: float,
    backlight: float | None = None,
    length: float | None = None,
) -> np.ndarray:
    """障碍物符号距离：流体区为正，内部为负。与 DeepCFD 的 SDF 通道同义。"""
    dx = xx - cx
    dy = yy - cy
    if shape == "cylinder":
        return np.sqrt(dx * dx + dy * dy) - 0.5 * D
    if shape == "square":
        return np.maximum(np.abs(dx), np.abs(dy)) - 0.5 * D
    if shape == "car":
        if backlight is None:
            raise ValueError("车形需要后背倾角 backlight")
        poly = car_polygon(cx, cy, D, backlight, CAR_LENGTH if length is None else length)
        return polygon_sdf(xx, yy, poly)
    raise ValueError(f"unknown shape: {shape}")


def wall_sdf(yy: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """上下壁面距离（通道侧壁）。"""
    return np.minimum(yy - y_min, y_max - yy).astype(np.float32)


def geometry_channels(
    cfg: dict,
    shape: str | None = None,
    cx: float | None = None,
    cy: float | None = None,
    D: float | None = None,
    Re: float = 100.0,
    backlight: float | None = None,
    length: float | None = None,
) -> dict[str, np.ndarray]:
    phys = cfg["physics"]
    domain = cfg["domain"]
    shape = shape or phys["shape"]
    cx = phys["cx"] if cx is None else cx
    cy = phys["cy"] if cy is None else cy
    D = phys["D"] if D is None else D
    _, _, xx, yy = make_grid(cfg)
    sdf_obs = obstacle_sdf(xx, yy, shape, cx, cy, D, backlight=backlight, length=length).astype(np.float32)
    mask = (sdf_obs > 0).astype(np.float32)
    sdf_wall = wall_sdf(yy, domain["y_min"], domain["y_max"])
    re_ch = np.full_like(sdf_obs, np.float32(Re / 200.0))
    geom = np.stack([sdf_obs, mask, sdf_wall, re_ch], axis=0)
    return {
        "xx": xx.astype(np.float32),
        "yy": yy.astype(np.float32),
        "sdf_obs": sdf_obs,
        "mask": mask,
        "sdf_wall": sdf_wall,
        "geom": geom,
    }


# 数据里涡脱落大约 6–10 秒一圈（0.10–0.16 Hz）。两侧留余量，每隔 0.005 Hz 放一对正弦、余弦。
TIME_FREQS_HZ = tuple(round(0.06 + 0.005 * i, 5) for i in range(33))


def time_feature_channels(
    t: float,
    height: int,
    width: int,
    t0: float = 40.0,
    t1: float = 80.0,
) -> np.ndarray:
    """把物理时刻铺成整幅图上的通道，让几何模型知道这一帧在周期里的哪一拍。"""
    span = max(t1 - t0, 1e-6)
    feats = [np.full((height, width), np.float32((t - t0) / span))]
    for freq in TIME_FREQS_HZ:
        angle = np.float32(2.0 * np.pi * freq * t)
        feats.append(np.full((height, width), np.sin(angle), dtype=np.float32))
        feats.append(np.full((height, width), np.cos(angle), dtype=np.float32))
    return np.stack(feats, axis=0)
