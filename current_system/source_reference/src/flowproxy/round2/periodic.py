"""几何先验加启动编码器。z 作为额外通道拼进 UNet，相位按解析式推进。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowproxy.models.unetex import UNetEx

K = 4
Z_DIM = 16
N_COEFF = 3 * (1 + 2 * K)
BETA = 0.3


class StartupEncoder(nn.Module):
    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.geom = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32 + 16, z_dim + 3)

    def forward(self, geom: torch.Tensor, frames: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """frames [B,M,3,H,W]，tau [B,M] 相对启动末端，小于等于 0。"""
        batch, steps = frames.shape[:2]
        clock = tau.reshape(batch, steps, 1, 1, 1).expand(-1, -1, 1, frames.shape[-2], frames.shape[-1])
        packed = torch.cat([frames, clock], dim=2).reshape(batch * steps, 4, frames.shape[-2], frames.shape[-1])
        feat = self.frame(packed).reshape(batch, steps, 32)
        _out, hidden = self.gru(feat)
        geo = self.geom(geom).reshape(batch, 16)
        raw = self.head(torch.cat([hidden[-1], geo], dim=1))
        z = raw[:, :Z_DIM]
        phase = raw[:, Z_DIM : Z_DIM + 2]
        phase = phase / phase.norm(dim=1, keepdim=True).clamp_min(1e-6)
        nu = raw[:, -1]
        return z, phase, nu


class PeriodicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StartupEncoder()
        self.decoder = UNetEx(4 + 3 + Z_DIM, N_COEFF, filters=[8, 16, 32, 32], kernel_size=5, layers=2)
        self.freq_head = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def omega_g(self, geom: torch.Tensor) -> torch.Tensor:
        # 几何频率先验，正值，初值靠近 1 rad/s。
        return F.softplus(self.freq_head(geom).squeeze(1)) + 0.3

    def omega(self, geom: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return self.omega_g(geom) * torch.exp(BETA * torch.tanh(nu))

    def coefficients(self, geom: torch.Tensor, coord: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        zmap = z[:, :, None, None].expand(-1, -1, geom.shape[-2], geom.shape[-1])
        return self.decoder(torch.cat([geom, coord, zmap], dim=1))

    def encode(self, geom: torch.Tensor, frames: torch.Tensor, tau: torch.Tensor):
        z, phase, nu = self.encoder(geom, frames, tau)
        theta = torch.atan2(phase[:, 1], phase[:, 0])
        return z, theta, nu


def generate(coeff: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """theta [B] 或 [B,T]。返回与 theta 对齐的场。"""
    single = theta.ndim == 1
    if single:
        theta = theta[:, None]
    mean = coeff[:, None, 0:3]
    out = mean.expand(-1, theta.shape[1], -1, -1, -1).clone()
    view = theta.reshape(theta.shape[0], theta.shape[1], 1, 1, 1)
    for k in range(1, K + 1):
        base = 3 + 6 * (k - 1)
        amp_c = coeff[:, None, base : base + 3]
        amp_s = coeff[:, None, base + 3 : base + 6]
        out = out + torch.cos(k * view) * amp_c + torch.sin(k * view) * amp_s
    return out[:, 0] if single else out


def phase_derivative(coeff: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """∂q/∂θ，theta [B]。"""
    out = torch.zeros(coeff.shape[0], 3, *coeff.shape[-2:], device=coeff.device, dtype=coeff.dtype)
    for k in range(1, K + 1):
        base = 3 + 6 * (k - 1)
        out = out + k * (
            -coeff[:, base : base + 3] * torch.sin(k * theta)[:, None, None, None]
            + coeff[:, base + 3 : base + 6] * torch.cos(k * theta)[:, None, None, None]
        )
    return out


def _flat_fluid(field: torch.Tensor, mask: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """field [M,3,H,W] 或 [3,H,W]，按通道尺度白化后只留流体点，拉成一维。"""
    if field.ndim == 3:
        field = field.unsqueeze(0)
    whitened = field / scales.reshape(1, 3, 1, 1)
    fluid = mask > 0.5
    return whitened[:, :, fluid].reshape(-1)


def thin_basis(j_theta: torch.Tensor, j_omega: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    """两列张成的子空间，最多两列。近相关方向按相对奇异值丢掉。"""
    columns = torch.stack([j_theta, j_omega], dim=1)
    norms = columns.norm(dim=0).clamp_min(1e-8)
    columns = columns / norms
    gram = columns.T @ columns
    evals, evecs = torch.linalg.eigh(gram)
    keep = evals > tol * evals.max().clamp_min(1e-12)
    if not bool(keep.any()):
        return columns[:, :0]
    basis = columns @ evecs[:, keep]
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return basis.detach()


def support_fields(model: PeriodicModel, geom, coord, z, theta, nu, tau) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = model.coefficients(geom, coord, z)
    omega = model.omega(geom, nu)
    # tau <= 0，末端为 0。θ(t)=θs+ω τ
    theta_t = theta[:, None] + omega[:, None] * tau
    pred = generate(coeff, theta_t)
    dtheta = phase_derivative(coeff, theta)
    return pred, dtheta


def projected_spatial_loss(pred, truth, dtheta, tau, mask, scales, z, z0) -> torch.Tensor:
    """每个算例单独做两列投影。Q 不回传，残差对 z 保留梯度。"""
    losses = []
    for i in range(pred.shape[0]):
        residual = _flat_fluid(pred[i] - truth[i], mask[i], scales)
        j_theta = _flat_fluid(dtheta[i], mask[i], scales)
        # 启动各时刻共用同一 ∂q/∂θ 在 θs 处的值，频率方向再乘该时刻的 τ。
        pieces_t = []
        pieces_w = []
        for j in range(pred.shape[1]):
            pieces_t.append(j_theta)
            pieces_w.append(j_theta * tau[i, j])
        basis = thin_basis(torch.cat(pieces_t), torch.cat(pieces_w))
        if basis.shape[1] == 0:
            perp = residual
        else:
            perp = residual - basis @ (basis.T @ residual)
        count = max(int(residual.shape[0]), 1)
        losses.append(0.5 * perp.square().sum() / count + 1e-3 * (z[i] - z0[i]).square().sum() / z.shape[1])
    return torch.stack(losses).mean()


INNER_LR = {"z": 1e-2, "theta": 1e-2, "nu": 5e-3}


def variable_updates(model, geom, coord, support, tau, mask, scales, z, theta, nu, z0, nu0, count: int, projected: bool):
    """一阶内循环。梯度截断，状态仍连回编码器给出的初值。"""
    for step in range(count):
        phase_step = step % 2 == 0
        pred, dtheta = support_fields(model, geom, coord, z, theta, nu, tau)
        if phase_step or not projected:
            loss = joint_support_loss(pred, support, mask, scales, z, z0, nu, nu0)
            grads = torch.autograd.grad(loss, [theta, nu] if phase_step else [z], retain_graph=False)
            if phase_step:
                theta = theta - INNER_LR["theta"] * grads[0].detach()
                nu = nu - INNER_LR["nu"] * grads[1].detach()
            else:
                z = z - INNER_LR["z"] * grads[0].detach()
        else:
            loss = projected_spatial_loss(pred, support, dtheta, tau, mask, scales, z, z0)
            grad_z, = torch.autograd.grad(loss, [z])
            z = z - INNER_LR["z"] * grad_z.detach()
    return z, theta, nu


def joint_support_loss(pred, truth, mask, scales, z, z0, nu, nu0) -> torch.Tensor:
    fluid = (mask > 0.5)[:, None, None]
    err = (pred - truth) / scales.reshape(1, 1, 3, 1, 1)
    mse = (err.square() * fluid).sum() / fluid.sum().clamp_min(1) / 3
    return mse + 1e-3 * (z - z0).square().mean() + 1e-3 * (nu - nu0).square().mean()
