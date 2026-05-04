"""WGAN-GP: vektör uzayında üretici ve eleştirmen (ölçeklenmiş ifade profilleri)."""

from __future__ import annotations

import torch
import torch.nn as nn


class WGANCritic(nn.Module):
    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WGGenerator(nn.Module):
    def __init__(
        self,
        n_features: int,
        noise_dim: int = 32,
        hidden_dims: tuple[int, ...] = (256, 512),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = noise_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def gradient_penalty(critic: WGANCritic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, device=real.device, dtype=real.dtype)
    interp = alpha * real + (1.0 - alpha) * fake
    interp = interp.requires_grad_(True)
    out = critic(interp)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
