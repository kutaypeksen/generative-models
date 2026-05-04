"""Adversarial Autoencoder: rekonstrüksiyon + latent uzayda önsel eşleme."""

from __future__ import annotations

import torch
import torch.nn as nn


class AAEncoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dims: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dims: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentCritic(nn.Module):
    """Önsel N(0,I) ile encoder çıktısını ayırt eder."""

    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
