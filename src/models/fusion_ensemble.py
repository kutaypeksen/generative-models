"""Uc sinir agi ureticinin ciktisini birlestiren fusion agi (ortalama uzayi ogrenir)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMixer(nn.Module):
    """x_cat = [x_vae; x_wgan; x_aae] -> tek sentetik profil."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (768, 512),
    ) -> None:
        super().__init__()
        in_dim = n_features * 3
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2, inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        return self.net(x_cat)


class FusionCritic(nn.Module):
    """Sentetik / gercek ayird etmek icin hafif elestirmen."""

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
