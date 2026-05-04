"""Basit tam bağlı VAE — PBMC3k ölçeklenmiş HVG vektörleri için."""

from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 32, hidden_dims: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec_layers.append(nn.Linear(prev, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Öklid rekonstrüksiyon + KL(N(mu,σ²) || N(0,I))."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl
