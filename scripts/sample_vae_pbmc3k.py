"""Eğitilmiş VAE ile sentetik hücre vektörleri üretir (ölçeklenmiş HVG uzayında)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vae import VAE
from src.pbmc3k_data import PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "vae_pbmc3k.pt")
    p.add_argument("--n", type=int, default=2000, help="Üretilecek sentetik hücre sayısı")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "pbmc3k_vae_synthetic.npy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    n_features = int(ckpt["n_features"])
    latent_dim = int(ckpt["latent_dim"])
    model = VAE(n_features=n_features, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    z = torch.randn(args.n, latent_dim, device=device)
    synth = model.decode(z).cpu().numpy().astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, synth)
    print(f"Kaydedildi: {args.out} shape={synth.shape}")


if __name__ == "__main__":
    main()
