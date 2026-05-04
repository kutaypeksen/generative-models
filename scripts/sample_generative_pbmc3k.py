"""Checkpoint'ten VAE / WGAN-GP / AAE ile sentetik örnek üretimi."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.aae import AEDecoder
from src.models.vae import VAE
from src.models.wgan_gp import WGGenerator
from src.pbmc3k_data import PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("vae", "wgan_gp", "aae"), required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-vae", type=Path, default=ROOT / "checkpoints" / "vae_pbmc3k.pt")
    p.add_argument("--checkpoint-wgan", type=Path, default=ROOT / "checkpoints" / "wgan_gp_pbmc3k.pt")
    p.add_argument("--checkpoint-aae", type=Path, default=ROOT / "checkpoints" / "aae_pbmc3k.pt")
    return p.parse_args()


def default_out_path(model: str) -> Path:
    return PROCESSED_DIR / f"pbmc3k_{model}_synthetic.npy"


@torch.no_grad()
def sample_vae(ckpt_path: Path, n: int, device: torch.device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_features = int(ckpt["n_features"])
    latent_dim = int(ckpt["latent_dim"])
    model = VAE(n_features=n_features, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    z = torch.randn(n, latent_dim, device=device)
    x = model.decode(z).cpu().numpy().astype(np.float32)
    return x


@torch.no_grad()
def sample_wgan(ckpt_path: Path, n: int, device: torch.device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_features = int(ckpt["n_features"])
    noise_dim = int(ckpt["noise_dim"])
    G = WGGenerator(n_features=n_features, noise_dim=noise_dim)
    G.load_state_dict(ckpt["generator"])
    G.eval().to(device)
    z = torch.randn(n, noise_dim, device=device)
    x = G(z).cpu().numpy().astype(np.float32)
    return x


@torch.no_grad()
def sample_aae(ckpt_path: Path, n: int, device: torch.device) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_features = int(ckpt["n_features"])
    latent_dim = int(ckpt["latent_dim"])
    dec = AEDecoder(n_features=n_features, latent_dim=latent_dim)
    dec.load_state_dict(ckpt["decoder"])
    dec.eval().to(device)
    z = torch.randn(n, latent_dim, device=device)
    x = dec(z).cpu().numpy().astype(np.float32)
    return x


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.model == "vae":
        ck = args.checkpoint_vae
    elif args.model == "wgan_gp":
        ck = args.checkpoint_wgan
    else:
        ck = args.checkpoint_aae

    if not ck.is_file():
        raise SystemExit(f"Checkpoint yok: {ck}")

    if args.model == "vae":
        arr = sample_vae(ck, args.n, device)
    elif args.model == "wgan_gp":
        arr = sample_wgan(ck, args.n, device)
    else:
        arr = sample_aae(ck, args.n, device)

    out = default_out_path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"model={args.model} checkpoint={ck}")
    print(f"Kaydedildi: {out} shape={arr.shape}")


if __name__ == "__main__":
    main()
