"""VAE + WGAN-GP + AAE ortak z ile ortalama veya fusion mixer ile sentetik uretim."""
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
from src.models.fusion_ensemble import FusionMixer
from src.models.vae import VAE
from src.models.wgan_gp import WGGenerator
from src.pbmc3k_data import PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("mean", "fusion"), required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-vae", type=Path, default=ROOT / "checkpoints" / "vae_pbmc3k.pt")
    p.add_argument("--checkpoint-wgan", type=Path, default=ROOT / "checkpoints" / "wgan_gp_pbmc3k.pt")
    p.add_argument("--checkpoint-aae", type=Path, default=ROOT / "checkpoints" / "aae_pbmc3k.pt")
    p.add_argument("--checkpoint-fusion", type=Path, default=ROOT / "checkpoints" / "fusion_ensemble_pbmc3k.pt")
    return p.parse_args()


def _build_generators(
    device: torch.device,
    ck_v_path: Path,
    ck_w_path: Path,
    ck_a_path: Path,
):
    ck_v = torch.load(ck_v_path, map_location="cpu")
    ck_w = torch.load(ck_w_path, map_location="cpu")
    ck_a = torch.load(ck_a_path, map_location="cpu")
    n_features = int(ck_v["n_features"])
    latent_dim = int(ck_v["latent_dim"])
    if int(ck_a["latent_dim"]) != latent_dim or int(ck_w["noise_dim"]) != latent_dim:
        raise SystemExit("Checkpoint latent/gurultu boyutlari uyusmuyor")

    vae = VAE(n_features=n_features, latent_dim=latent_dim).to(device)
    vae.load_state_dict(ck_v["model"])
    vae.eval()

    G = WGGenerator(n_features=n_features, noise_dim=latent_dim).to(device)
    G.load_state_dict(ck_w["generator"])
    G.eval()

    dec = AEDecoder(n_features=n_features, latent_dim=latent_dim).to(device)
    dec.load_state_dict(ck_a["decoder"])
    dec.eval()

    for p in list(vae.parameters()) + list(G.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    return vae, G, dec, n_features, latent_dim


@torch.no_grad()
def main() -> None:
    args = parse_args()
    for p in (args.checkpoint_vae, args.checkpoint_wgan, args.checkpoint_aae):
        if not p.is_file():
            raise SystemExit(f"Checkpoint yok: {p}")

    device = torch.device(args.device)
    vae, G, dec, n_features, latent_dim = _build_generators(
        device, args.checkpoint_vae, args.checkpoint_wgan, args.checkpoint_aae
    )

    fusion_mod: FusionMixer | None = None
    if args.kind == "fusion":
        if not args.checkpoint_fusion.is_file():
            raise SystemExit(f"Fusion checkpoint yok: {args.checkpoint_fusion}")
        ck_f = torch.load(args.checkpoint_fusion, map_location="cpu")
        fusion_mod = FusionMixer(n_features=int(ck_f["n_features"])).to(device)
        fusion_mod.load_state_dict(ck_f["fusion"])
        fusion_mod.eval()

    out_parts: list[np.ndarray] = []
    remaining = args.n
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        z = torch.randn(bs, latent_dim, device=device)
        xv = vae.decode(z)
        xw = G(z)
        xa = dec(z)
        if args.kind == "mean":
            x = (xv + xw + xa) / 3.0
        else:
            assert fusion_mod is not None
            cat = torch.cat([xv, xw, xa], dim=1)
            x = fusion_mod(cat)
        out_parts.append(x.cpu().numpy().astype(np.float32))
        remaining -= bs

    arr = np.concatenate(out_parts, axis=0)
    suffix = "ensemble_mean_synthetic" if args.kind == "mean" else "ensemble_fusion_synthetic"
    out = PROCESSED_DIR / f"pbmc3k_{suffix}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"kind={args.kind} Kaydedildi: {out} shape={arr.shape}")


if __name__ == "__main__":
    main()
