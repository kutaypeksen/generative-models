"""
Uc dondurulmus uretici (VAE, WGAN-GP, AAE) + ortak gurultu z ile FusionMixer egitimi.
Ayird macı loss + hafif ortalama ankraj (kararsizlik azaltir).
Once: train_*_pbmc3k.py ile checkpointler hazir olmali.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.aae import AEDecoder
from src.models.fusion_ensemble import FusionCritic, FusionMixer
from src.models.vae import VAE
from src.models.wgan_gp import WGGenerator
from src.pbmc3k_data import Pbmc3kExpressionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-fusion", type=float, default=2e-4)
    p.add_argument("--lr-critic", type=float, default=2e-4)
    p.add_argument("--lambda-mean-anchor", type=float, default=0.05, help="Ortalama birlesime MSE cezasi")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-vae", type=Path, default=ROOT / "checkpoints" / "vae_pbmc3k.pt")
    p.add_argument("--checkpoint-wgan", type=Path, default=ROOT / "checkpoints" / "wgan_gp_pbmc3k.pt")
    p.add_argument("--checkpoint-aae", type=Path, default=ROOT / "checkpoints" / "aae_pbmc3k.pt")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints" / "fusion_ensemble_pbmc3k.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for ck in (args.checkpoint_vae, args.checkpoint_wgan, args.checkpoint_aae):
        if not ck.is_file():
            raise SystemExit(f"Checkpoint eksik: {ck}")

    ck_v = torch.load(args.checkpoint_vae, map_location="cpu")
    ck_w = torch.load(args.checkpoint_wgan, map_location="cpu")
    ck_a = torch.load(args.checkpoint_aae, map_location="cpu")

    n_features = int(ck_v["n_features"])
    latent_v = int(ck_v["latent_dim"])
    latent_a = int(ck_a["latent_dim"])
    noise_w = int(ck_w["noise_dim"])
    if not (latent_v == latent_a == noise_w):
        raise SystemExit(f"Latent/gurultu boyutlari eslesmiyor: VAE={latent_v}, WGAN={noise_w}, AAE={latent_a}")

    latent_dim = latent_v
    device = torch.device(args.device)

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

    fusion = FusionMixer(n_features=n_features).to(device)
    critic = FusionCritic(n_features=n_features).to(device)
    opt_f = torch.optim.Adam(fusion.parameters(), lr=args.lr_fusion, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    ds_tr = Pbmc3kExpressionDataset("train")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)

    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        fusion.train()
        critic.train()
        lf_acc = lc_acc = 0.0
        nb = 0
        for real in dl_tr:
            real = real.to(device)
            bs = real.size(0)
            ones = torch.ones(bs, device=device)
            zeros = torch.zeros(bs, device=device)
            z = torch.randn(bs, latent_dim, device=device)

            with torch.no_grad():
                xv = vae.decode(z)
                xw = G(z)
                xa = dec(z)
            cat = torch.cat([xv, xw, xa], dim=1)
            x_fused = fusion(cat)
            mean_blend = (xv + xw + xa) / 3.0

            opt_c.zero_grad()
            logits_real = critic(real)
            logits_fake = critic(x_fused.detach())
            loss_c = bce(logits_real, ones) + bce(logits_fake, zeros)
            loss_c.backward()
            opt_c.step()

            opt_f.zero_grad()
            logits_fool = critic(x_fused)
            loss_f = bce(logits_fool, ones) + args.lambda_mean_anchor * mse(x_fused, mean_blend)
            loss_f.backward()
            opt_f.step()

            lf_acc += loss_f.item()
            lc_acc += loss_c.item()
            nb += 1

        print(f"epoch {epoch:03d}  loss_fusion={lf_acc/max(nb,1):.4f}  loss_critic={lc_acc/max(nb,1):.4f}")

        torch.save(
            {
                "fusion": fusion.state_dict(),
                "n_features": n_features,
                "latent_dim": latent_dim,
                "epoch": epoch,
            },
            args.save,
        )

    print(f"Tamam | {args.save}")


if __name__ == "__main__":
    main()
