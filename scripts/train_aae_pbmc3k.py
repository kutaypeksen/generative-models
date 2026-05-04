"""PBMC3K ölçeklenmiş HVG üzerinde Adversarial Autoencoder eğitimi."""
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

from src.models.aae import AAEncoder, AEDecoder, LatentCritic
from src.pbmc3k_data import Pbmc3kExpressionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--lr-ae", type=float, default=1e-3)
    p.add_argument("--lr-d", type=float, default=1e-3)
    p.add_argument("--lambda-adv", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints" / "aae_pbmc3k.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds_tr = Pbmc3kExpressionDataset("train")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device(args.device)
    enc = AAEncoder(n_features=ds_tr.n_features, latent_dim=args.latent_dim).to(device)
    dec = AEDecoder(n_features=ds_tr.n_features, latent_dim=args.latent_dim).to(device)
    d_lat = LatentCritic(latent_dim=args.latent_dim).to(device)

    opt_ae = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr_ae)
    opt_d = torch.optim.Adam(d_lat.parameters(), lr=args.lr_d)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        d_lat.train()
        loss_rec_acc = 0.0
        loss_d_acc = 0.0
        n_batches = 0
        for x in dl_tr:
            x = x.to(device)
            bs = x.size(0)
            ones = torch.ones(bs, device=device)
            zeros = torch.zeros(bs, device=device)

            opt_d.zero_grad()
            z_prior = torch.randn(bs, args.latent_dim, device=device)
            enc_x = enc(x)
            logits_prior = d_lat(z_prior)
            logits_fake = d_lat(enc_x.detach())
            loss_d = bce(logits_prior, ones) + bce(logits_fake, zeros)
            loss_d.backward()
            opt_d.step()

            opt_ae.zero_grad()
            enc_x = enc(x)
            recon = dec(enc_x)
            loss_rec = mse(recon, x)
            logits_adv = d_lat(enc_x)
            loss_adv = bce(logits_adv, ones)
            loss_ae = loss_rec + args.lambda_adv * loss_adv
            loss_ae.backward()
            opt_ae.step()

            loss_rec_acc += loss_rec.item()
            loss_d_acc += loss_d.item()
            n_batches += 1

        loss_rec_acc /= max(n_batches, 1)
        loss_d_acc /= max(n_batches, 1)
        print(f"epoch {epoch:03d}  recon={loss_rec_acc:.4f}  loss_d={loss_d_acc:.4f}")

        torch.save(
            {
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "latent_critic": d_lat.state_dict(),
                "n_features": ds_tr.n_features,
                "latent_dim": args.latent_dim,
                "epoch": epoch,
            },
            args.save,
        )

    print(f"Tamam | son checkpoint: {args.save}")


if __name__ == "__main__":
    main()
