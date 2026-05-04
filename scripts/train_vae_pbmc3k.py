"""PBMC3K ölçeklenmiş HVG üzerinde kısa VAE eğitimi (demonstrasyon)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.vae import VAE, vae_loss
from src.pbmc3k_data import Pbmc3kExpressionDataset, PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints" / "vae_pbmc3k.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds_tr = Pbmc3kExpressionDataset("train")
    ds_va = Pbmc3kExpressionDataset("val")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = VAE(n_features=ds_tr.n_features, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        n_tr = 0
        for batch in dl_tr:
            x = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            loss_tr += loss.item() * x.size(0)
            n_tr += x.size(0)
        loss_tr /= max(n_tr, 1)

        model.eval()
        loss_va = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar)
                loss_va += loss.item() * x.size(0)
                n_va += x.size(0)
        loss_va /= max(n_va, 1)

        print(f"epoch {epoch:03d}  train_loss={loss_tr:.4f}  val_loss={loss_va:.4f}")
        if loss_va < best_val:
            best_val = loss_va
            torch.save(
                {
                    "model": model.state_dict(),
                    "n_features": ds_tr.n_features,
                    "latent_dim": args.latent_dim,
                    "epoch": epoch,
                    "val_loss": loss_va,
                },
                args.save,
            )
            print(f"  kaydedildi: {args.save}")

    print(f"Tamam. En iyi val_loss ~ {best_val:.4f} | {args.save}")


if __name__ == "__main__":
    main()
