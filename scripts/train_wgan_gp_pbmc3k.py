"""PBMC3K ölçeklenmiş HVG üzerinde WGAN-GP eğitimi."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.wgan_gp import WGANCritic, WGGenerator, gradient_penalty
from src.pbmc3k_data import Pbmc3kExpressionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--noise-dim", type=int, default=32)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=Path, default=ROOT / "checkpoints" / "wgan_gp_pbmc3k.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds_tr = Pbmc3kExpressionDataset("train")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device(args.device)
    G = WGGenerator(n_features=ds_tr.n_features, noise_dim=args.noise_dim).to(device)
    D = WGANCritic(n_features=ds_tr.n_features).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        loss_g_epoch = 0.0
        loss_d_epoch = 0.0
        n_batches = 0
        for real in dl_tr:
            real = real.to(device)
            bs = real.size(0)

            for _ in range(args.n_critic):
                opt_d.zero_grad()
                noise = torch.randn(bs, args.noise_dim, device=device)
                fake = G(noise).detach()
                gp = gradient_penalty(D, real, fake)
                loss_d = D(fake).mean() - D(real).mean() + args.gp_lambda * gp
                loss_d.backward()
                opt_d.step()

            opt_g.zero_grad()
            noise = torch.randn(bs, args.noise_dim, device=device)
            fake = G(noise)
            loss_g = -D(fake).mean()
            loss_g.backward()
            opt_g.step()

            loss_g_epoch += loss_g.item()
            loss_d_epoch += loss_d.item()
            n_batches += 1

        loss_g_epoch /= max(n_batches, 1)
        loss_d_epoch /= max(n_batches, 1)
        print(f"epoch {epoch:03d}  loss_g={loss_g_epoch:.4f}  loss_d={loss_d_epoch:.4f}")

        torch.save(
            {
                "generator": G.state_dict(),
                "critic": D.state_dict(),
                "n_features": ds_tr.n_features,
                "noise_dim": args.noise_dim,
                "epoch": epoch,
            },
            args.save,
        )

    print(f"Tamam | son checkpoint: {args.save}")


if __name__ == "__main__":
    main()
