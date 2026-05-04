"""PCA + latent Gaussian orneklemesi -> pbmc3k_ppca_gaussian_synthetic.npy"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pbmc3k_data import PROCESSED_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pca-in", type=Path, default=ROOT / "checkpoints" / "ppca_gaussian_pbmc3k.pkl")
    p.add_argument("--latent-in", type=Path, default=ROOT / "checkpoints" / "ppca_gaussian_latent.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pca_in.is_file() or not args.latent_in.is_file():
        raise SystemExit("Once fit_ppca_gaussian_pbmc3k.py")

    with open(args.pca_in, "rb") as f:
        blob = pickle.load(f)
    pca = blob["pca"]

    L = np.load(args.latent_in)
    mean_z = np.asarray(L["mean"], dtype=np.float64)
    cov_z = np.asarray(L["cov"], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    Z = rng.multivariate_normal(mean_z, cov_z, size=args.n)
    X = pca.inverse_transform(Z).astype(np.float32)

    out = PROCESSED_DIR / "pbmc3k_ppca_gaussian_synthetic.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, X)
    print(f"Kaydedildi: {out} shape={X.shape}")


if __name__ == "__main__":
    main()
