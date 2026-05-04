"""Gerçek vs sentetik: VAE, WGAN-GP, AAE, ensemble (ortalama/fusion), PPCA+Gaussian; MMD, WD, CD."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import correlation_discrepancy, maximum_mean_discrepancy, wasserstein_1d_per_feature
from src.pbmc3k_data import PROCESSED_DIR

SYN_FILES = {
    "VAE": PROCESSED_DIR / "pbmc3k_vae_synthetic.npy",
    "WGAN-GP": PROCESSED_DIR / "pbmc3k_wgan_gp_synthetic.npy",
    "AAE": PROCESSED_DIR / "pbmc3k_aae_synthetic.npy",
    "ENSEMBLE-mean": PROCESSED_DIR / "pbmc3k_ensemble_mean_synthetic.npy",
    "ENSEMBLE-fusion": PROCESSED_DIR / "pbmc3k_ensemble_fusion_synthetic.npy",
    "PPCA-Gaussian": PROCESSED_DIR / "pbmc3k_ppca_gaussian_synthetic.npy",
}


def main() -> None:
    real_path = PROCESSED_DIR / "pbmc3k_X_scaled_hvg.npy"
    if not real_path.is_file():
        raise SystemExit(f"Eksik: {real_path} -- preprocess_pbmc3k_scanpy.py calistirin")

    X_full = np.load(real_path).astype(np.float64)
    rng = np.random.default_rng(0)

    print("model\tMMD2_RBF\tWD1_mean_feat\tCorrDisc_Frob")
    for name, syn_path in SYN_FILES.items():
        if not syn_path.is_file():
            print(f"{name}\t(dosya yok: {syn_path.name})")
            continue
        Y = np.load(syn_path).astype(np.float64)
        n = min(X_full.shape[0], Y.shape[0])
        xi = rng.choice(X_full.shape[0], size=n, replace=False)
        yi = rng.choice(Y.shape[0], size=n, replace=False)
        Xs = X_full[xi]
        Ys = Y[yi]
        with np.errstate(invalid="ignore", divide="ignore"):
            mmd = maximum_mean_discrepancy(Xs, Ys)
            wd = wasserstein_1d_per_feature(Xs, Ys)
            cd = correlation_discrepancy(Xs, Ys)
        print(f"{name}\t{mmd:.6f}\t{wd:.6f}\t{cd:.6f}")


if __name__ == "__main__":
    main()
