"""Gerçek (ölçeklenmiş HVG) ile yalnızca VAE sentetiği: MMD / WD / CD.
Üç model için: scripts/compare_all_generative_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import correlation_discrepancy, maximum_mean_discrepancy, wasserstein_1d_per_feature
from src.pbmc3k_data import PROCESSED_DIR


def main() -> None:
    real_path = PROCESSED_DIR / "pbmc3k_X_scaled_hvg.npy"
    syn_path = PROCESSED_DIR / "pbmc3k_vae_synthetic.npy"
    if not real_path.is_file():
        raise SystemExit(f"Eksik: {real_path} — önce preprocess_pbmc3k_scanpy.py")
    if not syn_path.is_file():
        raise SystemExit(f"Eksik: {syn_path} — önce train_ve_sample_vae veya sample_vae_pbmc3k.py")

    X = np.load(real_path).astype(np.float64)
    Y = np.load(syn_path).astype(np.float64)
    n = min(X.shape[0], Y.shape[0])
    rng = np.random.default_rng(0)
    xi = rng.choice(X.shape[0], size=n, replace=False)
    yi = rng.choice(Y.shape[0], size=n, replace=False)
    Xs = X[xi]
    Ys = Y[yi]

    print(f"Ayar: {n} hücre, {Xs.shape[1]} özellik (alt örnekleme, tek seed)")
    print(f"MMD^2 (RBF): {maximum_mean_discrepancy(Xs, Ys):.6f}")
    print(f"Wasserstein-1 (özellik ort., 1B): {wasserstein_1d_per_feature(Xs, Ys):.6f}")
    print(f"Correlation discrepancy (Frob): {correlation_discrepancy(Xs, Ys):.6f}")


if __name__ == "__main__":
    main()
