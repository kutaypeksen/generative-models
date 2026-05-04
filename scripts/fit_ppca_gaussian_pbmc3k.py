"""
PCA + latent uzayda empirik Gaussian - klasik (sinir agi degil) generatif baseline.
Egitim indeksleri ile fit (pbmc3k_train_indices.npy).
Kayit: checkpoints/ppca_gaussian_pbmc3k.pkl + checkpoints/ppca_gaussian_latent.npz
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--cov-jitter", type=float, default=1e-4, help="Kovaryans PSD icin diagonal stabilizasyon")
    p.add_argument("--pca-out", type=Path, default=ROOT / "checkpoints" / "ppca_gaussian_pbmc3k.pkl")
    p.add_argument("--latent-out", type=Path, default=ROOT / "checkpoints" / "ppca_gaussian_latent.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn gerekli", file=sys.stderr)
        sys.exit(1)

    X_path = PROCESSED_DIR / "pbmc3k_X_scaled_hvg.npy"
    idx_path = PROCESSED_DIR / "pbmc3k_train_indices.npy"
    if not X_path.is_file() or not idx_path.is_file():
        raise SystemExit("Once preprocess_pbmc3k_scanpy.py")

    X_all = np.load(X_path).astype(np.float64)
    idx = np.load(idx_path)
    X_tr = X_all[idx]

    k = min(args.n_components, X_tr.shape[0], X_tr.shape[1])
    pca = PCA(n_components=k, random_state=0)
    Z_tr = pca.fit_transform(X_tr)
    mean_z = Z_tr.mean(axis=0)
    cov_z = np.cov(Z_tr, rowvar=False)
    cov_z = cov_z + float(args.cov_jitter) * np.eye(Z_tr.shape[1])

    args.pca_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.pca_out, "wb") as f:
        pickle.dump({"pca": pca, "n_components": k}, f)
    np.savez(args.latent_out, mean=mean_z, cov=cov_z)
    print(f"PCA kaydedildi: {args.pca_out}")
    print(f"Latent Gaussian kaydedildi: {args.latent_out} | k={k}")


if __name__ == "__main__":
    main()
