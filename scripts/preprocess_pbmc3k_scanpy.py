"""
PBMC3K: Scanpy ile QC, normalizasyon, HVG, PCA, UMAP, Leiden kümeleme.
Çıktı: data/processed/pbmc3k_processed.h5ad ve isteğe bağlı model girdisi dizileri.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_MTX = ROOT / "data" / "raw" / "pbmc3k" / "filtered_gene_bc_matrices" / "hg19"
PROCESSED_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PBMC3K Scanpy ön işleme")
    p.add_argument("--skip-plots", action="store_true", help="Figürleri kaydetme")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "figures" / "pbmc3k")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not RAW_MTX.is_dir():
        print(f"Ham veri bulunamadı: {RAW_MTX}", file=sys.stderr)
        print("Önce scripts/download_pbmc3k.py çalıştırın.", file=sys.stderr)
        sys.exit(1)

    import scanpy as sc

    sc.settings.verbosity = 3
    sc.set_figure_params(dpi=120, facecolor="white")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_h5ad = PROCESSED_DIR / "pbmc3k_processed.h5ad"

    # Windows + Türkçe/Unicode yol: scipy mmread başarısız olabiliyor; ASCII geçici dizin kullan.
    with tempfile.TemporaryDirectory(prefix="pbmc3k_mtx_") as tmp:
        tmp_path = Path(tmp)
        for name in ("matrix.mtx", "genes.tsv", "barcodes.tsv"):
            shutil.copy2(RAW_MTX / name, tmp_path / name)
        adata = sc.read_10x_mtx(tmp_path, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Mitokondrial oran (insan sembolleri MT-)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)

    adata.raw = adata
    adata = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, random_state=0)
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(
        adata,
        resolution=0.5,
        random_state=0,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    print(adata)
    adata.write_h5ad(out_h5ad)
    print(f"Kaydedildi: {out_h5ad}")

    # Model girdisi: ölçeklenmiş HVG matrisi (hücre x gen)
    X = adata.X
    try:
        X_dense = X.toarray().astype(np.float32)
    except AttributeError:
        X_dense = np.asarray(X, dtype=np.float32)

    np.save(PROCESSED_DIR / "pbmc3k_X_scaled_hvg.npy", X_dense)
    n_cells = X_dense.shape[0]
    rng = np.random.default_rng(0)
    idx = np.arange(n_cells)
    rng.shuffle(idx)
    n_train = int(0.9 * n_cells)
    train_idx = np.sort(idx[:n_train])
    val_idx = np.sort(idx[n_train:])
    np.save(PROCESSED_DIR / "pbmc3k_train_indices.npy", train_idx)
    np.save(PROCESSED_DIR / "pbmc3k_val_indices.npy", val_idx)

    labels_path = PROCESSED_DIR / "pbmc3k_leiden_labels.npy"
    leiden_codes = adata.obs["leiden"].astype("category").cat.codes.to_numpy(np.int64)
    np.save(labels_path, leiden_codes)
    meta_cols = ["leiden", "n_genes_by_counts", "total_counts", "pct_counts_mt"]
    adata.obs[meta_cols].to_csv(PROCESSED_DIR / "pbmc3k_obs_meta.csv")

    if not args.skip_plots:
        fig_dir = args.figures_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = str(fig_dir)
        sc.pl.umap(adata, color=["leiden"], show=False, save="_leiden.pdf")
        sc.pl.umap(adata, color=["total_counts"], show=False, save="_total_counts.pdf")
        print(f"Şekiller: {fig_dir}")

    print(
        "Tensor hazırlığı: pbmc3k_X_scaled_hvg.npy, pbmc3k_train_indices.npy, "
        "pbmc3k_val_indices.npy, pbmc3k_leiden_labels.npy"
    )


if __name__ == "__main__":
    main()
