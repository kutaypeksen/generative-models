"""
Hücre tipi / küme etiketi dağılımını koruyarak AnnData alt kümesi.

Buyuk veri setlerinde (ornegin HCA) hocanın istegi: toplam hucre sayisini dusururken
her etiketteki oranlari mumkun oldugunca koru.

Ornek:
  python scripts/stratified_subset_h5ad.py ^
    --in-h5ad data/processed/pbmc3k_processed.h5ad ^
    --label-key leiden --train-size 1500 ^
    --out-h5ad data/processed/pbmc3k_subset_stratified.h5ad ^
    --out-indices data/processed/pbmc3k_subset_indices.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified subset AnnData (etiket oranlari korunur)")
    p.add_argument("--in-h5ad", type=Path, required=True)
    p.add_argument("--label-key", type=str, required=True, help="obs sutunu ornegin leiden, cell_type")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train-size", type=int, help="Alt kumedeki hucre sayisi")
    g.add_argument("--frac", type=float, help="Her siniftan orneklem orani (0-1], sinir kontrollu")
    p.add_argument("--out-h5ad", type=Path, default=None)
    p.add_argument("--out-indices", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.in_h5ad.is_file():
        print(f"Dosya yok: {args.in_h5ad}", file=sys.stderr)
        sys.exit(1)

    try:
        import anndata as ad
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    adata = ad.read_h5ad(args.in_h5ad)
    if args.label_key not in adata.obs.columns:
        print(f"obs'de '{args.label_key}' yok. Kolonlar: {list(adata.obs.columns)}", file=sys.stderr)
        sys.exit(1)

    y = np.asarray(adata.obs[args.label_key].astype(str))
    n = adata.n_obs
    idx_all = np.arange(n)

    if args.train_size is not None:
        size = min(int(args.train_size), n)
        if size < n:
            sub_idx, _ = train_test_split(
                idx_all,
                train_size=size,
                stratify=y,
                random_state=args.seed,
            )
        else:
            sub_idx = idx_all
    else:
        assert args.frac is not None
        fr = float(args.frac)
        if fr <= 0 or fr > 1:
            print("--frac (0,1] olmali", file=sys.stderr)
            sys.exit(1)
        sub_idx, _ = train_test_split(
            idx_all,
            train_size=fr,
            stratify=y,
            random_state=args.seed,
        )

    sub_idx = np.sort(np.asarray(sub_idx, dtype=np.int64))
    args.out_indices.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_indices, sub_idx)
    print(f"Kaydedildi: {args.out_indices} | n={sub_idx.size}")

    if args.out_h5ad is not None:
        sub = adata[sub_idx].copy()
        args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
        sub.write_h5ad(args.out_h5ad)
        print(f"Kaydedildi: {args.out_h5ad}")


if __name__ == "__main__":
    main()
