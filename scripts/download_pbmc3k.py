"""PBMC3K 10x filtrelenmiş matrisini data/raw/pbmc3k altına indirir ve açar."""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
# Bazı CDN'ler varsayılan Python UA ile 403 döndürür.
UA = "Mozilla/5.0 (compatible; research-download/1.0)"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "raw" / "pbmc3k"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    if not tar_path.exists():
        print(f"İndiriliyor: {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            tar_path.write_bytes(resp.read())
    print(f"Açılıyor: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        extract_kw = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        tf.extractall(out_dir, **extract_kw)
    print(f"Tamam: {out_dir}")


if __name__ == "__main__":
    main()
