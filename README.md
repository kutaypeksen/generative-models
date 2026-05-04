# Generative models benchmark — scRNA-seq (PBMC3K)

Repo: [https://github.com/kutaypeksen/generative-models](https://github.com/kutaypeksen/generative-models)

Tek hücreli RNA-seq verisinde çeşitli **generatif modeller** (VAE, WGAN-GP, AAE), **ensemble** (ortalama + fusion) ve **PPCA–Gaussian** baseline; **MMD**, **Wasserstein (1B gen özeti)** ve **korelasyon uyumsuzluğu** metrikleri ile karşılaştırma.

## Gereksinimler

- Python 3.11+ önerilir  
- `pip install -r requirements.txt`

## Hızlı akış

```bash
python scripts/download_pbmc3k.py
python scripts/preprocess_pbmc3k_scanpy.py --skip-plots
python scripts/train_vae_pbmc3k.py --epochs 50
python scripts/train_wgan_gp_pbmc3k.py --epochs 50
python scripts/train_aae_pbmc3k.py --epochs 50
python scripts/sample_generative_pbmc3k.py --model vae --n 2000
python scripts/sample_generative_pbmc3k.py --model wgan_gp --n 2000
python scripts/sample_generative_pbmc3k.py --model aae --n 2000
python scripts/train_fusion_ensemble_pbmc3k.py --epochs 35
python scripts/sample_ensemble_pbmc3k.py --kind mean --n 2000
python scripts/sample_ensemble_pbmc3k.py --kind fusion --n 2000
python scripts/fit_ppca_gaussian_pbmc3k.py
python scripts/sample_ppca_gaussian_pbmc3k.py --n 2000
python scripts/compare_all_generative_metrics.py
```

**Not (Windows):** Proje yolu Türkçe karakter içeriyorsa `matrix.mtx` okuması için ön işlem betiği geçici ASCII dizinine kopyalama kullanır.

## Veri kaynağı

PBMC3K filtrelenmiş 10x matrisi: [10x Genomics örnek arşivi](https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz) (Seurat PBMC3K tutorial ile uyumlu kaynak).


