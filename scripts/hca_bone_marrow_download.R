# HCA insan kemik iliği 10x — OSCA.workflows ile uyumlu ön işleme için veri indirme.
# Bioconductor 3.13 kitabı: https://bioconductor.org/books/3.13/OSCA.workflows/hca-human-bone-marrow-10x-genomics.html
#
# Çalıştırmadan önce R paketleri:
# if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# BiocManager::install(c("SingleCellExperiment", "zellkonverter", "HDF5Array",
#                        "DropletUtils", "scRNAseq", "scuttle", "Matrix"))

suppressPackageStartupMessages({
  library(Matrix)
})

#' OSCA örneğinde olduğu gibi ExperimentHub üzerinden cache'lenmiş nesneyi kullanın.
#' Kitaptaki kod genelde şuna benzer (sürüme göre isim değişebilir):
#'   library(scRNAseq)
#'   sce <- BaronPancreasData()  # farklı örnekler için paket dokümantasyonuna bakın
#'
#' HCA kemik iliği için güncel yol: `celldex`, `scRNAseq`, veya doğrudan HCA portal export.
#' Aşağıdaki şablon, kullanıcının Bioconductor sürümüne göre güncellenmelidir.

message(
  "Bu dosya şablon niteliğindedir. OSCA.workflows kitabındaki 'hca-human-bone-marrow' ",
  "bölümündeki tam kodu kopyalayıp buraya yapıştırın; veri büyük olduğu için ",
  "SingleCellExperiment nesnesini RDS olarak kaydedin:\n",
  "  saveRDS(sce, 'data/raw/hca_bone_marrow/sce_subset.rds')\n",
  "Sonrasında Python tarafında zellkonverter veya h5ad export kullanılabilir."
)
