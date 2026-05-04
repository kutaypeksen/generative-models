# PBMC3K filtrelenmiş 10x matrisi (Seurat öğreticisi ile uyumlu kaynak)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root "data\raw\pbmc3k"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$url = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
$tar = Join-Path $dataDir "pbmc3k_filtered_gene_bc_matrices.tar.gz"

Write-Host "İndiriliyor: $url"
$headers = @{ "User-Agent" = "Mozilla/5.0 (compatible; research-download/1.0)" }
Invoke-WebRequest -Uri $url -OutFile $tar -UseBasicParsing -Headers $headers

Write-Host "Açılıyor: $tar"
tar -xzf $tar -C $dataDir
Write-Host "Tamam. Çıktı: $dataDir"
