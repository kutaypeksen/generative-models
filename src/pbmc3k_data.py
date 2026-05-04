"""İşlenmiş PBMC3K için PyTorch Dataset ve yol sabitleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Proje kökü: src'nin üst dizini
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class Pbmc3kExpressionDataset(Dataset):
    """
    pbmc3k_X_scaled_hvg.npy üzerinden ifade matrisi.
    İsteğe bağlı: küme/hücre tipi etiketi (Leiden kodları).
    """

    def __init__(
        self,
        split: str = "train",
        *,
        processed_dir: Path | None = None,
        return_label: bool = False,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split 'train', 'val' veya 'all' olmalı")
        d = processed_dir or PROCESSED_DIR
        X_path = d / "pbmc3k_X_scaled_hvg.npy"
        if not X_path.is_file():
            raise FileNotFoundError(
                f"{X_path} yok. Önce scripts/preprocess_pbmc3k_scanpy.py çalıştırın."
            )
        X_all = np.load(X_path).astype(np.float32)
        if split == "all":
            idx = np.arange(X_all.shape[0])
        else:
            name = "pbmc3k_train_indices.npy" if split == "train" else "pbmc3k_val_indices.npy"
            idx = np.load(d / name)
        self._X = X_all[idx]
        self._labels: np.ndarray | None = None
        if return_label:
            lab = np.load(d / "pbmc3k_leiden_labels.npy")
            self._labels = lab[idx]
        self.return_label = return_label

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, i: int):
        x = torch.from_numpy(self._X[i])
        if self.return_label:
            assert self._labels is not None
            y = int(self._labels[i])
            return x, y
        return x

    @property
    def n_features(self) -> int:
        return self._X.shape[1]
