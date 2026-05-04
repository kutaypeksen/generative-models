"""
İstatistiksel benzerlik metrikleri: gerçek vs sentetik (veya iki örnek kümesi).

Notlar:
- MMD: RBF çekirdeği ile iki örneklemin dağılımları arasındaki farkın bir proxy'si.
- WD: düşük boyutta scipy.stats.wasserstein_distance; yüksek boyutta genelde özet istatistikler veya dilimlenmiş Wasserstein kullanılır.
- CD (Correlation Discrepancy): gen ifade korelasyon matrisleri arasındaki Frob norm farkı gibi tanımlanabilir.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel


def maximum_mean_discrepancy(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """
    İki küme için unbiased MMD^2 tahmini (RBF çekirdeği).

    x, y: (n_samples, n_features) yoğun matrisler.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.shape[0], y.shape[0]
    if nx < 2 or ny < 2:
        raise ValueError("MMD için her kümede en az 2 örnek gerekir.")

    xy = np.vstack([x, y])
    if gamma is None:
        # median heuristic (çiftler arası uzaklıkların medyanı)
        dists = pdist(xy, metric="euclidean")
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * (med**2 + 1e-12))

    k = rbf_kernel(xy, gamma=gamma)
    k_xx = k[:nx, :nx]
    k_yy = k[nx:, nx:]
    k_xy = k[:nx, nx:]

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    term_xx = k_xx.sum() / (nx * (nx - 1))
    term_yy = k_yy.sum() / (ny * (ny - 1))
    term_xy = k_xy.mean()
    mmd2 = term_xx + term_yy - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def wasserstein_1d_per_feature(
    x: np.ndarray,
    y: np.ndarray,
    aggregate: str = "mean",
) -> float:
    """
    Her özellik için 1B Wasserstein-1 (Earth Mover's Distance benzeri),
    sonra ortalama veya medyan ile tek skaler.

    x, y: (n_samples, n_features)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x.shape[1]
    w = np.empty(d, dtype=np.float64)
    for j in range(d):
        w[j] = stats.wasserstein_distance(x[:, j], y[:, j])
    if aggregate == "mean":
        return float(np.mean(w))
    if aggregate == "median":
        return float(np.median(w))
    raise ValueError("aggregate 'mean' veya 'median' olmalı")


def correlation_discrepancy(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
) -> float:
    """
    Korelasyon matrisleri arasındaki Frob norm: ||Corr(X) - Corr(Y)||_F.

    Çok fazla gen olduğunda alt örnekleme veya en değişken genler önerilir.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if method == "pearson":
        cx = np.corrcoef(x.T)
        cy = np.corrcoef(y.T)
    else:
        raise ValueError("Şimdilik sadece pearson destekleniyor.")
    cx = np.nan_to_num(cx, nan=0.0)
    cy = np.nan_to_num(cy, nan=0.0)
    return float(np.linalg.norm(cx - cy, ord="fro"))
