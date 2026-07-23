"""Métricas de evaluación de clasificación (puras, solo numpy).

Convención de la matriz de confusión: **CM[pred, true]** (filas = clase predicha,
columnas = clase verdadera). Todas las funciones asumen esa orientación.

Incluye:
- overall_accuracy, kappa (Cohen)
- producers_users (PA=recall por columna, UA=precision por fila), f1_per_class, f1_macro
- quantity_allocation_disagreement (Pontius & Millones 2011); invariante: Q + A = 1 - OA
- estimate_autocorr_range: correlograma de Moran's I por bins de distancia (diagnóstico
  para elegir block_size_m de la CV espacial).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], class_ids: Sequence[int]
) -> np.ndarray:
    """CM[pred, true] (int64). Ignora muestras cuya clase no está en class_ids."""
    class_ids = [int(c) for c in class_ids]
    idx = {c: i for i, c in enumerate(class_ids)}
    k = len(class_ids)
    cm = np.zeros((k, k), dtype=np.int64)
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    for t, p in zip(yt, yp):
        it = idx.get(int(t))
        ip = idx.get(int(p))
        if it is None or ip is None:
            continue
        cm[ip, it] += 1
    return cm


def overall_accuracy(cm: np.ndarray) -> float:
    n = cm.sum()
    return float(np.trace(cm) / n) if n else 0.0


def kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    p0 = np.trace(cm) / n
    row = cm.sum(axis=1)  # predichos
    col = cm.sum(axis=0)  # verdad
    pe = float(np.sum(row * col)) / (n * n)
    denom = 1.0 - pe
    return float((p0 - pe) / denom) if denom != 0 else 0.0


def producers_users(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(PA, UA) por clase.

    PA_i (producer's / recall)  = CM[i,i] / sum_col_i   (verdad de la clase i)
    UA_i (user's / precision)   = CM[i,i] / sum_row_i   (predicho como clase i)
    Devuelve NaN donde el denominador es 0.
    """
    diag = np.diag(cm).astype(float)
    row = cm.sum(axis=1).astype(float)  # predichos
    col = cm.sum(axis=0).astype(float)  # verdad
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = np.where(row > 0, diag / row, np.nan)
        pa = np.where(col > 0, diag / col, np.nan)
    return pa, ua


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    pa, ua = producers_users(cm)
    with np.errstate(invalid="ignore"):
        denom = pa + ua
        f1 = np.where(denom > 0, 2.0 * pa * ua / denom, np.nan)
    return f1


def f1_macro(cm: np.ndarray) -> float:
    f1 = f1_per_class(cm)
    vals = f1[~np.isnan(f1)]
    return float(vals.mean()) if vals.size else 0.0


def quantity_allocation_disagreement(cm: np.ndarray) -> Tuple[float, float]:
    """Desacuerdo de cantidad (Q) y de asignación (A) — Pontius & Millones (2011).

    En proporciones p = CM/N: por clase g, q_g = |sum_row_g - sum_col_g|,
    a_g = 2·min(sum_row_g - p_gg, sum_col_g - p_gg); Q = Σq_g/2, A = Σa_g/2.
    Invariante: Q + A = 1 - OA.
    """
    n = cm.sum()
    if n == 0:
        return 0.0, 0.0
    p = cm.astype(float) / n
    row = p.sum(axis=1)  # predichos
    col = p.sum(axis=0)  # verdad
    diag = np.diag(p)
    q = float(np.sum(np.abs(row - col)) / 2.0)
    a = float(np.sum(2.0 * np.minimum(row - diag, col - diag)) / 2.0)
    return q, a


def first_principal_component(X: np.ndarray) -> np.ndarray:
    """1ª componente principal (proyección centrada) de una matriz de features (N,F)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD es estable y no requiere la covarianza explícita
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = Xc @ vt[0]
    except np.linalg.LinAlgError:
        pc1 = Xc[:, 0]
    return pc1


def estimate_autocorr_range(
    coords: np.ndarray,
    values: Sequence[float],
    n_bins: int = 12,
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Correlograma de Moran's I por bins de distancia sobre `values` (1D).

    I(d) = (N/W_d) · Σ_ij w_ij(d)(x_i-x̄)(x_j-x̄) / Σ_i (x_i-x̄)²,
    con w_ij(d)=1 si la distancia del par cae en el bin d (excluyendo la diagonal).

    El `range_m` estimado es el centro del primer bin donde I cae a <= threshold
    (por defecto E[I] = -1/(N-1), la ausencia de autocorrelación). Se usa como
    valor sugerido para block_size_m de la CV espacial.
    """
    coords = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float).ravel()
    n = z.size
    if n < 3:
        return {"range_m": 0.0, "e_i": 0.0, "correlogram": []}
    z = z - z.mean()
    denom = float(np.sum(z * z))
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    dmax = float(d.max())
    if dmax <= 0 or denom <= 0:
        return {"range_m": 0.0, "e_i": -1.0 / (n - 1), "correlogram": []}
    edges = np.linspace(0.0, dmax, n_bins + 1)
    zz = np.outer(z, z)
    e_i = -1.0 / (n - 1)
    thr = e_i if threshold is None else float(threshold)
    correlogram: List[Tuple[float, Optional[float], float]] = []
    range_m = dmax
    found = False
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        w = ((d >= lo) & (d < hi)).astype(float)
        np.fill_diagonal(w, 0.0)
        wsum = float(w.sum())
        center = 0.5 * (lo + hi)
        if wsum == 0:
            correlogram.append((center, None, 0.0))
            continue
        I = float((n / wsum) * (np.sum(w * zz) / denom))
        correlogram.append((center, I, wsum))
        if (not found) and I <= thr:
            range_m = center
            found = True
    return {"range_m": float(range_m), "e_i": float(e_i), "correlogram": correlogram}


__all__ = [
    "confusion_matrix",
    "overall_accuracy",
    "kappa",
    "producers_users",
    "f1_per_class",
    "f1_macro",
    "quantity_allocation_disagreement",
    "first_principal_component",
    "estimate_autocorr_range",
]
