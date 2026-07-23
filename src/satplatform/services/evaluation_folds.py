"""Generadores de folds para evaluación de clasificación (puros, sin I/O).

- `random_folds`      — K-fold aleatorio (baseline P0). Reproduce el sesgo que infla
                        la accuracy cuando las muestras están espacialmente agrupadas.
- `spatial_block_folds` — K-fold por **bloques espaciales** (celda de rejilla UTM): un
                        bloque entero cae en un único fold, evitando fuga espacial
                        train↔test (P1). Ver Jocea 2025 / Meyer & Pebesma 2022.
- `leave_one_date_out` — para TFC temporal (P2): entrena en N-1 fechas, testea en la
                        restante.
- `anchor_date_folds`  — TFC ancla (P3): entrena en UNA fecha, clasifica cada otra fecha.

Todas devuelven índices posicionales (0..N-1) sobre las filas del DataFrame de muestras.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

FoldIdx = Tuple[np.ndarray, np.ndarray]              # (train_idx, test_idx)
DatedFold = Tuple[np.ndarray, np.ndarray, str]       # (train_idx, test_idx, held_date)


def random_folds(n: int, n_folds: int = 5, seed: int = 42) -> List[FoldIdx]:
    """K-fold aleatorio sobre n muestras."""
    if n_folds < 2:
        raise ValueError("n_folds debe ser >= 2")
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    parts = np.array_split(idx, n_folds)
    folds: List[FoldIdx] = []
    for k in range(n_folds):
        test = parts[k]
        train = np.concatenate([parts[j] for j in range(n_folds) if j != k]) if n_folds > 1 else parts[k]
        folds.append((np.sort(train), np.sort(test)))
    return folds


def spatial_block_folds(
    coords: np.ndarray, block_size_m: float, n_folds: int = 5, seed: int = 42
) -> List[FoldIdx]:
    """K-fold por bloques de rejilla espacial.

    Cada muestra se asigna a la celda `(floor(E/b), floor(N/b))`; las celdas se
    reparten a folds (round-robin sobre celdas barajadas), de modo que **ninguna
    celda aparece en train y test a la vez**. Los folds con test vacío se omiten
    (posible si hay menos celdas que folds).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords debe ser (N, 2)")
    if block_size_m <= 0:
        raise ValueError("block_size_m debe ser > 0")
    if n_folds < 2:
        raise ValueError("n_folds debe ser >= 2")

    cells = np.floor(coords / float(block_size_m)).astype(np.int64)
    # id de celda contiguo 0..B-1
    _, block_ids = np.unique(cells, axis=0, return_inverse=True)
    block_ids = np.asarray(block_ids).ravel()
    unique_blocks = np.unique(block_ids)

    rng = np.random.default_rng(seed)
    shuffled = unique_blocks.copy()
    rng.shuffle(shuffled)
    fold_of_block = {int(b): (i % n_folds) for i, b in enumerate(shuffled)}
    fold_assign = np.array([fold_of_block[int(b)] for b in block_ids])

    folds: List[FoldIdx] = []
    for k in range(n_folds):
        test = np.where(fold_assign == k)[0]
        train = np.where(fold_assign != k)[0]
        if test.size == 0 or train.size == 0:
            continue
        folds.append((np.sort(train.astype(np.int64)), np.sort(test.astype(np.int64))))
    return folds


def leave_one_date_out(dates: Sequence[str]) -> List[DatedFold]:
    """Un fold por fecha: entrena en todas las demás, testea en esa fecha.

    Devuelve [] si hay menos de 2 fechas distintas (TFC temporal no aplica).
    """
    dates = np.asarray([str(d) for d in dates])
    uniq = sorted(set(dates.tolist()))
    if len(uniq) < 2:
        return []
    folds: List[DatedFold] = []
    for d in uniq:
        test = np.where(dates == d)[0]
        train = np.where(dates != d)[0]
        folds.append((np.sort(train.astype(np.int64)), np.sort(test.astype(np.int64)), d))
    return folds


def anchor_date_folds(dates: Sequence[str], anchor: str | None = None) -> List[DatedFold]:
    """TFC ancla: entrena en `anchor` (por defecto la fecha más antigua) y clasifica
    cada otra fecha (un fold por fecha destino, held_date = fecha destino).

    Devuelve [] si hay menos de 2 fechas distintas.
    """
    dates = np.asarray([str(d) for d in dates])
    uniq = sorted(set(dates.tolist()))
    if len(uniq) < 2:
        return []
    anchor = uniq[0] if anchor is None else str(anchor)
    if anchor not in uniq:
        return []
    train = np.where(dates == anchor)[0]
    folds: List[DatedFold] = []
    for d in uniq:
        if d == anchor:
            continue
        test = np.where(dates == d)[0]
        folds.append((np.sort(train.astype(np.int64)), np.sort(test.astype(np.int64)), d))
    return folds


__all__ = [
    "random_folds",
    "spatial_block_folds",
    "leave_one_date_out",
    "anchor_date_folds",
    "FoldIdx",
    "DatedFold",
]
