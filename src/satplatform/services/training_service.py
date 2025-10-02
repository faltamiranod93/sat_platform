# src/satplatform/services/training_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from ..contracts.geo import GeoRaster
from ..contracts.core import ClassId

@dataclass
class TrainingDataset:
    X: np.ndarray  # (N, F) float32
    y: np.ndarray  # (N,) int32 (ClassId)
    class_weights: Dict[int, float]
    feature_names: Tuple[str, ...] = ()

@dataclass
class TrainingService:
    """
    Servicio de preparación de dataset para entrenamiento (puro dominio):
    - Flatten de rasters de features y classmap
    - Filtrado de NaNs/nodata
    - Split reproducible
    - Pesos de clase
    """

    def _flatten_features(self, features: GeoRaster) -> np.ndarray:
        # features.data: (F, H, W) -> (N, F)
        F, H, W = features.data.shape
        X = features.data.reshape(F, H * W).T.astype("float32", copy=False)
        return X

    def _flatten_labels(self, classmap: GeoRaster) -> np.ndarray:
        # classmap.data: (1, H, W) o (H, W)
        arr = classmap.data
        if arr.ndim == 3:
            arr = arr[0]
        y = arr.reshape(-1).astype("int32", copy=False)
        return y

    def build_dataset(
        self,
        features: GeoRaster,
        classmap: GeoRaster,
        feature_names: Optional[Tuple[str, ...]] = None,
        nodata_label: Optional[int] = 0,
    ) -> TrainingDataset:
        X = self._flatten_features(features)
        y = self._flatten_labels(classmap)

        mask = np.isfinite(X).all(axis=1)
        if nodata_label is not None:
            mask &= (y != int(nodata_label))
        X = X[mask]
        y = y[mask]

        # Pesos inversos a la frecuencia
        unique, counts = np.unique(y, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        total = float(y.size) if y.size else 1.0
        class_weights = {int(k): float(total / (len(freq) * v)) for k, v in freq.items()}

        return TrainingDataset(
            X=X,
            y=y,
            class_weights=class_weights,
            feature_names=feature_names or tuple(f"f{i}" for i in range(X.shape[1])),
        )

    def split(
        self, ds: TrainingDataset, train_ratio: float = 0.8, seed: int = 42
    ) -> tuple[TrainingDataset, TrainingDataset]:
        n = ds.X.shape[0]
        rng = np.random.default_rng(seed)
        idx = np.arange(n, dtype="int64")
        rng.shuffle(idx)
        ntr = int(round(train_ratio * n))
        id_tr = idx[:ntr]
        id_te = idx[ntr:]

        tr = TrainingDataset(
            X=ds.X[id_tr], y=ds.y[id_tr],
            class_weights=ds.class_weights,
            feature_names=ds.feature_names,
        )
        te = TrainingDataset(
            X=ds.X[id_te], y=ds.y[id_te],
            class_weights=ds.class_weights,
            feature_names=ds.feature_names,
        )
        return tr, te
