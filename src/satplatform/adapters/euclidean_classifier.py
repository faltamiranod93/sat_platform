"""Clasificador por distancia euclidiana al vector de referencia por clase.

Equivalente a ClassMap v3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..contracts.core import ClassLabel
from ..contracts.geo import GeoProfile, GeoRaster
from ..contracts.products import BandSet
from ..services.feature_service import FeatureService


@dataclass
class EuclideanClassifierAdapter:
    """Implementa PixelClassifierPort via distancia euclidiana al centroide."""

    _classes: tuple[ClassLabel, ...]
    _reference: np.ndarray   # (Ng, F) vectores de referencia
    _band_filter: tuple[str, ...]
    _include_hsl: bool
    _indices: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        mcal_df: pd.DataFrame,
        classes: Sequence[ClassLabel],
        band_filter: Sequence[str],
        include_hsl: bool = True,
        indices: Sequence[str] = (),
    ) -> "EuclideanClassifierAdapter":
        """Calcula centroides por clase desde el DataFrame Mcal."""
        feat = FeatureService()
        X_all, names = feat.from_dataframe(mcal_df, band_filter, include_hsl, indices)
        ng = mcal_df["Ng"].to_numpy()
        ref = np.zeros((len(classes), len(names)), dtype=np.float32)

        for i, cls_label in enumerate(classes):
            X = X_all[ng == cls_label.id]
            ref[i] = X.mean(axis=0) if len(X) else np.zeros(len(names))

        return cls(
            _classes=tuple(classes),
            _reference=ref,
            _band_filter=tuple(band_filter),
            _include_hsl=include_hsl,
            _indices=tuple(indices),
        )

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Índice (en self._classes) del centroide euclidiano más cercano por fila."""
        # d2[n, g] = ||X[n] - ref[g]||²
        d2 = np.stack([
            np.sum((X - self._reference[g]) ** 2, axis=1)
            for g in range(len(self._classes))
        ], axis=1)  # (N, Ng)
        return np.argmin(d2, axis=1)

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        feat = FeatureService()
        X, _ = feat.from_bandset(bands, self._band_filter, self._include_hsl, self._indices)
        first = next(iter(bands.bands.values()))  # type: ignore[arg-type]
        H, W = first.profile.height, first.profile.width

        best_idx = self._score(X)
        class_ids = np.array([c.id for c in self._classes], dtype=np.int16)
        label_arr = class_ids[best_idx].reshape(H, W)

        p0 = first.profile
        return GeoRaster(
            data=label_arr,
            profile=GeoProfile(
                count=1, dtype="int16", width=W, height=H,
                transform=p0.transform, crs=p0.crs, nodata=-9999,
            ),
        )

    def predict_points(self, df: pd.DataFrame) -> np.ndarray:
        """Labels (class-ids, shape (N,)) para muestras 1D (DataFrame con columnas de banda)."""
        feat = FeatureService()
        X, _ = feat.from_dataframe(df, self._band_filter, self._include_hsl, self._indices)
        best_idx = self._score(X.astype(np.float32))
        class_ids = np.array([c.id for c in self._classes], dtype=np.int16)
        return class_ids[best_idx]

    def classes(self) -> Sequence[ClassLabel]:
        return self._classes

    def name(self) -> str:
        return "euclidean"


__all__ = ["EuclideanClassifierAdapter"]
