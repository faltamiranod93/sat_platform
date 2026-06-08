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


@dataclass
class EuclideanClassifierAdapter:
    """Implementa PixelClassifierPort via distancia euclidiana al centroide."""

    _classes: tuple[ClassLabel, ...]
    _reference: np.ndarray   # (Ng, F) vectores de referencia
    _band_filter: tuple[str, ...]
    _include_hsl: bool

    @classmethod
    def fit(
        cls,
        mcal_df: pd.DataFrame,
        classes: Sequence[ClassLabel],
        band_filter: Sequence[str],
        include_hsl: bool = True,
    ) -> "EuclideanClassifierAdapter":
        """Calcula centroides por clase desde el DataFrame Mcal."""
        feature_cols = list(band_filter) + (["H", "S", "L"] if include_hsl else [])
        ref = np.zeros((len(classes), len(feature_cols)), dtype=np.float32)

        for i, cls_label in enumerate(classes):
            mask = mcal_df["Ng"] == cls_label.id
            X = mcal_df.loc[mask, feature_cols].values.astype(np.float32)
            ref[i] = X.mean(axis=0) if len(X) else np.zeros(len(feature_cols))

        return cls(
            _classes=tuple(classes),
            _reference=ref,
            _band_filter=tuple(band_filter),
            _include_hsl=include_hsl,
        )

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        available = [b for b in self._band_filter if b in bands.bands]  # type: ignore[operator]
        stacked = bands.stack(available)  # type: ignore[arg-type]
        n_bands, H, W = stacked.data.shape
        X = stacked.data.reshape(n_bands, H * W).T.astype(np.float32)

        # d2[n, g] = ||X[n] - ref[g]||²
        d2 = np.stack([
            np.sum((X - self._reference[g]) ** 2, axis=1)
            for g in range(len(self._classes))
        ], axis=1)  # (N, Ng)

        best_idx = np.argmin(d2, axis=1)
        class_ids = np.array([c.id for c in self._classes], dtype=np.int16)
        label_arr = class_ids[best_idx].reshape(H, W)

        p0 = stacked.profile
        return GeoRaster(
            data=label_arr,
            profile=GeoProfile(
                count=1, dtype="int16", width=W, height=H,
                transform=p0.transform, crs=p0.crs, nodata=-9999,
            ),
        )

    def classes(self) -> Sequence[ClassLabel]:
        return self._classes

    def name(self) -> str:
        return "euclidean"


__all__ = ["EuclideanClassifierAdapter"]
