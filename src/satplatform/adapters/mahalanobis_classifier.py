"""Clasificador por distancia de Mahalanobis con regularización LedoitWolf.

include_hsl=False → equivalente a ClassMap v9  (features: bandas filtradas)
include_hsl=True  → equivalente a ClassMap v9p2 (features: bandas + H,S,L físico)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..contracts.core import ClassLabel, S2BandName
from ..contracts.geo import GeoProfile, GeoRaster
from ..contracts.products import BandSet
from ..services.feature_service import FeatureService

DEFAULT_BAND_FILTER: tuple[str, ...] = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)


@dataclass
class MahalanobisClassifierAdapter:
    """Implementa PixelClassifierPort via distancia de Mahalanobis.

    No usar el constructor directamente — usar MahalanobisClassifierAdapter.fit().
    """

    _classes: tuple[ClassLabel, ...]
    _models: dict[int, tuple[np.ndarray, np.ndarray]]  # class_id → (mean, precision)
    _band_filter: tuple[str, ...]
    _include_hsl: bool
    _indices: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        mcal_df: pd.DataFrame,
        classes: Sequence[ClassLabel],
        band_filter: Sequence[str] = DEFAULT_BAND_FILTER,
        include_hsl: bool = True,
        indices: Sequence[str] = (),
        diag_reg: float = 1e-6,
    ) -> "MahalanobisClassifierAdapter":
        """Ajusta modelos Mahalanobis por clase desde el DataFrame Mcal.

        mcal_df debe contener 'Ng' y las columnas de banda necesarias para las
        features (band_filter, + B04/B03/B02 si include_hsl, + bandas de indices).
        Las features derivadas se calculan vía FeatureService (no se leen H/S/L).
        """
        feat = FeatureService()
        X_all, names = feat.from_dataframe(mcal_df, band_filter, include_hsl, indices)
        n_f = len(names)
        ng = mcal_df["Ng"].to_numpy()
        models: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for cls_label in classes:
            X = X_all[ng == cls_label.id].astype(np.float32)

            if len(X) < 2:
                models[cls_label.id] = (
                    X.mean(axis=0) if len(X) else np.zeros(n_f, dtype=np.float32),
                    np.eye(n_f, dtype=np.float32),
                )
                continue

            lw = LedoitWolf().fit(X)
            mean = lw.location_.astype(np.float32)
            cov = lw.covariance_.astype(np.float32) + diag_reg * np.eye(n_f, dtype=np.float32)
            precision = np.linalg.pinv(cov).astype(np.float32)
            models[cls_label.id] = (mean, precision)

        return cls(
            _classes=tuple(classes),
            _models=models,
            _band_filter=tuple(band_filter),
            _include_hsl=include_hsl,
            _indices=tuple(indices),
        )

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Índice (en self._classes) de la clase de mínima distancia de Mahalanobis por fila."""
        n_classes = len(self._classes)
        d2 = np.full((X.shape[0], n_classes), np.inf, dtype=np.float32)
        for i, cls_label in enumerate(self._classes):
            mean, prec = self._models[cls_label.id]
            diff = X - mean
            d2[:, i] = np.sum((diff @ prec) * diff, axis=1)
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
        return "mahalanobis_hsl" if self._include_hsl else "mahalanobis"


__all__ = ["MahalanobisClassifierAdapter", "DEFAULT_BAND_FILTER"]
