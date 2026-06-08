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
from ..services.spectral_service import SpectralService

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

    @classmethod
    def fit(
        cls,
        mcal_df: pd.DataFrame,
        classes: Sequence[ClassLabel],
        band_filter: Sequence[str] = DEFAULT_BAND_FILTER,
        include_hsl: bool = True,
        diag_reg: float = 1e-6,
    ) -> "MahalanobisClassifierAdapter":
        """Ajusta modelos Mahalanobis por clase a partir del DataFrame Mcal.

        mcal_df debe contener columna 'Ng' y las columnas de band_filter.
        Si include_hsl=True, también necesita columnas 'H', 'S', 'L'.
        """
        feature_cols = list(band_filter) + (["H", "S", "L"] if include_hsl else [])
        models: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for cls_label in classes:
            mask = mcal_df["Ng"] == cls_label.id
            X = mcal_df.loc[mask, feature_cols].values.astype(np.float32)
            n_f = len(feature_cols)

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
        )

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        available = [b for b in self._band_filter if b in bands.bands]  # type: ignore[operator]
        stacked = bands.stack(available)  # type: ignore[arg-type]
        n_bands, H, W = stacked.data.shape
        X = stacked.data.reshape(n_bands, H * W).T.astype(np.float32)

        if self._include_hsl:
            svc = SpectralService()
            hsl = svc.rgb_to_hsl(bands, order=("B04", "B03", "B02"))
            hsl_cols = np.stack([
                hsl.data[0].ravel() * 360.0,
                hsl.data[1].ravel() * 100.0,
                hsl.data[2].ravel() * 100.0,
            ], axis=1).astype(np.float32)
            X = np.concatenate([X, hsl_cols], axis=1)

        n_classes = len(self._classes)
        d2 = np.full((H * W, n_classes), np.inf, dtype=np.float32)
        for i, cls_label in enumerate(self._classes):
            mean, prec = self._models[cls_label.id]
            diff = X - mean
            d2[:, i] = np.sum((diff @ prec) * diff, axis=1)

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
        return "mahalanobis_hsl" if self._include_hsl else "mahalanobis"


__all__ = ["MahalanobisClassifierAdapter", "DEFAULT_BAND_FILTER"]
