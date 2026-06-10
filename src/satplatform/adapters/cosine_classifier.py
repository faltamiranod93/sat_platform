"""Clasificador por similitud coseno con refinamiento euclidiano opcional.

two_stage=False → equivalente a ClassMap v4  (solo coseno)
two_stage=True  → equivalente a ClassMap v5.0 / v5.1  (coseno + Euclídea para clases 3-5)
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
class CosineClassifierAdapter:
    """Implementa PixelClassifierPort via similitud coseno."""

    _classes: tuple[ClassLabel, ...]
    _reference: np.ndarray          # (Ng, F) vectores de referencia L2-normalizados
    _reference_raw: np.ndarray      # (Ng, F) sin normalizar (para etapa 2)
    _band_filter: tuple[str, ...]
    _include_hsl: bool
    _two_stage: bool
    _stage2_class_ids: tuple[int, ...]
    _indices: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        mcal_df: pd.DataFrame,
        classes: Sequence[ClassLabel],
        band_filter: Sequence[str],
        include_hsl: bool = True,
        two_stage: bool = False,
        stage2_class_ids: Sequence[int] = (3, 4, 5),
        indices: Sequence[str] = (),
        hsl_scale: float = 10000.0,
    ) -> "CosineClassifierAdapter":
        """Calcula vectores de referencia por clase desde el DataFrame Mcal."""
        feat = FeatureService()
        X_all, names = feat.from_dataframe(mcal_df, band_filter, include_hsl, indices)
        ng = mcal_df["Ng"].to_numpy()
        ref_raw = np.zeros((len(classes), len(names)), dtype=np.float32)

        for i, cls_label in enumerate(classes):
            X = X_all[ng == cls_label.id]
            ref_raw[i] = X.mean(axis=0) if len(X) else np.zeros(len(names))

        norms = np.linalg.norm(ref_raw, axis=1, keepdims=True) + 1e-12
        ref_normalized = (ref_raw / norms).astype(np.float32)

        return cls(
            _classes=tuple(classes),
            _reference=ref_normalized,
            _reference_raw=ref_raw,
            _band_filter=tuple(band_filter),
            _include_hsl=include_hsl,
            _two_stage=two_stage,
            _stage2_class_ids=tuple(stage2_class_ids),
            _indices=tuple(indices),
        )

    def _build_features(self, bands: BandSet) -> tuple[np.ndarray, int, int]:
        """Devuelve (X: (N, F), H, W)."""
        feat = FeatureService()
        X, _ = feat.from_bandset(bands, self._band_filter, self._include_hsl, self._indices)
        first = next(iter(bands.bands.values()))  # type: ignore[arg-type]
        H, W = first.profile.height, first.profile.width
        return X, H, W

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        X, H, W = self._build_features(bands)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X_norm = (X / norms).astype(np.float32)

        # Stage 1: cosine similarity → argmax
        similarity = X_norm @ self._reference.T  # (N, Ng)
        labels_idx = np.argmax(similarity, axis=1)

        # Stage 2: refine stage2 classes with Euclidean distance
        if self._two_stage and len(self._stage2_class_ids) > 0:
            class_ids = np.array([c.id for c in self._classes])
            stage2_indices = [
                i for i, c in enumerate(self._classes)
                if c.id in self._stage2_class_ids
            ]
            if stage2_indices:
                s2_ref = self._reference_raw[stage2_indices]  # (k, F)
                s2_ids = [self._classes[i].id for i in stage2_indices]
                all_class_ids = np.array([c.id for c in self._classes])

                # Pixels assigned to any stage2 class
                assigned_ids = all_class_ids[labels_idx]
                mask = np.isin(assigned_ids, list(self._stage2_class_ids))

                if mask.any():
                    X_masked = X[mask]
                    d2 = np.stack([
                        np.sum((X_masked - s2_ref[j]) ** 2, axis=1)
                        for j in range(len(stage2_indices))
                    ], axis=1)  # (M, k)
                    best_s2 = np.argmin(d2, axis=1)
                    refined_ids = np.array(s2_ids)[best_s2]
                    # Map back to class index
                    id_to_idx = {c.id: i for i, c in enumerate(self._classes)}
                    labels_idx[mask] = np.array([id_to_idx[cid] for cid in refined_ids])

        class_ids_arr = np.array([c.id for c in self._classes], dtype=np.int16)
        label_arr = class_ids_arr[labels_idx].reshape(H, W)

        stacked = bands.stack(
            [b for b in self._band_filter if b in bands.bands]  # type: ignore[arg-type, list-item]
        )
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
        suffix = "_twostage" if self._two_stage else ""
        return f"cosine{suffix}"


__all__ = ["CosineClassifierAdapter"]
