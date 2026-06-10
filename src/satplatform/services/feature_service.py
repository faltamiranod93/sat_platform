"""Construcción centralizada del vector de features para clasificación.

Único lugar donde se ensambla la matriz de features (bandas + H,S,L + índices
espectrales), usado idénticamente en entrenamiento (`from_dataframe`, muestras
1D) e inferencia (`from_bandset`, escena 2D). Las features derivadas (HSL,
índices) se **calculan de las bandas** con las mismas fórmulas en ambos casos,
de modo que el espacio de features es consistente train↔predict por construcción.

Orden canónico de columnas: band_filter → [H, S, L] → indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from ..contracts.products import BandSet
from .spectral_service import SpectralService

# Bandas RGB para HSL (r=B04, g=B03, b=B02) y escala de salida (grados / %).
_HSL_RGB = ("B04", "B03", "B02")
_HSL_SCALE = (360.0, 100.0, 100.0)


@dataclass
class FeatureService:
    spectral: SpectralService = field(default_factory=SpectralService)

    def feature_names(
        self,
        band_filter: Sequence[str],
        include_hsl: bool,
        indices: Sequence[str],
    ) -> Tuple[str, ...]:
        names = list(band_filter)
        if include_hsl:
            names += ["H", "S", "L"]
        names += list(indices)
        return tuple(names)

    def _hsl_columns(self, arrays: Dict[str, np.ndarray]) -> list[np.ndarray]:
        H, S, L = self.spectral.hsl_from_rgb(
            arrays[_HSL_RGB[0]], arrays[_HSL_RGB[1]], arrays[_HSL_RGB[2]]
        )
        return [
            H.ravel() * _HSL_SCALE[0],
            S.ravel() * _HSL_SCALE[1],
            L.ravel() * _HSL_SCALE[2],
        ]

    def _build(
        self,
        arrays: Dict[str, np.ndarray],
        band_filter: Sequence[str],
        include_hsl: bool,
        indices: Sequence[str],
    ) -> np.ndarray:
        """Ensambla (N, F) a partir de un dict {banda: array} (1D o 2D ravel)."""
        cols: list[np.ndarray] = [arrays[b].ravel().astype(np.float32) for b in band_filter]
        if include_hsl:
            cols += [c.astype(np.float32) for c in self._hsl_columns(arrays)]
        for idx in indices:
            cols.append(self.spectral.index_from_arrays(idx, arrays).ravel().astype(np.float32))
        return np.stack(cols, axis=1)

    def from_bandset(
        self,
        bandset: BandSet,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Matriz de features (N=H*W, F) para inferir sobre una escena."""
        arrays = {name: r.data for name, r in bandset.bands.items()}
        X = self._build(arrays, band_filter, include_hsl, indices)
        return X, self.feature_names(band_filter, include_hsl, indices)

    def from_dataframe(
        self,
        df: pd.DataFrame,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Matriz de features (N=filas, F) para entrenar desde muestras Mcal.

        HSL e índices se derivan de las columnas de banda del DataFrame (no se
        leen columnas pre-calculadas), con las mismas fórmulas que en inferencia.
        """
        arrays = {c: df[c].to_numpy() for c in df.columns}
        X = self._build(arrays, band_filter, include_hsl, indices)
        return X, self.feature_names(band_filter, include_hsl, indices)


__all__ = ["FeatureService"]
