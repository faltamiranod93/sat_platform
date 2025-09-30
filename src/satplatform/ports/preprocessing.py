# src/satplatform/ports/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Tuple, Optional
from ..contracts.geo import GeoRaster

@dataclass(frozen=True)
class NormalizeSpec:
    """Especificación de normalización/histogramas."""
    method: str = "percent_clip"        # 'percent_clip'|'zscore'|'minmax'
    p_low: float = 2.0                  # percentil bajo para clip
    p_high: float = 98.0                # percentil alto para clip
    eps: float = 1e-6

@runtime_checkable
class PreprocessingPort(Protocol):
    """
    Transformaciones previas al clasificador.
    """
    def rgb_to_hsl(self, r: GeoRaster, g: GeoRaster, b: GeoRaster) -> Tuple[GeoRaster, GeoRaster, GeoRaster]: ...
    def normalize(self, x: GeoRaster, spec: NormalizeSpec = NormalizeSpec()) -> GeoRaster: ...

__all__ = ["PreprocessingPort", "NormalizeSpec"]
