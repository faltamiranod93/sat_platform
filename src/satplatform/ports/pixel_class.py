# src/satplatform/ports/pixel_class.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Sequence, Optional
from ..contracts.products import BandSet
from ..contracts.core import ClassLabel
from ..contracts.geo import GeoRaster

@runtime_checkable
class PixelClassifierPort(Protocol):
    """
    Clasificador por píxel.
    Reglas:
      - predict() devuelve raster de labels (uint8/uint16) alineado a BandSet.
      - classes() devuelve el catálogo de clases (id/nombre/color/macro).
    """
    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster: ...
    def classes(self) -> Sequence[ClassLabel]: ...
    def name(self) -> str: ...

__all__ = ["PixelClassifierPort"]
