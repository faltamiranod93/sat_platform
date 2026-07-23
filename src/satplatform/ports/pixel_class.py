# src/satplatform/ports/pixel_class.py
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable, Sequence, Optional
from ..contracts.products import BandSet
from ..contracts.core import ClassLabel
from ..contracts.geo import GeoRaster

if TYPE_CHECKING:  # solo para anotaciones; evita importar pandas/numpy en runtime del puerto
    import numpy as np
    import pandas as pd


@runtime_checkable
class PixelClassifierPort(Protocol):
    """
    Clasificador por píxel.
    Reglas:
      - predict() devuelve raster de labels (uint8/uint16) alineado a BandSet.
      - predict_points() devuelve labels (class-ids, shape (N,)) para muestras 1D
        (un DataFrame Mcal con columnas de banda). Usado por la evaluación (TFC/CV).
      - classes() devuelve el catálogo de clases (id/nombre/color/macro).
    """
    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster: ...
    def predict_points(self, df: "pd.DataFrame") -> "np.ndarray": ...
    def classes(self) -> Sequence[ClassLabel]: ...
    def name(self) -> str: ...

__all__ = ["PixelClassifierPort"]
