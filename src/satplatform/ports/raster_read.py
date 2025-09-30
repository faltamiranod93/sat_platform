# src/satplatform/ports/raster_read.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, Tuple
from ..contracts.geo import GeoRaster, GeoProfile

URI = str

@runtime_checkable
class RasterReaderPort(Protocol):
    """
    Lector de raster genérico (GeoTIFF/COG, etc.).
    Reglas: devuelve SIEMPRE GeoRaster con count==1.
    """
    def read(self, uri: URI, band_index: int | None = None) -> GeoRaster: ...
    def profile(self, uri: URI) -> GeoProfile: ...
    def size(self, uri: URI) -> Tuple[int, int]: ...  # (width, height)
    def exists(self, uri: URI) -> bool: ...

__all__ = ["RasterReaderPort", "URI"]
