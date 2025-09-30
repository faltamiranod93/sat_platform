# src/satplatform/ports/raster_write.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Mapping, Any, Optional
from ..contracts.geo import GeoRaster, GeoProfile

URI = str

@runtime_checkable
class RasterWriterPort(Protocol):
    """
    Escritor de rasters (GeoTIFF/COG).
    """
    def write(self, uri: URI, raster: GeoRaster, *, compress: Optional[str] = None, tiled: bool = True) -> URI: ...
    def write_profile(self, uri: URI, profile: GeoProfile, *, metadata: Optional[Mapping[str, Any]] = None) -> URI: ...
    def mkdirs(self, uri: URI) -> None: ...

__all__ = ["RasterWriterPort", "URI"]
