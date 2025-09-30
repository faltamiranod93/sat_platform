# src/satplatform/ports/roi.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Mapping, Sequence, Tuple
from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, Bounds

GeoJSON = Mapping[str, Any]
WKT = str

@runtime_checkable
class ROIClipperPort(Protocol):
    """
    Recorta/alinea rasters a una ROI (polígono o bbox).
    Implementación típica: gdalwarp/warp API.
    """
    def clip_raster(self, raster: GeoRaster, roi: GeoJSON | WKT, roi_crs: CRSRef) -> GeoRaster: ...
    def clip_profile(self, profile: GeoProfile, roi: GeoJSON | WKT, roi_crs: CRSRef) -> GeoProfile: ...
    def roi_bounds(self, roi: GeoJSON | WKT, roi_crs: CRSRef, out_crs: CRSRef) -> Bounds: ...

__all__ = ["ROIClipperPort", "GeoJSON", "WKT"]
