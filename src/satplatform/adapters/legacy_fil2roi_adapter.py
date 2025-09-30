## `src/satplatform/adapters/legacy_fil2roi_adapter.py`

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, Bounds
from ..ports.roi import ROIClipperPort, GeoJSON, WKT

@dataclass(frozen=True)
class LegacyFil2RoiAdapter(ROIClipperPort):
    """Adapter "legacy" auto-contenido (sin gdalwarp): recorta por bbox simple.

    Solo acepta ROI como Bounds (minx,miny,maxx,maxy) dentro del CRS del raster.
    """

    def clip_raster(self, raster: GeoRaster, roi: GeoJSON | WKT | Bounds, roi_crs: CRSRef) -> GeoRaster:
        if not isinstance(roi, tuple) or len(roi) != 4:
            raise NotImplementedError("LegacyFil2RoiAdapter solo admite Bounds")
        # Convertimos bbox a índices de pixel via world_to_pixel
        from ..contracts.geo import world_to_pixel
        minx, miny, maxx, maxy = roi
        i0, j0 = world_to_pixel(minx, maxy, raster.profile.transform)
        i1, j1 = world_to_pixel(maxx, miny, raster.profile.transform)
        i0, i1 = int(max(0, min(i0, i1))), int(min(raster.profile.width, max(i0, i1)))
        j0, j1 = int(max(0, min(j0, j1))), int(min(raster.profile.height, max(j0, j1)))
        sub = raster.data[j0:j1, i0:i1].copy()
        from ..contracts.geo import geotransform_from_bounds
        gt = geotransform_from_bounds((minx, miny, maxx, maxy), i1 - i0, j1 - j0)
        newp = GeoProfile(
            count=1,
            dtype=raster.profile.dtype,
            width=i1 - i0,
            height=j1 - j0,
            transform=gt,
            crs=raster.profile.crs,
            nodata=raster.profile.nodata,
        )
        return GeoRaster(data=sub, profile=newp)

    def clip_profile(self, profile: GeoProfile, roi: GeoJSON | WKT | Bounds, roi_crs: CRSRef) -> GeoProfile:
        if not isinstance(roi, tuple):
            raise NotImplementedError("LegacyFil2RoiAdapter solo admite Bounds")
        from ..contracts.geo import geotransform_from_bounds
        minx, miny, maxx, maxy = roi
        # Mantiene resolución original
        px, py = profile.pixel_size()
        w = int(round((maxx - minx) / px))
        h = int(round((miny - maxy) / py))
        gt = geotransform_from_bounds((minx, miny, maxx, maxy), w, h)
        return GeoProfile(profile.count, profile.dtype, w, h, gt, profile.crs, profile.nodata)

    def roi_bounds(self, roi: GeoJSON | WKT | Bounds, roi_crs: CRSRef, out_crs: CRSRef) -> Bounds:
        if isinstance(roi, tuple) and len(roi) == 4:
            return roi  # ya es Bounds
        raise NotImplementedError("LegacyFil2RoiAdapter solo admite Bounds")
