"""Decorator de RasterReaderPort que corrige georreferencia al vuelo.

Envuelve un reader base y aplica `GeorefFixService.fix_if_needed` al perfil de
cada raster leído. Pensado para las escenas Sentinel Hub de Laguna Seca, que se
descargan mal etiquetadas (EPSG:4326 con origen lon/lat y píxel métrico) cuando
en realidad son UTM 19S (EPSG:32719).

Es una red de seguridad: si el perfil ya está correcto, `fix_if_needed` devuelve
el mismo objeto y el decorator no reconstruye nada (passthrough).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..contracts.geo import GeoProfile, GeoRaster
from ..ports.raster_read import RasterReaderPort, URI
from ..services.georef_fix_service import GeorefFixService


@dataclass(frozen=True)
class GeorefFixingRasterReader(RasterReaderPort):
    """Decora un RasterReaderPort corrigiendo la georef geográfica-mal-etiquetada."""

    base: RasterReaderPort
    fix_service: GeorefFixService
    target_epsg: int = 32719

    def read(self, uri: URI, band_index: int | None = None) -> GeoRaster:
        r = self.base.read(uri, band_index)
        fixed = self.fix_service.fix_if_needed(r.profile, self.target_epsg)
        if fixed is r.profile:
            return r
        return GeoRaster(data=r.data, profile=fixed)

    def profile(self, uri: URI) -> GeoProfile:
        return self.fix_service.fix_if_needed(self.base.profile(uri), self.target_epsg)

    def size(self, uri: URI) -> Tuple[int, int]:
        return self.base.size(uri)

    def exists(self, uri: URI) -> bool:
        return self.base.exists(uri)


__all__ = ["GeorefFixingRasterReader"]
