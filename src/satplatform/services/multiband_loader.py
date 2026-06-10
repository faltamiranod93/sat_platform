"""Carga un GeoTIFF multibanda Sentinel-2 como BandSet.

Las escenas de Laguna Seca son un único TIFF de 12 bandas (no archivos por
banda). El `RasterReaderPort` devuelve una banda por llamada (`band_index`,
1-based), así que aquí se orquestan N lecturas y se nombran las bandas por su
posición física, según el orden con que las escribe el evalscript del script de
descarga `s_sen2_down_v3.py`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..contracts.core import S2BandName
from ..contracts.geo import GeoRaster
from ..contracts.products import BandSet, ResolutionM
from ..ports.raster_read import RasterReaderPort, URI

# Orden físico de banda en los TIFF Sentinel Hub.
# Fuente: evalscript de s_sen2_down_v3.py (`bands: [...]`) == config_bandas_v3.json `Nband_sort`.
# El índice GDAL/rasterio 1-based de cada banda es su posición en esta tupla + 1.
SENTINEL_HUB_BAND_ORDER: Tuple[S2BandName, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)


def load_multiband_bandset(
    reader: RasterReaderPort,
    uri: URI,
    *,
    band_order: Sequence[S2BandName] = SENTINEL_HUB_BAND_ORDER,
    wanted: Optional[Sequence[S2BandName]] = None,
    resolution_m: ResolutionM = 10,
) -> BandSet:
    """Lee un GeoTIFF multibanda como BandSet, nombrando bandas por posición.

    Args:
        reader: lector de raster (idealmente el decorado con fix de georef).
        uri: ruta al GeoTIFF multibanda.
        band_order: nombre de cada banda en orden físico (posición i → band_index i+1).
        wanted: si se indica, solo lee/incluye estas bandas (ahorra I/O y memoria).
        resolution_m: resolución de las bandas (10 m para Sentinel-2 óptico).

    Returns:
        BandSet con las bandas solicitadas; todas comparten el profile del TIFF.
    """
    names = list(wanted) if wanted is not None else list(band_order)
    order = list(band_order)
    bands: Dict[S2BandName, GeoRaster] = {}
    for name in names:
        if name not in order:
            raise KeyError(f"Banda {name} no está en band_order {tuple(order)}")
        idx = order.index(name) + 1  # 1-based para GDAL/rasterio
        bands[name] = reader.read(uri, band_index=idx)
    return BandSet(resolution_m=resolution_m, bands=bands)


def bandset_to_arrays(bandset: BandSet) -> Dict[str, np.ndarray]:
    """Convierte un BandSet en dict {banda: ndarray 2D} para extract_at_utm_points."""
    return {name: r.data for name, r in bandset.bands.items()}


__all__ = ["SENTINEL_HUB_BAND_ORDER", "load_multiband_bandset", "bandset_to_arrays"]
