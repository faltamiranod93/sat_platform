## `src/satplatform/adapters/gdal_raster_writer.py`
from __future__ import annotations

import os
from typing import Mapping, Any, Optional

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.enums import Resampling
    from rasterio.errors import NotGeoreferencedWarning
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

try:
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:  # pragma: no cover
    _HAS_GDAL = False

from ..contracts.geo import GeoRaster, GeoProfile, CRSRef
from ..ports.raster_write import RasterWriterPort


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class GdalRasterWriter(RasterWriterPort):
    def write(self, uri: str, raster: GeoRaster, *, compress: Optional[str] = None, tiled: bool = True) -> str:
        _ensure_dir(uri)
        data = raster.data
        p = raster.profile
        compress = (compress or "DEFLATE").upper()

        if _HAS_RASTERIO:
            profile = {
                "driver": "GTiff",
                "height": p.height,
                "width": p.width,
                "count": 1 if data.ndim == 2 else data.shape[0],
                "dtype": data.dtype,
                "transform": Affine.from_gcp(()) if False else (p.transform[1], p.transform[2], p.transform[0], p.transform[4], p.transform[5], p.transform[3]),
                "compress": compress,
                "tiled": tiled,
                "nodata": p.nodata,
            }
            if p.crs.epsg is not None:
                profile["crs"] = f"EPSG:{p.crs.epsg}"
            elif p.crs.wkt:
                profile["crs"] = p.crs.wkt
            with rasterio.open(uri, "w", **profile) as dst:
                if data.ndim == 2:
                    dst.write(data, 1)
                else:
                    for i in range(data.shape[0]):
                        dst.write(data[i], i + 1)
            return uri

        if _HAS_GDAL:
            driver = gdal.GetDriverByName("GTiff")
            count = 1 if data.ndim == 2 else data.shape[0]
            dtype = gdal.GDT_Float32
            _NP2GDAL = {
                np.dtype("uint8"): gdal.GDT_Byte,
                np.dtype("uint16"): gdal.GDT_UInt16,
                np.dtype("int16"): gdal.GDT_Int16,
                np.dtype("uint32"): gdal.GDT_UInt32,
                np.dtype("int32"): gdal.GDT_Int32,
                np.dtype("float32"): gdal.GDT_Float32,
                np.dtype("float64"): gdal.GDT_Float64,
            }
            dtype = _NP2GDAL.get(data.dtype, gdal.GDT_Float32)
            ds = driver.Create(uri, p.width, p.height, count, dtype, options=[f"COMPRESS={compress}", "TILED=YES" if tiled else "TILED=NO"])
            ds.SetGeoTransform(p.transform)
            if p.crs.wkt:
                ds.SetProjection(p.crs.wkt)
            elif p.crs.epsg:
                srs = osr.SpatialReference(); srs.ImportFromEPSG(int(p.crs.epsg))
                ds.SetProjection(srs.ExportToWkt())
            if data.ndim == 2:
                ds.GetRasterBand(1).WriteArray(data)
                if p.nodata is not None:
                    ds.GetRasterBand(1).SetNoDataValue(float(p.nodata))
            else:
                for i in range(count):
                    band = ds.GetRasterBand(i+1)
                    band.WriteArray(data[i])
                    if p.nodata is not None:
                        band.SetNoDataValue(float(p.nodata))
            ds.FlushCache(); ds = None
            return uri

        raise RuntimeError("No hay backend para escribir rasters (instala rasterio o GDAL)")

    def write_profile(self, uri: str, profile: GeoProfile, *, metadata: Optional[Mapping[str, Any]] = None) -> str:
        # Escribe un raster vacío (todo nodata) con perfil dado
        import numpy as np
        data = np.full((profile.height, profile.width), profile.nodata if profile.nodata is not None else 0, dtype=np.float32)
        return self.write(uri, GeoRaster(data=data, profile=profile))

    def mkdirs(self, uri: str) -> None:
        _ensure_dir(uri)
