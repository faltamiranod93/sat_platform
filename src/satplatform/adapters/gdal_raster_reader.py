# src/satplatform/adapters/gdal_raster_reader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import os
import math
import numpy as np

# Try rasterio first; fallback to GDAL; last resort tifffile
try:  # rasterio path
    import rasterio
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

try:  # GDAL path
    from osgeo import gdal, osr  # type: ignore
    try:
        from osgeo import gdal_array  # type: ignore
        _HAS_GDAL_ARRAY = True
    except Exception:  # pragma: no cover
        _HAS_GDAL_ARRAY = False
    _HAS_GDAL = True
except Exception:  # pragma: no cover
    _HAS_GDAL = False
    _HAS_GDAL_ARRAY = False

try:  # tifffile minimal path (no georeferencing)
    import tifffile as tiff  # type: ignore
    _HAS_TIFFILE = True
except Exception:  # pragma: no cover
    _HAS_TIFFILE = False

from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, GeoTransform, DTypeStr
from ..ports.raster_read import RasterReaderPort

_DTYPE_MAP = {
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("int16"): "int16",
    np.dtype("uint32"): "uint32",
    np.dtype("int32"): "int32",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
}


def _np_to_dtype_str(dt: np.dtype) -> DTypeStr:
    try:
        return _DTYPE_MAP[np.dtype(dt)]  # type: ignore[return-value]
    except KeyError as e:  # pragma: no cover
        raise ValueError(f"dtype {dt} no soportado") from e


def _affine_to_gt(a: "Affine") -> GeoTransform:
    return (a.c, a.a, a.b, a.f, a.d, a.e)


def _rasterio_crs_to_crsref(crs_obj) -> CRSRef:
    """Convierte rasterio CRS → CRSRef (intenta EPSG, si no WKT, si no vacío)."""
    if not crs_obj:
        return CRSRef()
    # EPSG si se puede
    try:
        epsg = crs_obj.to_epsg()
    except Exception:
        epsg = None
    if epsg is not None:
        return CRSRef.from_epsg(int(epsg))
    # WKT como fallback
    try:
        wkt = crs_obj.to_wkt()
    except Exception:
        wkt = None
    return CRSRef.from_wkt(wkt) if wkt else CRSRef()


def _gdal_datatype_to_np_dtype(dt_code: int) -> np.dtype:
    """Mapea GDALDataType a numpy.dtype sin leer la banda completa."""
    if _HAS_GDAL_ARRAY:
        try:
            np_code = gdal_array.GDALTypeCodeToNumericTypeCode(dt_code)
            if np_code is not None:
                return np.dtype(np_code)
        except Exception:
            pass
    # fallback conservador
    return np.dtype("float32")


@dataclass(frozen=True)
class GdalRasterReader(RasterReaderPort):
    """Lector de raster. Prefiere rasterio; si no, GDAL; último recurso, tifffile.

    Regla: `read()` devuelve un **GeoRaster count==1**. Para datasets multibanda
    debes pasar `band_index` (1-based si viene de GDAL/rasterio).
    """

    # --------------- rasterio ---------------
    def _read_with_rasterio(self, uri: str, band_index: int | None) -> GeoRaster:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            idx = 1 if band_index is None else int(band_index)
            arr = ds.read(idx)
            if arr.ndim != 2:
                raise ValueError("Se esperaba banda 2D (count==1)")
            crs_ref = _rasterio_crs_to_crsref(ds.crs)
            profile = GeoProfile(
                count=1,
                dtype=_np_to_dtype_str(arr.dtype),
                width=ds.width,
                height=ds.height,
                transform=_affine_to_gt(ds.transform),
                crs=crs_ref,
                nodata=float(ds.nodata) if ds.nodata is not None else None,
            )
            return GeoRaster(arr, profile)

    def _profile_with_rasterio(self, uri: str) -> GeoProfile:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            crs_ref = _rasterio_crs_to_crsref(ds.crs)
            dtype0 = np.dtype(ds.dtypes[0]) if ds.dtypes and ds.dtypes[0] else np.dtype("float32")
            return GeoProfile(
                count=ds.count,
                dtype=_np_to_dtype_str(dtype0),
                width=ds.width,
                height=ds.height,
                transform=_affine_to_gt(ds.transform),
                crs=crs_ref,
                nodata=float(ds.nodata) if ds.nodata is not None else None,
            )

    def _size_with_rasterio(self, uri: str) -> Tuple[int, int]:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            return ds.width, ds.height

    # --------------- GDAL ---------------
    def _read_with_gdal(self, uri: str, band_index: int | None) -> GeoRaster:
        assert _HAS_GDAL
        ds = gdal.Open(uri, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(uri)
        try:
            idx = 1 if band_index is None else int(band_index)
            band = ds.GetRasterBand(idx)
            arr = band.ReadAsArray()
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            srs_wkt = ds.GetProjection() or None
            nodata = band.GetNoDataValue()
            profile = GeoProfile(
                count=1,
                dtype=_np_to_dtype_str(arr.dtype),
                width=w,
                height=h,
                transform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
                crs=CRSRef.from_wkt(srs_wkt) if srs_wkt else CRSRef(),
                nodata=float(nodata) if nodata is not None and not math.isnan(nodata) else None,
            )
            return GeoRaster(data=arr, profile=profile)
        finally:
            ds = None  # cierre explícito

    def _profile_with_gdal(self, uri: str) -> GeoProfile:
        assert _HAS_GDAL
        ds = gdal.Open(uri, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(uri)
        try:
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            srs_wkt = ds.GetProjection() or None
            band = ds.GetRasterBand(1)
            np_dt = _gdal_datatype_to_np_dtype(band.DataType)
            nodata = band.GetNoDataValue()
            return GeoProfile(
                count=ds.RasterCount,
                dtype=_np_to_dtype_str(np_dt),
                width=w,
                height=h,
                transform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
                crs=CRSRef.from_wkt(srs_wkt) if srs_wkt else CRSRef(),
                nodata=float(nodata) if nodata is not None and not math.isnan(nodata) else None,
            )
        finally:
            ds = None

    # --------------- tifffile ---------------
    def _read_with_tifffile(self, uri: str) -> GeoRaster:
        assert _HAS_TIFFILE
        arr = tiff.imread(uri)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                raise ValueError("TIFF multibanda no soportado por tifffile fallback")
        profile = GeoProfile(
            count=1,
            dtype=_np_to_dtype_str(arr.dtype),
            width=arr.shape[1],
            height=arr.shape[0],
            transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),  # sin georreferencia
            crs=CRSRef(),
            nodata=None,
        )
        return GeoRaster(data=arr, profile=profile)

    # --------------- RasterReaderPort ---------------
    def read(self, uri: str, band_index: int | None = None) -> GeoRaster:
        if _HAS_RASTERIO:
            return self._read_with_rasterio(uri, band_index)
        if _HAS_GDAL:
            return self._read_with_gdal(uri, band_index)
        if _HAS_TIFFILE:
            return self._read_with_tifffile(uri)
        raise RuntimeError("No hay backend para leer rasters (instala rasterio o GDAL)")

    def profile(self, uri: str) -> GeoProfile:
        if _HAS_RASTERIO:
            return self._profile_with_rasterio(uri)
        if _HAS_GDAL:
            return self._profile_with_gdal(uri)
        # tifffile fallback
        r = self._read_with_tifffile(uri)
        return r.profile

    def size(self, uri: str) -> Tuple[int, int]:
        if _HAS_RASTERIO:
            return self._size_with_rasterio(uri)
        if _HAS_GDAL:
            p = self._profile_with_gdal(uri)
            return p.width, p.height
        r = self._read_with_tifffile(uri)
        return r.profile.width, r.profile.height

    def exists(self, uri: str) -> bool:
        return os.path.exists(uri)
