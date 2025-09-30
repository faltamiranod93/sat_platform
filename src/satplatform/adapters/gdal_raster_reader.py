## `src/satplatform/adapters/gdal_raster_reader.py`
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
    from osgeo import gdal, osr
    _HAS_GDAL = True
except Exception:  # pragma: no cover
    _HAS_GDAL = False

try:  # tifffile minimal path (no georeferencing)
    import tifffile as tiff
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


@dataclass(frozen=True)
class GdalRasterReader(RasterReaderPort):
    """Lector de raster. Prefiere rasterio; si no, GDAL; último recurso, tifffile.

    Regla: `read()` devuelve un **GeoRaster count==1**. Para datasets multibanda
    debes pasar `band_index` (1-based si viene de GDAL/rasterio).
    """

    def _read_with_rasterio(self, uri: str, band_index: int | None) -> GeoRaster:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            idx = 1 if band_index is None else int(band_index)
            arr = ds.read(idx)
            if arr.ndim != 2:
                raise ValueError("Se esperaba banda 2D (count==1)")
            profile = GeoProfile(
                count=1,
                dtype=_np_to_dtype_str(arr.dtype),
                width=ds.width,
                height=ds.height,
                transform=_affine_to_gt(ds.transform),
                crs=CRSRef.from_epsg(int(ds.crs.to_epsg())) if ds.crs else CRSRef(),
                nodata=float(ds.nodata) if ds.nodata is not None else None,
            )
            return GeoRaster(data=arr, profile=profile)

    def _profile_with_rasterio(self, uri: str) -> GeoProfile:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            return GeoProfile(
                count=ds.count,
                dtype=_np_to_dtype_str(np.dtype(ds.dtypes[0])),
                width=ds.width,
                height=ds.height,
                transform=_affine_to_gt(ds.transform),
                crs=CRSRef.from_epsg(int(ds.crs.to_epsg())) if ds.crs else CRSRef(),
                nodata=float(ds.nodata) if ds.nodata is not None else None,
            )

    def _size_with_rasterio(self, uri: str) -> Tuple[int, int]:
        assert _HAS_RASTERIO
        with rasterio.open(uri) as ds:
            return ds.width, ds.height

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
            return GeoProfile(
                count=ds.RasterCount,
                dtype=_np_to_dtype_str(np.dtype(band.DataType)),
                width=w,
                height=h,
                transform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
                crs=CRSRef.from_wkt(srs_wkt) if srs_wkt else CRSRef(),
                nodata=float(band.GetNoDataValue()) if band.GetNoDataValue() is not None else None,
            )
        finally:
            ds = None

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

    # --- RasterReaderPort ---
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

## `src/satplatform/adapters/legacy_histnorm_adapter.py`

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..contracts.geo import GeoRaster
from ..ports.preprocessing import PreprocessingPort, NormalizeSpec

@dataclass(frozen=True)
class LegacyHistNormAdapter(PreprocessingPort):
    def rgb_to_hsl(self, r: GeoRaster, g: GeoRaster, b: GeoRaster) -> Tuple[GeoRaster, GeoRaster, GeoRaster]:
        # Implementación vectorizada básica
        R = r.data.astype(np.float32); G = g.data.astype(np.float32); B = b.data.astype(np.float32)
        R /= (R.max() + 1e-6); G /= (G.max() + 1e-6); B /= (B.max() + 1e-6)
        cmax = np.maximum(np.maximum(R, G), B)
        cmin = np.minimum(np.minimum(R, G), B)
        delta = cmax - cmin
        L = (cmax + cmin) / 2.0
        S = np.zeros_like(L)
        mask = delta > 1e-6
        S[mask] = delta[mask] / (1 - np.abs(2 * L[mask] - 1) + 1e-6)
        H = np.zeros_like(L)
        idx = (cmax == R) & mask
        H[idx] = ((G[idx] - B[idx]) / (delta[idx] + 1e-6)) % 6
        idx = (cmax == G) & mask
        H[idx] = ((B[idx] - R[idx]) / (delta[idx] + 1e-6)) + 2
        idx = (cmax == B) & mask
        H[idx] = ((R[idx] - G[idx]) / (delta[idx] + 1e-6)) + 4
        H = (H / 6.0)
        # Empaqueta en GeoRaster con el perfil de entrada R
        from ..contracts.geo import GeoProfile, GeoRaster
        p = r.profile
        return (
            GeoRaster(H.astype(np.float32), p),
            GeoRaster(S.astype(np.float32), p),
            GeoRaster(L.astype(np.float32), p),
        )

    def normalize(self, x: GeoRaster, spec: NormalizeSpec = NormalizeSpec()) -> GeoRaster:
        arr = x.data.astype(np.float32)
        if spec.method == "percent_clip":
            lo = np.nanpercentile(arr, spec.p_low)
            hi = np.nanpercentile(arr, spec.p_high)
            y = np.clip((arr - lo) / (hi - lo + spec.eps), 0, 1)
        elif spec.method == "zscore":
            mu = float(np.nanmean(arr)); sd = float(np.nanstd(arr) + spec.eps)
            y = (arr - mu) / sd
        elif spec.method == "minmax":
            mn = float(np.nanmin(arr)); mx = float(np.nanmax(arr))
            y = (arr - mn) / (mx - mn + spec.eps)
        else:
            raise ValueError(f"NormalizeSpec.method desconocido: {spec.method}")
        from ..contracts.geo import GeoProfile, GeoRaster
        p = x.profile
        p2 = GeoProfile(p.count, "float32", p.width, p.height, p.transform, p.crs, p.nodata)
        return GeoRaster(y.astype(np.float32), p2)
