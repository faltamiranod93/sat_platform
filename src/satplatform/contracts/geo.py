# src/satplatform/contracts/geo.py

from __future__ import annotations
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Tuple, Optional
import math

try:
    from osgeo import osr  # type: ignore
except Exception:  # pragma: no cover
    osr = None  # type: ignore

GeoTransform = Tuple[float, float, float, float, float, float]

DTypeStr = Literal["uint8","uint16","int16","uint32","int32","float32","float64"]

class Bounds(NamedTuple):
    minx: float; miny: float; maxx: float; maxy: float

@dataclass(frozen=True)
class CRSRef:
    wkt: Optional[str] = None
    epsg: Optional[int] = None

    @staticmethod
    def from_epsg(code: int) -> "CRSRef": return CRSRef(epsg=code)
    @staticmethod
    def from_wkt(wkt: str) -> "CRSRef": return CRSRef(wkt=wkt)

    def to_osr(self) -> "osr.SpatialReference":
        if osr is None:
            raise RuntimeError("osgeo.osr no disponible. Instala GDAL/OSR.")
        srs = osr.SpatialReference()
        if self.epsg is not None:
            srs.ImportFromEPSG(int(self.epsg)); return srs
        if self.wkt:
            srs.ImportFromWkt(self.wkt); return srs
        raise ValueError("CRSRef vacío: requiere 'epsg' o 'wkt'.")

    def to_wkt(self) -> str:
        if self.wkt: return self.wkt
        if self.epsg is not None:
            if osr is None: return f"EPSG:{self.epsg}"
            return self.to_osr().ExportToWkt()
        raise ValueError("CRSRef vacío: no hay WKT ni EPSG.")
    
    # geo.py (añade dentro de CRSRef)
    def _epsg_from_wkt_best_effort(self) -> Optional[int]:
        try:
            if self.wkt and "EPSG" in self.wkt.upper():
                # heurística barata
                import re
                m = re.search(r"EPSG[:\s]*([0-9]{4,5})", self.wkt, re.IGNORECASE)
                return int(m.group(1)) if m else None
        except Exception:
            pass
        return None

    def equals(self, other: "CRSRef") -> bool:
        if self is other: return True
        if self.epsg is not None and other.epsg is not None:
            return int(self.epsg) == int(other.epsg)
        if osr is not None:
            return bool(self.to_osr().IsSame(other.to_osr()))
        # fallback sin OSR
        a = self.epsg or self._epsg_from_wkt_best_effort()
        b = other.epsg or other._epsg_from_wkt_best_effort()
        if a is not None and b is not None:
            return int(a) == int(b)
        return self.to_wkt() == other.to_wkt()


@dataclass(frozen=True)
class GeoProfile:
    count: int
    dtype: DTypeStr
    width: int
    height: int
    transform: GeoTransform
    crs: CRSRef
    nodata: Optional[float] = None

    @property
    def bounds(self) -> Bounds:
        return geotransform_bounds(self.transform, self.width, self.height)

    def pixel_size(self) -> Tuple[float, float]:
        _, px, _, _, _, py = self.transform
        return (px, py)

    def with_transform(self, gt: GeoTransform) -> "GeoProfile":
        return GeoProfile(self.count, self.dtype, self.width, self.height, gt, self.crs, self.nodata)

    def with_crs(self, crs: CRSRef) -> "GeoProfile":
        return GeoProfile(self.count, self.dtype, self.width, self.height, self.transform, crs, self.nodata)

@dataclass(frozen=True)
class GeoRaster:
    data: "npt.NDArray[Any]"  # type: ignore[valid-type]
    profile: GeoProfile

    def __post_init__(self):
        # Bloquea mutaciones accidentales sobre los datos
        if np is not None and hasattr(self.data, "setflags"):
            try:
                self.data.setflags(write=False)
            except Exception:
                pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]

    def is_single_band(self) -> bool:
        return self.profile.count == 1 or getattr(self.data, "ndim", 2) == 2

def geotransform_bounds(gt: GeoTransform, width: int, height: int) -> Bounds:
    x0, px, rx, y0, ry, py = gt
    x_w = x0 + width * px + height * rx
    y_w = y0 + width * ry + height * py
    minx, maxx = (x0, x_w) if x0 <= x_w else (x_w, x0)
    miny, maxy = (y_w, y0) if y_w <= y0 else (y0, y_w)
    return Bounds(minx, miny, maxx, maxy)

def bounds_to_geotransform(bounds: Bounds, width: int, height: int) -> GeoTransform:
    minx, miny, maxx, maxy = bounds
    px = (maxx - minx) / float(width)
    py = (miny - maxy) / float(height)  # negativo
    return (minx, px, 0.0, maxy, 0.0, py)

def pixel_to_world(col: int, row: int, gt: GeoTransform) -> Tuple[float, float]:
    x0, px, rx, y0, ry, py = gt
    x = x0 + col * px + row * rx
    y = y0 + col * ry + row * py
    return x, y

def world_to_pixel(x: float, y: float, gt: GeoTransform) -> Tuple[float, float]:
    x0, px, rx, y0, ry, py = gt
    det = px * py - rx * ry
    if abs(det) < 1e-18: raise ValueError("GeoTransform no invertible (det≈0).")
    inv00 =  py / det; inv01 = -rx / det
    inv10 = -ry / det; inv11 =  px / det
    dx = x - x0; dy = y - y0
    col = inv00 * dx + inv01 * dy
    row = inv10 * dx + inv11 * dy
    return col, row

def build_osr_transform(src: CRSRef, dst: CRSRef) -> "osr.CoordinateTransformation":
    if osr is None: raise RuntimeError("osgeo.osr no disponible. Instala GDAL/OSR.")
    return osr.CoordinateTransformation(src.to_osr(), dst.to_osr())

def transform_points_xy(
    xs: "npt.NDArray[np.floating] | list[float]",  # type: ignore[name-defined]
    ys: "npt.NDArray[np.floating] | list[float]",  # type: ignore[name-defined]
    ct: "osr.CoordinateTransformation",
):
    if np is None:
        out_xs: list[float] = []; out_ys: list[float] = []
        for x, y in zip(xs, ys):
            x2, y2, _ = ct.TransformPoint(float(x), float(y))  # type: ignore[arg-type]
            out_xs.append(x2); out_ys.append(y2)
        return out_xs, out_ys  # type: ignore[return-value]
    xs_arr = np.asarray(xs, dtype="float64"); ys_arr = np.asarray(ys, dtype="float64")
    out_x = np.empty_like(xs_arr); out_y = np.empty_like(ys_arr)
    for i in range(xs_arr.size):
        x2, y2, _ = ct.TransformPoint(float(xs_arr[i]), float(ys_arr[i]))
        out_x[i] = x2; out_y[i] = y2
    return out_x, out_y

def _gt_close(a: GeoTransform, b: GeoTransform, tol=1e-6) -> bool:
    return all(math.isclose(x, y, rel_tol=0.0, abs_tol=tol) for x, y in zip(a, b))

def validate_profile_compat(a: GeoProfile, b: GeoProfile, *, require_same_crs: bool = True) -> None:
    if require_same_crs and not a.crs.equals(b.crs):
        raise ValueError("CRS no coincide.")
    if a.width != b.width or a.height != b.height:
        raise ValueError("Dimensiones no coinciden.")
    if not _gt_close(a.transform, b.transform):
        raise ValueError("GeoTransform no coincide (requiere resampling/alineación).")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype no coincide: {a.dtype} vs {b.dtype}")
    if (a.nodata is None) != (b.nodata is None) or (a.nodata != b.nodata):
        raise ValueError(f"nodata no coincide: {a.nodata} vs {b.nodata}")

def pretty_bounds(b: Bounds, ndigits: int = 3) -> str:
    return f"Bounds(minx={b.minx:.{ndigits}f}, miny={b.miny:.{ndigits}f}, maxx={b.maxx:.{ndigits}f}, maxy={b.maxy:.{ndigits}f})"

__all__ = [
    "GeoTransform","Bounds","CRSRef","GeoProfile","GeoRaster","geotransform_bounds",
    "bounds_to_geotransform","pixel_to_world","world_to_pixel","build_osr_transform",
    "transform_points_xy","validate_profile_compat","pretty_bounds","DTypeStr",
]
