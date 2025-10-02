# src/satplatform/contracts/geo.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Tuple, Optional

import numpy as np
import numpy.typing as npt

GeoTransform = Tuple[float, float, float, float, float, float]
DTypeStr = Literal["uint8","uint16","int16","uint32","int32","float32","float64"]

class Bounds(NamedTuple):
    minx: float; miny: float; maxx: float; maxy: float

# ---------- CRS (puro dominio, sin GDAL) ----------
@dataclass(frozen=True)
class CRSRef:
    wkt: Optional[str] = None
    epsg: Optional[int] = None

    @staticmethod
    def from_epsg(code: int) -> "CRSRef":
        return CRSRef(epsg=int(code))

    @staticmethod
    def from_wkt(wkt: str) -> "CRSRef":
        return CRSRef(wkt=wkt)

    def to_wkt(self) -> str:
        """
        Devuelve una representación de texto del CRS.
        - Si hay WKT, retorna el WKT tal cual.
        - Si no hay WKT pero sí EPSG, retorna 'EPSG:<code>' como representación textual.
        - Si no hay nada, error.
        """
        if self.wkt:
            return self.wkt
        if self.epsg is not None:
            return f"EPSG:{int(self.epsg)}"
        raise ValueError("CRSRef vacío: no hay WKT ni EPSG.")

    @staticmethod
    def _normalize_wkt(wkt: str) -> str:
        """
        Normalización determinista para comparación:
        - strip
        - upper
        - colapsar espacios en blanco internos
        - eliminar espacios alrededor de comas y corchetes
        No intenta parsear ni reordenar nodos.
        """
        s = wkt.strip().upper()
        # colapsar espacios múltiples
        parts = s.split()
        s = " ".join(parts)
        # compactar espacios antes/después de separadores comunes
        s = s.replace(" ,", ",").replace(", ", ",")
        s = s.replace("[ ", "[").replace(" ]", "]")
        return s

    def equals(self, other: "CRSRef") -> bool:
        """
        Comparación determinista sin GDAL:
        1) Si ambos tienen EPSG -> compara enteros.
        2) En caso contrario, si ambos tienen WKT -> compara WKT normalizado.
        3) Cualquier mezcla (uno EPSG y otro WKT) -> devuelve False.
        """
        if self is other:
            return True
        if self.epsg is not None and other.epsg is not None:
            return int(self.epsg) == int(other.epsg)
        if self.wkt and other.wkt:
            return self._normalize_wkt(self.wkt) == self._normalize_wkt(other.wkt)
        return False

# ---------- Perfil y Raster (puro dominio) ----------
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

# ---------- GeoTransform helpers (afines a GDAL pero sin dependencia) ----------
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
    py = (miny - maxy) / float(height)  # negativo (origen en esquina sup-izq)
    return (minx, px, 0.0, maxy, 0.0, py)

def pixel_to_world(col: int, row: int, gt: GeoTransform) -> Tuple[float, float]:
    x0, px, rx, y0, ry, py = gt
    x = x0 + col * px + row * rx
    y = y0 + col * ry + row * py
    return x, y

def world_to_pixel(x: float, y: float, gt: GeoTransform) -> Tuple[float, float]:
    x0, px, rx, y0, ry, py = gt
    det = px * py - rx * ry
    if abs(det) < 1e-18:
        raise ValueError("GeoTransform no invertible (det≈0).")
    inv00 =  py / det; inv01 = -rx / det
    inv10 = -ry / det; inv11 =  px / det
    dx = x - x0; dy = y - y0
    col = inv00 * dx + inv01 * dy
    row = inv10 * dx + inv11 * dy
    return col, row

def _gt_close(a: GeoTransform, b: GeoTransform, tol: float = 1e-6) -> bool:
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
    return (f"Bounds(minx={b.minx:.{ndigits}f}, miny={b.miny:.{ndigits}f}, "
            f"maxx={b.maxx:.{ndigits}f}, maxy={b.maxy:.{ndigits}f})")

__all__ = [
    "GeoTransform","Bounds","CRSRef","GeoProfile","GeoRaster","geotransform_bounds",
    "bounds_to_geotransform","pixel_to_world","world_to_pixel",
    "validate_profile_compat","pretty_bounds","DTypeStr",
]
