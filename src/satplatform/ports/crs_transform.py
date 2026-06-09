# src/satplatform/ports/crs_transform.py
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable


@runtime_checkable
class CrsTransformPort(Protocol):
    """Reproyección de coordenadas puntuales entre dos CRS (por código EPSG).

    Abstrae la dependencia geodésica concreta (pyproj/GDAL) para que los
    servicios de dominio no la importen directamente.

    Convención de ejes: x = Este/longitud, y = Norte/latitud (always_xy).
    """

    def transform_xy(
        self, x: float, y: float, src_epsg: int, dst_epsg: int
    ) -> Tuple[float, float]:
        """Transforma (x, y) de `src_epsg` a `dst_epsg`. Devuelve (x', y')."""
        ...


__all__ = ["CrsTransformPort"]
