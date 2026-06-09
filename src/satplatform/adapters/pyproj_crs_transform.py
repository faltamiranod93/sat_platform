# src/satplatform/adapters/pyproj_crs_transform.py
"""Adapter de reproyección de coordenadas basado en pyproj.

Implementa CrsTransformPort. Cachea los Transformer por par (src, dst) para
evitar reconstruirlos en bucles sobre muchos puntos/escenas.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..ports.crs_transform import CrsTransformPort


class PyprojCrsTransform(CrsTransformPort):
    def __init__(self) -> None:
        # Import diferido: pyproj solo se exige si el adapter se usa.
        from pyproj import Transformer  # noqa: F401

        self._Transformer = Transformer
        self._cache: Dict[Tuple[int, int], object] = {}

    def _transformer(self, src_epsg: int, dst_epsg: int):
        key = (int(src_epsg), int(dst_epsg))
        tr = self._cache.get(key)
        if tr is None:
            tr = self._Transformer.from_crs(key[0], key[1], always_xy=True)
            self._cache[key] = tr
        return tr

    def transform_xy(
        self, x: float, y: float, src_epsg: int, dst_epsg: int
    ) -> Tuple[float, float]:
        x2, y2 = self._transformer(src_epsg, dst_epsg).transform(float(x), float(y))
        return float(x2), float(y2)


__all__ = ["PyprojCrsTransform"]
