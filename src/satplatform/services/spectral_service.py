# src/satplatform/services/spectral_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Mapping
import numpy as np

from ..contracts.core import S2BandName
from ..contracts.products import BandSet
from ..contracts.geo import GeoRaster, GeoProfile

@dataclass
class SpectralService:
    """
    Servicio de features espectrales puro (sin I/O).
    - Asume arrays float32/float64; si llegan enteros, normaliza localmente.
    - Mantiene profile/coordenadas.
    """

    def ensure_float01(self, arr: np.ndarray) -> np.ndarray:
        # Heurística: si parece reflectancia escalada 0..10000 la normaliza.
        a = arr.astype("float32", copy=False)
        a = np.where(a > 1.0, a / 10000.0, a)
        return np.clip(a, 0.0, 1.0)

    def rgb_to_hsl(self, bandset: BandSet, order: Sequence[S2BandName] = ("B02","B03","B04")) -> GeoRaster:
        for b in order:
            if b not in bandset.bands:
                raise KeyError(f"Faltan bandas requeridas: {order}. No está {b}.")

        r = self.ensure_float01(bandset.bands[order[0]].data)
        g = self.ensure_float01(bandset.bands[order[1]].data)
        b = self.ensure_float01(bandset.bands[order[2]].data)

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        L = (maxc + minc) / 2.0

        diff = maxc - minc
        S = np.zeros_like(L, dtype="float32")
        nz = diff > 1e-12
        S[nz] = (diff[nz] / (1.0 - np.abs(2.0 * L[nz] - 1.0) + 1e-12)).astype("float32")

        H = np.zeros_like(L, dtype="float32")
        r_is = (maxc == r) & nz
        g_is = (maxc == g) & nz
        b_is = (maxc == b) & nz
        denom = diff + 1e-12

        H[r_is] = ((g[r_is] - b[r_is]) / denom[r_is]) % 6.0
        H[g_is] = ((b[g_is] - r[g_is]) / denom[g_is]) + 2.0
        H[b_is] = ((r[b_is] - g[b_is]) / denom[b_is]) + 4.0
        H = (H / 6.0).astype("float32")

        H = np.clip(H, 0.0, 1.0)
        S = np.clip(S, 0.0, 1.0)
        L = np.clip(L, 0.0, 1.0).astype("float32", copy=False)

        p0 = bandset.bands[order[0]].profile
        prof = GeoProfile(
            count=3,
            dtype="float32",
            width=p0.width,
            height=p0.height,
            transform=p0.transform,
            crs=p0.crs,
            nodata=p0.nodata,
        )
        data = np.stack([H, S, L], axis=0).astype("float32", copy=False)
        return GeoRaster(data=data, profile=prof)

    def nd_index(self, top: np.ndarray, bot: np.ndarray) -> np.ndarray:
        top = self.ensure_float01(top)
        bot = self.ensure_float01(bot)
        return ((top - bot) / (top + bot + 1e-12)).astype("float32")

    def compute_indices(self, bandset: BandSet, indices: Iterable[str]) -> GeoRaster:
        """
        indices: lista de nombres; soporta NDVI=B08/B04, NDBI=B11/B08, NDWI=B03/B08 (clásicos)
        """
        required: dict[str, tuple[S2BandName, S2BandName]] = {
            "NDVI": ("B08", "B04"),
            "NDBI": ("B11", "B08"),
            "NDWI": ("B03", "B08"),
        }
        use: dict[str, tuple[S2BandName, S2BandName]] = {}
        for name in indices:
            if name not in required:
                raise ValueError(f"Índice no soportado: {name}")
            use[name] = required[name]

        for name, (a, b) in use.items():
            if a not in bandset.bands or b not in bandset.bands:
                raise KeyError(f"Para {name} faltan bandas {a}/{b}")

        arrs = []
        for name, (a, b) in use.items():
            arrs.append(self.nd_index(bandset.bands[a].data, bandset.bands[b].data))

        first = next(iter(bandset.bands.values()))
        prof = GeoProfile(
            count=len(arrs),
            dtype="float32",
            width=first.profile.width,
            height=first.profile.height,
            transform=first.profile.transform,
            crs=first.profile.crs,
            nodata=first.profile.nodata,
        )
        data = np.stack(arrs, axis=0).astype("float32", copy=False)
        return GeoRaster(data=data, profile=prof)
