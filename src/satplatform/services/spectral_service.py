# src/satplatform/services/spectral_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple
import numpy as np

from ..contracts.core import S2BandName
from ..contracts.products import BandSet
from ..contracts.geo import GeoRaster, GeoProfile

# Catálogo de índices de diferencia normalizada (a-b)/(a+b).
# NDSI ≡ MNDWI numéricamente (misma fórmula B03/B11); se mantienen ambos por trazabilidad.
ND_INDICES: dict[str, tuple[S2BandName, S2BandName]] = {
    "NDVI": ("B08", "B04"),
    "NDWI": ("B03", "B08"),
    "NDBI": ("B11", "B08"),
    "MNDWI": ("B03", "B11"),
    "NDSI": ("B03", "B11"),
}

# Índices con fórmula propia (más de 2 bandas). Devuelven array elementwise.
# BSI = ((B11+B04)-(B08+B02)) / ((B11+B04)+(B08+B02))
CUSTOM_INDICES: tuple[str, ...] = ("BSI",)

SUPPORTED_INDICES: tuple[str, ...] = tuple(ND_INDICES.keys()) + CUSTOM_INDICES


@dataclass
class SpectralService:
    """
    Servicio de features espectrales puro (sin I/O).
    - Asume arrays float32/float64; si llegan enteros, normaliza localmente.
    - Las fórmulas operan elementwise sobre arrays de cualquier shape (2D escena
      o 1D columnas de muestras), garantizando consistencia train/predict.
    """

    def ensure_float01(self, arr: np.ndarray) -> np.ndarray:
        # Heurística: si parece reflectancia escalada 0..10000 la normaliza.
        a = arr.astype("float32", copy=False)
        a = np.where(a > 1.0, a / 10000.0, a)
        return np.clip(a, 0.0, 1.0)

    # ---------- HSL ----------
    def hsl_from_rgb(
        self, r: np.ndarray, g: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convierte arrays R,G,B (cualquier shape) a (H,S,L) en escala 0..1.

        Núcleo compartido por rgb_to_hsl (escena 2D) y FeatureService (muestras 1D).
        """
        r = self.ensure_float01(r)
        g = self.ensure_float01(g)
        b = self.ensure_float01(b)

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
        return H, S, L

    def rgb_to_hsl(self, bandset: BandSet, order: Sequence[S2BandName] = ("B02","B03","B04")) -> GeoRaster:
        for b in order:
            if b not in bandset.bands:
                raise KeyError(f"Faltan bandas requeridas: {order}. No está {b}.")

        H, S, L = self.hsl_from_rgb(
            bandset.bands[order[0]].data,
            bandset.bands[order[1]].data,
            bandset.bands[order[2]].data,
        )

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

    # ---------- Índices espectrales ----------
    def nd_index(self, top: np.ndarray, bot: np.ndarray) -> np.ndarray:
        top = self.ensure_float01(top)
        bot = self.ensure_float01(bot)
        return ((top - bot) / (top + bot + 1e-12)).astype("float32")

    def index_from_arrays(self, name: str, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        """Calcula un índice espectral a partir de un dict {banda: array}.

        Funciona con arrays 2D (escena) o 1D (columnas de muestras). Usado tanto
        en inferencia como en entrenamiento → mismo resultado por construcción.
        """
        if name in ND_INDICES:
            a, b = ND_INDICES[name]
            self._require_bands(name, arrays, (a, b))
            return self.nd_index(arrays[a], arrays[b])
        if name == "BSI":
            self._require_bands(name, arrays, ("B11", "B04", "B08", "B02"))
            b11 = self.ensure_float01(arrays["B11"])
            b04 = self.ensure_float01(arrays["B04"])
            b08 = self.ensure_float01(arrays["B08"])
            b02 = self.ensure_float01(arrays["B02"])
            num = (b11 + b04) - (b08 + b02)
            den = (b11 + b04) + (b08 + b02) + 1e-12
            return (num / den).astype("float32")
        raise ValueError(f"Índice no soportado: {name}. Soportados: {SUPPORTED_INDICES}")

    @staticmethod
    def _require_bands(name: str, arrays: Mapping[str, np.ndarray], needed: Iterable[str]) -> None:
        missing = [b for b in needed if b not in arrays]
        if missing:
            raise KeyError(f"Para {name} faltan bandas {missing}")

    def compute_indices(self, bandset: BandSet, indices: Iterable[str]) -> GeoRaster:
        """Calcula índices sobre una escena (BandSet) → GeoRaster apilado.

        Soporta los de ND_INDICES (NDVI, NDWI, NDBI, MNDWI, NDSI) y BSI.
        """
        names = list(indices)
        arrays = {n: r.data for n, r in bandset.bands.items()}
        arrs = [self.index_from_arrays(n, arrays) for n in names]

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
