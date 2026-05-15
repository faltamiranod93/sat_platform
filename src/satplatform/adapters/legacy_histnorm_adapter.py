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
        """Conversión RGB→HSL preservando la relación cromática entre canales.

        BUG HISTÓRICO (corregido): la versión anterior normalizaba cada canal
        por su PROPIO máximo (R/=R.max(), G/=G.max(), B/=B.max()), lo que
        rompía la proporción entre canales y producía colores artificiales
        antes de la conversión HSL.

        Estrategia actual:
          1. Detecta la escala (reflectancia 0..10000 o ya 0..1).
          2. Normaliza los tres canales con un ESCALAR COMÚN (el percentil 99.5
             conjunto), preservando la cromaticidad.
          3. Aplica la fórmula HSL estándar.
        """
        R = r.data.astype(np.float32)
        G = g.data.astype(np.float32)
        B = b.data.astype(np.float32)

        stacked = np.stack([R, G, B], axis=0)
        finite = np.isfinite(stacked)
        if finite.any():
            scale = float(np.nanpercentile(stacked[finite], 99.5))
        else:
            scale = 1.0
        if scale <= 1.5:  # ya parece estar en [0,1]
            scale = 1.0

        R = np.clip(R / (scale + 1e-6), 0.0, 1.0)
        G = np.clip(G / (scale + 1e-6), 0.0, 1.0)
        B = np.clip(B / (scale + 1e-6), 0.0, 1.0)

        cmax = np.maximum(np.maximum(R, G), B)
        cmin = np.minimum(np.minimum(R, G), B)
        delta = cmax - cmin
        L = (cmax + cmin) / 2.0

        S = np.zeros_like(L)
        mask = delta > 1e-6
        # HSL estándar: S = delta / (1 - |2L - 1|)
        S[mask] = delta[mask] / (1.0 - np.abs(2.0 * L[mask] - 1.0) + 1e-6)

        H = np.zeros_like(L)
        idx_r = (cmax == R) & mask
        idx_g = (cmax == G) & mask
        idx_b = (cmax == B) & mask
        H[idx_r] = ((G[idx_r] - B[idx_r]) / (delta[idx_r] + 1e-6)) % 6.0
        H[idx_g] = ((B[idx_g] - R[idx_g]) / (delta[idx_g] + 1e-6)) + 2.0
        H[idx_b] = ((R[idx_b] - G[idx_b]) / (delta[idx_b] + 1e-6)) + 4.0
        H = H / 6.0

        H = np.clip(H, 0.0, 1.0).astype(np.float32, copy=False)
        S = np.clip(S, 0.0, 1.0).astype(np.float32, copy=False)
        L = np.clip(L, 0.0, 1.0).astype(np.float32, copy=False)

        from ..contracts.geo import GeoProfile, GeoRaster
        p = r.profile
        prof_out = GeoProfile(
            count=1, dtype="float32",
            width=p.width, height=p.height,
            transform=p.transform, crs=p.crs, nodata=p.nodata,
        )
        return (
            GeoRaster(H, prof_out),
            GeoRaster(S, prof_out),
            GeoRaster(L, prof_out),
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
