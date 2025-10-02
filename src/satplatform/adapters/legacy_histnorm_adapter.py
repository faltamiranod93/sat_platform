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
