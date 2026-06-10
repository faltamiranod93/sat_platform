"""Soporte de visualización de escena para el explorador (dominio puro, sin UI).

Compone RGB a partir de un BandSet, proyecta puntos UTM del GeoJSON a píxel para
el overlay, y extrae la firma de un píxel (clic). No hace I/O: recibe el BandSet
ya cargado (por multiband_loader con el reader que corrige la georef).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..contracts.core import S2BandName
from ..contracts.geo import world_to_pixel
from ..contracts.products import BandSet

# Presets RGB (orden R, G, B) — convención Sentinel-2.
RGB_PRESETS: Dict[str, tuple[S2BandName, S2BandName, S2BandName]] = {
    "TrueColor": ("B04", "B03", "B02"),
    "FalseColor": ("B08", "B04", "B03"),
    "SWIR": ("B12", "B11", "B04"),
}


@dataclass
class SceneViewService:
    def rgb_composite(
        self,
        bandset: BandSet,
        preset: str = "TrueColor",
        pmin: float = 2.0,
        pmax: float = 98.0,
    ) -> np.ndarray:
        """Compone una imagen RGB uint8 (H, W, 3) con estiramiento por percentiles."""
        if preset not in RGB_PRESETS:
            raise ValueError(f"Preset no soportado: {preset}. Opciones: {list(RGB_PRESETS)}")
        chans = []
        for b in RGB_PRESETS[preset]:
            if b not in bandset.bands:
                raise KeyError(f"Falta banda {b} para preset {preset}")
            arr = bandset.bands[b].data.astype(np.float32)
            lo, hi = np.nanpercentile(arr, [pmin, pmax])
            c = np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
            chans.append((c * 255.0).astype(np.uint8))
        return np.stack(chans, axis=-1)

    def points_to_pixels(self, utm_df: pd.DataFrame, profile) -> pd.DataFrame:
        """Añade columnas col,row (float) proyectando UTM_E/UTM_N a píxel."""
        cols = np.empty(len(utm_df))
        rows = np.empty(len(utm_df))
        gt = profile.transform
        for k, (_, r) in enumerate(utm_df.iterrows()):
            c, rr = world_to_pixel(float(r["UTM_E"]), float(r["UTM_N"]), gt)
            cols[k] = c
            rows[k] = rr
        out = utm_df.copy()
        out["col"] = cols
        out["row"] = rows
        return out

    def pixel_signature(self, bandset: BandSet, col: float, row: float) -> Dict[str, float]:
        """Espectro de un píxel: {banda: valor}. Vacío si el píxel cae fuera."""
        ci, ri = int(round(col)), int(round(row))
        out: Dict[str, float] = {}
        for name, r in bandset.bands.items():
            h, w = r.data.shape
            if 0 <= ri < h and 0 <= ci < w:
                out[name] = float(r.data[ri, ci])
        return out

    @staticmethod
    def presets() -> Sequence[str]:
        return tuple(RGB_PRESETS.keys())


__all__ = ["SceneViewService", "RGB_PRESETS"]
