## `src/satplatform/adapters/legacy_pixelclass_adapter.py`

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..contracts.products import BandSet
from ..contracts.core import ClassLabel
from ..contracts.geo import GeoRaster, GeoProfile
from ..ports.pixel_class import PixelClassifierPort

@dataclass(frozen=True)
class LegacyPixelClassifier(PixelClassifierPort):
    """Clasificador mínimo basado en reglas (sin dependencias externas).

    Reglas por defecto (ejemplo):
      - Agua: B03 baja y B08 baja
      - Relave: alta reflectancia en B04 y B11 (claros)
      - Terreno: resto
    Ajusta según tu dominio.
    """
    classes_def: Sequence[ClassLabel]

    def name(self) -> str:
        return "legacy-rules-v1"

    def classes(self) -> Sequence[ClassLabel]:
        return self.classes_def

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        need = []
        # usa bandas comunes
        for b in ("B03", "B04", "B08", "B11"):
            if bands.has(b):
                need.append(b)
        if len(need) < 3:
            raise ValueError("Se requieren al menos 3 bandas entre B03,B04,B08,B11")

        # Referencia de tamaño y perfil
        base = bands.bands[need[0]]
        h, w = base.data.shape
        out = np.zeros((h, w), dtype=np.uint8)

        b03 = bands.bands.get("B03").data if bands.has("B03") else np.zeros_like(out)
        b04 = bands.bands.get("B04").data if bands.has("B04") else np.zeros_like(out)
        b08 = bands.bands.get("B08").data if bands.has("B08") else np.zeros_like(out)
        b11 = bands.bands.get("B11").data if bands.has("B11") else np.zeros_like(out)

        # Normaliza 0..1 si vienen en reflectancias 0..10000 típicas
        def _norm(x):
            x = x.astype(np.float32)
            m = np.nanpercentile(x, 99.5)
            return np.clip(x / (m + 1e-6), 0, 1)

        n3, n4, n8, n11 = _norm(b03), _norm(b04), _norm(b08), _norm(b11)

        # Heurísticas simples
        is_water = (n8 < 0.15) & (n3 < 0.15)
        is_tail  = (n4 > 0.5) & (n11 > 0.4)
        # Terreno = resto
        class_by_name = {c.name.lower(): c.id for c in self.classes_def}
        agua_id = class_by_name.get("agua", 1)
        relave_id = class_by_name.get("relave", 2)
        terreno_id = class_by_name.get("terreno", 3)

        out[is_water] = agua_id
        out[is_tail] = relave_id
        out[(~is_water) & (~is_tail)] = terreno_id

        profile = GeoProfile(
            count=1,
            dtype="uint8",
            width=w,
            height=h,
            transform=base.profile.transform,
            crs=base.profile.crs,
            nodata=0,
        )
        return GeoRaster(data=out, profile=profile)
