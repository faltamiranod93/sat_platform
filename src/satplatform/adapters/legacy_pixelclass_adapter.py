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

    def predict_points(self, df) -> "np.ndarray":  # noqa: F821
        """No soportado: las reglas normalizan por percentil sobre la escena completa
        (`_norm`), sin sentido a nivel de punto. Este clasificador no se usa en la
        evaluación por puntos (TFC/CV); se define solo para conformar el Protocol."""
        raise NotImplementedError(
            "LegacyPixelClassifier no soporta predict_points (normalización por percentil de escena)."
        )

    def predict(self, bands: BandSet, *, calibration_id: Optional[str] = None) -> GeoRaster:
        available = bands.names()
        required_candidates = ("B03", "B04", "B08", "B11")
        need = [b for b in required_candidates if b in available]
        if len(need) < 3:
            raise ValueError(
                f"Se requieren al menos 3 bandas entre {required_candidates}; presentes: {sorted(available)}"
            )

        base = bands.bands[need[0]]
        h, w = base.data.shape
        zeros = np.zeros((h, w), dtype=np.float32)

        b03 = bands.bands["B03"].data if "B03" in available else zeros
        b04 = bands.bands["B04"].data if "B04" in available else zeros
        b08 = bands.bands["B08"].data if "B08" in available else zeros
        b11 = bands.bands["B11"].data if "B11" in available else zeros

        # Normaliza 0..1 si vienen en reflectancias 0..10000 típicas
        def _norm(x):
            x = x.astype(np.float32)
            m = np.nanpercentile(x, 99.5)
            return np.clip(x / (m + 1e-6), 0, 1)

        n3, n4, n8, n11 = _norm(b03), _norm(b04), _norm(b08), _norm(b11)

        is_water = (n8 < 0.15) & (n3 < 0.15)
        is_tail  = (n4 > 0.5) & (n11 > 0.4)

        class_by_name = {c.name.lower(): c.id for c in self.classes_def}
        agua_id = class_by_name.get("agua", 1)
        relave_id = class_by_name.get("relave", 2)
        terreno_id = class_by_name.get("terreno", 3)

        out = np.zeros((h, w), dtype=np.uint8)
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
