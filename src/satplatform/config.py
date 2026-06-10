# src/satplatform/config.py
from __future__ import annotations

"""
SPRINT 1 — Bug corregido:
  - Eliminado doble model_config (antes frozen=True era sobreescrito por
    SettingsConfigDict, perdiendo la inmutabilidad).
  - Settings hereda de BaseSettings para que pydantic-settings cargue
    valores desde .env y variables de entorno SAT_*.
  - _parse_crs (método muerto) eliminado; la conversión vive en crs_out_ref().
"""

import re
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .contracts.core import ClassLabel, S2BandName
from .contracts.geo import CRSRef

# ---------------------------------------------------------------------------
# Placeholders permitidos por clave de patrón
# ---------------------------------------------------------------------------

INPUT_PLACEHOLDERS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "safe_dir":    ("product",),
    "granule_dir": ("product", "granule"),
    "jp2_file":    ("product", "granule", "tile", "sensing", "band", "res"),
    "scl_file":    ("product", "granule", "tile", "sensing", "res"),
    "mask_file":   ("product", "granule", "band"),
    "roi_file":    (),
})

OUTPUT_PLACEHOLDERS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "classmap":        ("date", "classifier"),
    "classmap_vis":    ("date", "classifier"),
    "features":        ("date",),
    "compare_summary": ("name",),
})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Configuración unificada del proyecto.

    Hereda de BaseSettings (pydantic-settings) para cargar valores desde:
      1. Argumentos explícitos al construir (mayor precedencia)
      2. Variables de entorno con prefijo SAT_  (ej. SAT_PROJECT_ROOT=...)
      3. Archivo .env en el directorio de trabajo
      4. Valores por defecto declarados aquí

    INMUTABILIDAD: model_config se declara UNA SOLA VEZ con frozen=True
    incluido. Antes había dos asignaciones de model_config; la segunda
    (SettingsConfigDict) sobreescribía a la primera (ConfigDict frozen=True)
    y la restricción de inmutabilidad se perdía silenciosamente.

    Uso correcto:
        # Desde YAML vía composition/di.py:
        settings = Settings(**yaml_data)

        # Desde entorno:
        settings = Settings()            # lee SAT_PROJECT_ROOT, etc.

        # Para tests (sin tocar disco):
        settings = Settings(project_root=Path("/tmp/test"))
    """

    # ── Un único model_config ────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        frozen=True,                  # <── inmutabilidad restaurada
        env_file=".env",
        env_prefix="SAT_",
        env_nested_delimiter="__",
        extra="forbid",
        arbitrary_types_allowed=True, # necesario para Path y tuple[ClassLabel,...]
    )

    # ── Campos básicos ───────────────────────────────────────────────────────
    project_root: Path

    # Guardamos como str para que pydantic-settings NO intente json.loads
    crs_out: str = "EPSG:32719"

    # ── Estructura de trabajo (relativa a project_root) ──────────────────────
    work_roi_dir:      Path = Path("02-Work/ROI")
    work_products_dir: Path = Path("03-Products")
    report_dir:        Path = Path("03-Products/REPORT")

    # ── Herramientas externas ────────────────────────────────────────────────
    gdalwarp_exe: Optional[Path] = None  # None → buscar en PATH desde adapters

    # ── Dominio ──────────────────────────────────────────────────────────────
    band_order: tuple[S2BandName, ...] = ("B02", "B03", "B04")
    classes:    tuple[ClassLabel, ...] = ()

    input_patterns: Dict[str, str] = Field(default_factory=lambda: {
        "safe_dir":    "01-Raw/s2/{product}.SAFE",
        "granule_dir": "01-Raw/s2/{product}.SAFE/GRANULE/{granule}",
        "jp2_file":    "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA"
                       "/R{res}/T{tile}_{sensing}_{band}_{res}.jp2",
        "scl_file":    "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA"
                       "/R{res}/T{tile}_{sensing}_SCL_{res}.jp2",
        "mask_file":   "01-Raw/s2/{product}.SAFE/GRANULE/{granule}"
                       "/QI_DATA/MSK_QUALIT_{band}.jp2",
        "roi_file":    "00-Config/roi_master.geojson",
    })

    output_patterns: Dict[str, str] = Field(default_factory=lambda: {
        "classmap":        "03-Products/CLASSMAP/{date}/classmap_{classifier}.tif",
        "classmap_vis":    "03-Products/VIS/{date}/classmap_{classifier}.png",
        "features":        "02-Work/FEATURES/{date}/features.tif",
        "compare_summary": "04-Analysis/CLASSMAP-COMPARE/{name}.csv",
    })

    # ── Validadores ──────────────────────────────────────────────────────────

    @field_validator("project_root", mode="before")
    @classmethod
    def _abs_root(cls, v: Path | str) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("crs_out", mode="before")
    @classmethod
    def _non_empty_crs(cls, v: str) -> str:
        v2 = str(v).strip()
        if not v2:
            raise ValueError("crs_out no puede ser vacío")
        return v2

    @field_validator("work_roi_dir", "work_products_dir", "report_dir", mode="after")
    @classmethod
    def _rel_to_root(cls, p: Path, info) -> Path:
        root: Path | None = info.data.get("project_root")
        if root is None:
            return p
        return p if p.is_absolute() else (root / p)

    @field_validator("gdalwarp_exe", mode="after")
    @classmethod
    def _abs_gdalwarp(cls, p: Optional[Path], info) -> Optional[Path]:
        if p is None:
            return None
        root: Path | None = info.data.get("project_root")
        if root is None:
            return p
        return p if p.is_absolute() else (root / p)

    @field_validator("input_patterns")
    @classmethod
    def _check_in(cls, d: Dict[str, str]) -> Dict[str, str]:
        for k, pat in d.items():
            allowed = set(INPUT_PLACEHOLDERS.get(k, ()))
            used    = {name for _, name in _iter_placeholders(pat)}
            unknown = used - allowed
            if unknown:
                raise ValueError(
                    f"input_patterns[{k!r}] usa placeholders no permitidos: "
                    f"{sorted(unknown)}"
                )
        return d

    @field_validator("output_patterns")
    @classmethod
    def _check_out(cls, d: Dict[str, str]) -> Dict[str, str]:
        for k, pat in d.items():
            allowed = set(OUTPUT_PLACEHOLDERS.get(k, ()))
            used    = {name for _, name in _iter_placeholders(pat)}
            unknown = used - allowed
            if unknown:
                raise ValueError(
                    f"output_patterns[{k!r}] usa placeholders no permitidos: "
                    f"{sorted(unknown)}"
                )
        return d

    # ── API pública ──────────────────────────────────────────────────────────

    def crs_out_ref(self) -> CRSRef:
        """Convierte crs_out (str) a CRSRef de dominio (sin GDAL)."""
        s = self.crs_out.strip()
        if s.upper().startswith("EPSG:"):
            code = int(s.split(":")[1])
            return CRSRef.from_epsg(code)
        return CRSRef.from_wkt(s)

    def in_path(self, key: str, **fmt) -> Path:
        """Resuelve patrón de entrada (no crea carpetas)."""
        pat = self.input_patterns[key]
        return (self.project_root / pat.format(**fmt)).resolve()

    def out_path(self, key: str, **fmt) -> Path:
        """Resuelve patrón de salida (no crea carpetas)."""
        pat = self.output_patterns[key]
        return (self.project_root / pat.format(**fmt)).resolve()


# ---------------------------------------------------------------------------
# Helpers internos (módulo, no métodos de clase)
# ---------------------------------------------------------------------------

def _iter_placeholders(fmt: str):
    """Genera (posición, nombre) para cada {placeholder} en fmt."""
    start = 0
    while True:
        i = fmt.find("{", start)
        if i == -1:
            break
        j = fmt.find("}", i + 1)
        if j == -1:
            break
        name = fmt[i + 1: j].strip()
        if name:
            yield (i, name)
        start = j + 1


# ---------------------------------------------------------------------------
# Instancia cacheada (solo para CLI/UI — prohibida en services y tests)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Instancia cacheada para uso desde CLI o composición.

    PROHIBIDO en services/ (el dominio no debe depender de configuración
    global).  En tests, limpia la caché antes de cada caso:
        get_settings.cache_clear()
    """
    return Settings()
