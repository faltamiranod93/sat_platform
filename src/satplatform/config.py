# src/satplatform/config.py
from __future__ import annotations
import re

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .contracts.core import ClassLabel, S2BandName
from .contracts.geo import CRSRef

# Placeholders permitidos por clave (defensa temprana)
INPUT_PLACEHOLDERS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "s2_dir": ("tile",),
    "band_file": ("date", "band"),
    "roi_file": (),  # ninguno
})
OUTPUT_PLACEHOLDERS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "stack": ("date",),
    "hist_norm": ("date",),
    "classmap": ("date",),
})

WKT_START = re.compile(r"^\s*(PROJCS|GEOGCS|PROJCRS|GEOGCRS)\s*[\(\[]", re.IGNORECASE)

class Settings(BaseSettings):
    """
    Config unificada del proyecto. No toca disco.
    Debe ser construida y provista por composition/di.py (CLI/UI/adapters).
    """
    # --- básicos ---
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    crs_out: CRSRef = Field(default_factory=lambda: CRSRef.from_epsg(32719))

    # --- estructura de trabajo (relativa a project_root) ---
    work_roi_dir: Path = Path("work/roi")
    work_products_dir: Path = Path("work/artifacts")
    report_dir: Path = Path("work/report")

    # --- herramientas externas ---
    gdalwarp_exe: Optional[Path] = None  # si None, se busca en PATH desde adapters

    # --- dominio ---
    band_order: tuple[S2BandName, ...] = ("B02", "B03", "B04")  # ajusta a tu flujo
    classes: tuple[ClassLabel, ...] = ()  # inmutable

    # --- patrones de I/O (centralizados, relativos a project_root) ---
    input_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "s2_dir": "data/s2/{tile}",
            "band_file": "{date}_{band}.tif",
            "roi_file": "data/roi/roi.geojson",
        }
    )
    output_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "stack": "artifacts/{date}/stack.tif",
            "hist_norm": "artifacts/{date}/stack_hn.tif",
            "classmap": "artifacts/{date}/classmap.tif",
        }
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # ----------------------------
    # Normalizadores / validadores
    # ----------------------------
    @field_validator("project_root", mode="before")
    @classmethod
    def _abs_root(cls, v: Path | str) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("crs_out", mode="before")
    @classmethod
    def _parse_crs(cls, v) -> CRSRef:
        # Acepta "EPSG:32719" o WKT o dict-like
        if isinstance(v, CRSRef): return v
        s = str(v).strip()
        if s.upper().startswith("EPSG:"):
            return CRSRef.from_epsg(int(s.split(":")[1]))
        if WKT_START.match(s):
            return CRSRef.from_wkt(s)
        raise ValueError(f"crs_out inválido: {v}")

    @field_validator("work_roi_dir", "work_products_dir", "report_dir", mode="after")
    @classmethod
    def _rel_to_root(cls, p: Path, info) -> Path:
        root: Path = info.data.get("project_root")
        return p if p.is_absolute() else (root / p)

    @field_validator("gdalwarp_exe", mode="after")
    @classmethod
    def _abs_gdalwarp(cls, p: Optional[Path], info) -> Optional[Path]:
        if p is None:
            return None
        root: Path = info.data.get("project_root")
        return p if p.is_absolute() else (root / p)

    @field_validator("input_patterns")
    @classmethod
    def _validate_input_patterns(cls, d: Dict[str, str]) -> Dict[str, str]:
        for k, pat in d.items():
            allowed = set(INPUT_PLACEHOLDERS.get(k, ()))
            used = {frag[1] for frag in _iter_placeholders(pat)}
            unknown = used - allowed
            if unknown:
                raise ValueError(f"input_patterns[{k}] usa placeholders no permitidos: {sorted(unknown)}")
        return d

    @field_validator("output_patterns")
    @classmethod
    def _validate_output_patterns(cls, d: Dict[str, str]) -> Dict[str, str]:
        for k, pat in d.items():
            allowed = set(OUTPUT_PLACEHOLDERS.get(k, ()))
            used = {frag[1] for frag in _iter_placeholders(pat)}
            unknown = used - allowed
            if unknown:
                raise ValueError(f"output_patterns[{k}] usa placeholders no permitidos: {sorted(unknown)}")
        return d

    # ----------------------------
    # Helpers puros (sin side-effects)
    # ----------------------------
    def out_path(self, key: str, **fmt) -> Path:
        """Resuelve patrón de salida (no crea carpetas)."""
        pat = self.output_patterns[key]
        return (self.project_root / pat.format(**fmt)).resolve()

    def in_path(self, key: str, **fmt) -> Path:
        pat = self.input_patterns[key]
        return (self.project_root / pat.format(**fmt)).resolve()

# Utilidad interna: detectar {placeholders}
def _iter_placeholders(fmt: str):
    # Busca {name} muy simple; evita formatear para no explotar
    start = 0
    while True:
        i = fmt.find("{", start)
        if i == -1: break
        j = fmt.find("}", i + 1)
        if j == -1: break
        name = fmt[i+1:j].strip()
        if name:
            yield (i, name)
        start = j + 1

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Instancia cacheada. Úsala SOLO desde composition/di.py o CLI/UI.
    Prohibido usarla en services/ (dominio). Para tests, recuerda limpiar:
        get_settings.cache_clear()
    """
    return Settings()
