# src/satplatform/contracts/core.py
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

# -------------------------
# Bandas y features
# -------------------------
S2BandName = Literal[
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12"
]
FeatureName = Literal["H", "S", "L"]
BandName = S2BandName | FeatureName

ClassId = PositiveInt

# -------------------------
# Colores tipados
# -------------------------
class RGB8(BaseModel):
    model_config = ConfigDict(frozen=True)
    r: int = Field(200, ge=0, le=255)
    g: int = Field(200, ge=0, le=255)
    b: int = Field(200, ge=0, le=255)
    def as_tuple(self) -> tuple[int, int, int]: return (self.r, self.g, self.b)
    def to_hex(self) -> str: return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

# -------------------------
# Etiquetas de clase
# -------------------------
class MacroClass(str, Enum):
    AGUA = "Agua"
    RELAVE = "Relave"
    TERRENO = "Terreno"

class ClassLabel(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: ClassId
    name: str
    macro: MacroClass
    color: RGB8 = RGB8()

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("name no puede ser vacío")
        return v2

# -------------------------
# Identidad de escena
# -------------------------
# zona 1..60, banda latitudinal C..X (sin I/O), 2 letras (sin I/O)
_MGRS_TILE_RE = re.compile(
    r"^(?P<zone>(?:[1-9]|[1-5][0-9]|60))(?P<band>[C-HJ-NP-X])(?P<sq>[A-HJ-NP-Z]{2})$"
)

class SceneId(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: date
    tile: Optional[str] = None  # ej. "19HFE"

    @field_validator("tile")
    @classmethod
    def _valid_tile(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v2 = v.strip().upper()
        m = _MGRS_TILE_RE.match(v2)
        if not m:
            raise ValueError(f"tile MGRS inválido: {v}")
        # opcional: validez semántica ya garantizada por regex (1..60)
        return v2

# -------------------------
# Calibraciones / normalización
# -------------------------
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

class CalibrationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: str = "1.0.0"
    ref_date: date
    source: str | None = None
    checksum: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError("schema_version debe ser SemVer (e.g., 1.0.0)")
        return v

    @field_validator("checksum")
    @classmethod
    def _hexsum(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (_HEX40_RE.match(v) or _HEX64_RE.match(v)):
            raise ValueError("checksum debe ser SHA1(40) o SHA256(64) hex")
        return v

# -------------------------
# Ejecuciones / auditoría
# -------------------------
# core.py
class Stage(str, Enum):
    STACK = "stack"
    HIST_NORM = "hist_norm"
    CLASSMAP = "classmap"
    EXPORT = "export"

class RunMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scene: SceneId
    roi_id: str | None = None
    calibration: CalibrationSpec | None = None
    notes: str | None = None
    ended_at: datetime | None = None

    def end_now(self) -> "RunMeta":
        return self.model_copy(update={"ended_at": datetime.now(timezone.utc)})

    @property
    def duration_s(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

class RunError(BaseModel):
    model_config = ConfigDict(frozen=True)
    stage: Stage
    message: str
    detail: str | None = None
