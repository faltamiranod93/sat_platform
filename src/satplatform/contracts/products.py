from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Dict, Iterable, Sequence, Set, Literal, Mapping, FrozenSet

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .core import S2BandName, SceneId
from .geo import GeoRaster, GeoProfile, CRSRef, Bounds, validate_profile_compat

ResolutionM = Literal[10, 20, 60]


class Band(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: S2BandName
    raster: GeoRaster

    @property
    def profile(self) -> GeoProfile:
        return self.raster.profile


class BandSet(BaseModel):
    """
    Bandas coherentes a una misma resolución.
    - Colección inmutable en tiempo de ejecución.
    """
    model_config = ConfigDict(frozen=True)

    bands: Mapping[S2BandName, GeoRaster]
    resolution_m: ResolutionM

    @field_validator("bands")
    @classmethod
    def _freeze_and_validate_profiles(cls, v: Mapping[S2BandName, GeoRaster]) -> Mapping[S2BandName, GeoRaster]:
        if not isinstance(v, Mapping):
            v = dict(v)
        d = dict(v)
        if not d:
            return MappingProxyType(d)
        it = iter(d.values())
        first = next(it)
        for r in it:
            validate_profile_compat(first.profile, r.profile, require_same_crs=True)
        return MappingProxyType(d)

    @model_validator(mode="after")
    def _check_resolution_vs_pixel_size(self) -> BandSet:
        if self.bands:
            any_raster = next(iter(self.bands.values()))
            px, py = any_raster.profile.pixel_size()
            px = abs(px); py = abs(py)
            if abs(px - py) > 1e-6:
                raise ValueError(f"Pixel no cuadrado: px={px}, py={py}")
            if abs(px - self.resolution_m) > 0.6:
                raise ValueError(f"resolution_m={self.resolution_m} no coincide con pixel_size≈{px:.3f} m")
        return self

    def names(self) -> Set[S2BandName]:
        return set(self.bands.keys())

    def require(self, required: Iterable[S2BandName]) -> None:
        missing = [b for b in required if b not in self.bands]
        if missing:
            raise KeyError(f"Faltan bandas requeridas: {missing}")

    def subset(self, names: Sequence[S2BandName]) -> BandSet:
        self.require(names)
        sub = {n: self.bands[n] for n in names}
        return BandSet(bands=sub, resolution_m=self.resolution_m)

    def stack(self, order: Sequence[S2BandName]) -> GeoRaster:
        self.require(order)
        arrs = [self.bands[n].data for n in order]
        data = np.stack(arrs, axis=0).astype(arrs[0].dtype, copy=False)
        p0 = self.bands[order[0]].profile
        profile = GeoProfile(
            count=len(order),
            dtype=p0.dtype,
            width=p0.width,
            height=p0.height,
            transform=p0.transform,
            crs=p0.crs,
            nodata=p0.nodata,
        )
        return GeoRaster(data=data, profile=profile)


class S2Asset(BaseModel):
    """
    Metadatos de una escena Sentinel-2 (sin paths).
    """
    model_config = ConfigDict(frozen=True)

    scene: SceneId
    sensing_datetime: datetime
    cloud_percent: float | None = None

    available_bands: FrozenSet[S2BandName]
    resolutions: Mapping[S2BandName, ResolutionM]

    crs: CRSRef
    bounds: Bounds

    @field_validator("available_bands")
    @classmethod
    def _freeze_bands(cls, v):
        return frozenset(v)

    @field_validator("resolutions")
    @classmethod
    def _freeze_resolutions(cls, v):
        return MappingProxyType(dict(v))

    @field_validator("sensing_datetime")
    @classmethod
    def _ensure_utc(cls, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _check_band_coherence(self):
        res_bands = frozenset(self.resolutions.keys())
        if self.available_bands and res_bands - self.available_bands:
            raise ValueError("resolutions contiene bandas no listadas en available_bands")
        missing = self.available_bands - res_bands
        if missing:
            raise ValueError(f"Faltan resoluciones para bandas: {sorted(missing)}")
        return self

    def has(self, band: S2BandName) -> bool:
        return band in self.available_bands

    def bands_at(self, res: ResolutionM) -> FrozenSet[S2BandName]:
        return frozenset(b for b, r in self.resolutions.items() if r == res)