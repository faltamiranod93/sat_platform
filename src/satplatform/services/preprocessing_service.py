# src/satplatform/services/preprocessing_service.py
from __future__ import annotations

"""
Servicio de preprocesamiento (histogram normalization, RGB->HSL, stack)
— contracts-first, sin dependencias duras fuera de *ports*.

Casos cubiertos:
  • normalize_single: normaliza un raster y (opcional) escribe a disco
  • normalize_many: normaliza varias bandas, valida compatibilidad y (opcional)
    genera un stack multibanda
  • rgb_to_hsl: convierte 3 bandas (R,G,B) a (H,S,L) y (opcional) escribe salidas

Todas las escrituras de salida usan `Settings.out_path()` con llaves:
  - "hist_norm" para normalizaciones
  - "stack" para stack multibanda

Nota: El servicio es deliberadamente minimalista; no reproyecta ni resamplea.
      Si necesitas ROI, inyecta un ROIClipperPort y pásale `roi_geojson`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import Settings, get_settings
from ..contracts.core import S2BandName
from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, validate_profile_compat
from ..contracts.products import BandSet

from ..ports.preprocessing import PreprocessingPort, NormalizeSpec
from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort
from ..ports.roi import ROIClipperPort, GeoJSON


# ----------------------
# DTOs de entrada/salida
# ----------------------

@dataclass(frozen=True)
class NormalizeSingleSpec:
    date: str                           # YYYYMMDD
    out_path: Optional[Path] = None     # si None -> Settings.out_path('hist_norm')
    norm: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class NormalizeManySpec:
    date: str
    order: Optional[Tuple[S2BandName, ...]] = None
    write_individuals: bool = False
    out_dir: Optional[Path] = None      # si None -> usa Settings.out_path('hist_norm').parent
    write_stack: bool = True
    out_stack: Optional[Path] = None    # si None -> Settings.out_path('stack')
    norm: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class RGB2HSLSpec:
    date: str
    write_individuals: bool = True
    out_dir: Optional[Path] = None      # si None -> Settings.out_path('hist_norm').parent
    write_stack: bool = False
    out_stack: Optional[Path] = None
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class NormalizedItem:
    band: Optional[S2BandName]
    raster: GeoRaster
    out_path: Optional[Path]

@dataclass(frozen=True)
class NormalizeManyResult:
    items: Tuple[NormalizedItem, ...]
    band_order: Tuple[S2BandName, ...]
    stack_path: Optional[Path]
    resolution_m: int

@dataclass(frozen=True)
class RGB2HSLResult:
    H: NormalizedItem
    S: NormalizedItem
    L: NormalizedItem
    stack_path: Optional[Path]


# ----------------------
# Servicio
# ----------------------

@dataclass
class PreprocessingService:
    reader: Optional[object] = None
    writer: Optional[object] = None
    preproc: Optional[object] = None
    clipper: Optional[ROIClipperPort] = None
    settings: Settings = field(default_factory=get_settings)
    
    REQUIRED_RGB: tuple[str, str, str] = ("B02", "B03", "B04")

    # ---------- Helpers ----------
    @staticmethod
    def _coerce_resolution_from_profile(p: GeoProfile) -> int:
        px, py = p.pixel_size()
        gsd = min(abs(px), abs(py))
        return min((10, 20, 60), key=lambda r: abs(gsd - r))

    @staticmethod
    def _stable_band_order(keys: Iterable[S2BandName]) -> Tuple[S2BandName, ...]:
        def _key(k: str) -> Tuple[int, str]:
            try: n = int(k[1:3])
            except Exception: n = 99
            suf = k[3:] if len(k) > 3 else ""
            return (n, suf)
        return tuple(sorted(keys, key=_key))

    def _maybe_clip(self, r: GeoRaster, roi_g: Optional[GeoJSON], roi_crs: CRSRef) -> GeoRaster:
        if roi_g is None:
            return r
        if not self.clipper:
            raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
        return self.clipper.clip_raster(r, roi_g, roi_crs)

    # ---------- Casos de uso ----------
    def normalize_single(self, uri: str, spec: NormalizeSingleSpec) -> NormalizedItem:
        r = self.reader.read(uri)
        roi_crs = spec.roi_crs or self.settings.crs_out
        r = self._maybe_clip(r, spec.roi_geojson, roi_crs)
        rn = self.preproc.normalize(r, spec.norm)

        out_path: Optional[Path] = None
        if spec.out_path is not None:
            out_path = Path(spec.out_path)
        else:
            out_path = self.settings.out_path("hist_norm", date=spec.date)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(out_path), rn)
        return NormalizedItem(band=None, raster=rn, out_path=out_path)

    def normalize_many(self, band_uris: Mapping[S2BandName, str], spec: NormalizeManySpec) -> NormalizeManyResult:
        if not band_uris:
            raise ValueError("band_uris vacío")
        rasters: Dict[S2BandName, GeoRaster] = {}
        ref: Optional[GeoProfile] = None
        roi_crs = spec.roi_crs or self.settings.crs_out

        for b, uri in band_uris.items():
            r = self.reader.read(uri)
            r = self._maybe_clip(r, spec.roi_geojson, roi_crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            rasters[b] = self.preproc.normalize(r, spec.norm)

        assert ref is not None
        res_m = self._coerce_resolution_from_profile(ref)
        order = spec.order or self._stable_band_order(band_uris.keys())

        items: list[NormalizedItem] = []
        out_dir = Path(spec.out_dir) if spec.out_dir else self.settings.out_path("hist_norm", date=spec.date).parent
        if spec.write_individuals:
            out_dir.mkdir(parents=True, exist_ok=True)
            for b in order:
                rp = out_dir / f"{b}_norm.tif"
                self.writer.write(str(rp), rasters[b])
                items.append(NormalizedItem(band=b, raster=rasters[b], out_path=rp))
        else:
            for b in order:
                items.append(NormalizedItem(band=b, raster=rasters[b], out_path=None))

        stack_path: Optional[Path] = None
        if spec.write_stack:
            bs = BandSet(resolution_m=res_m, bands=rasters)
            stacked = bs.stack(order)
            stack_path = Path(spec.out_stack) if spec.out_stack else self.settings.out_path("stack", date=spec.date)
            stack_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(stack_path), stacked)

        return NormalizeManyResult(
            items=tuple(items),
            band_order=tuple(order),
            stack_path=stack_path,
            resolution_m=res_m,
        )

    def rgb_to_hsl(self, bandset /*: BandSet*/, order: Sequence[str] = ("B02","B03","B04")):
        roi_crs = spec.roi_crs or self.settings.crs_out
        r = self._maybe_clip(self.reader.read(r_uri), spec.roi_geojson, roi_crs)
        g = self._maybe_clip(self.reader.read(g_uri), spec.roi_geojson, roi_crs)
        b = self._maybe_clip(self.reader.read(b_uri), spec.roi_geojson, roi_crs)
        # Valida compatibilidad
        validate_profile_compat(r.profile, g.profile)
        validate_profile_compat(r.profile, b.profile)

        H, S, L = self.preproc.rgb_to_hsl(r, g, b)

        out_dir = Path(spec.out_dir) if spec.out_dir else self.settings.out_path("hist_norm", date=spec.date).parent
        H_item = NormalizedItem(band=None, raster=H, out_path=None)
        S_item = NormalizedItem(band=None, raster=S, out_path=None)
        L_item = NormalizedItem(band=None, raster=L, out_path=None)

        if spec.write_individuals:
            out_dir.mkdir(parents=True, exist_ok=True)
            H_item = NormalizedItem(band=None, raster=H, out_path=out_dir / "H.tif")
            S_item = NormalizedItem(band=None, raster=S, out_path=out_dir / "S.tif")
            L_item = NormalizedItem(band=None, raster=L, out_path=out_dir / "L.tif")
            self.writer.write(str(H_item.out_path), H)
            self.writer.write(str(S_item.out_path), S)
            self.writer.write(str(L_item.out_path), L)

        stack_path: Optional[Path] = None
        if spec.write_stack:
            rasters = {"H": H, "S": S, "L": L}
            # Usa BandSet para apilar en orden H,S,L
            bs = BandSet(resolution_m=self._coerce_resolution_from_profile(H.profile), bands=rasters)  # type: ignore[arg-type]
            stacked = bs.stack(("H", "S", "L"))  # type: ignore[arg-type]
            stack_path = Path(spec.out_stack) if spec.out_stack else self.settings.out_path("stack", date=spec.date)
            stack_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(stack_path), stacked)

        return RGB2HSLResult(H=H_item, S=S_item, L=L_item, stack_path=stack_path)


__all__ = [
    "PreprocessingService",
    "NormalizeSingleSpec",
    "NormalizeManySpec",
    "RGB2HSLSpec",
    "NormalizedItem",
    "NormalizeManyResult",
    "RGB2HSLResult",
]
