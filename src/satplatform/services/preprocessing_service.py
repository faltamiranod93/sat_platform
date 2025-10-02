# src/satplatform/services/preprocessing_service.py
from __future__ import annotations
"""
Servicio de preprocesamiento (histogram normalization, RGB->HSL, stack)
— contracts-first: sin Settings, sin cálculo de rutas.

Reglas:
- Nunca construye paths: si se entregan out_path/out_dir/out_stack se escribe; si no, NO escribe.
- No reproyecta ni resamplea. Si se pasa ROI, requiere ROIClipperPort.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

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
    # 'date' puede ser útil para el llamador; el servicio NO la usa para paths
    date: str                           # YYYYMMDD (informativo)
    out_path: Optional[Path] = None     # si None -> no escribe
    norm: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class NormalizeManySpec:
    date: str                           # informativo
    order: Optional[Tuple[S2BandName, ...]] = None
    write_individuals: bool = False
    out_dir: Optional[Path] = None      # si None -> no escribe individuales
    write_stack: bool = True
    out_stack: Optional[Path] = None    # si None -> no escribe stack
    norm: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class RGB2HSLSpec:
    date: str                           # informativo
    write_individuals: bool = True
    out_dir: Optional[Path] = None      # si None -> no escribe H/S/L
    write_stack: bool = False
    out_stack: Optional[Path] = None    # si None -> no escribe stack
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None

@dataclass(frozen=True)
class NormalizedItem:
    band: Optional[S2BandName]          # None para H/S/L u otros
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
    reader: Optional[RasterReaderPort] = None
    writer: Optional[RasterWriterPort] = None
    preproc: Optional[PreprocessingPort] = None
    clipper: Optional[ROIClipperPort] = None

    REQUIRED_RGB: tuple[S2BandName, S2BandName, S2BandName] = ("B02", "B03", "B04")

    # ---------- Helpers ----------
    @staticmethod
    def _coerce_resolution_from_profile(p: GeoProfile) -> int:
        px, py = p.pixel_size()
        gsd = min(abs(px), abs(py))
        # Pick the closest nominal S2 resolution
        return min((10, 20, 60), key=lambda r: abs(gsd - r))

    @staticmethod
    def _stable_band_order(keys: Iterable[S2BandName]) -> Tuple[S2BandName, ...]:
        def _key(k: str) -> Tuple[int, str]:
            try:
                n = int(k[1:3])
            except Exception:
                n = 99
            suf = k[3:] if len(k) > 3 else ""
            return (n, suf)
        return tuple(sorted(keys, key=_key))

    def _require_reader(self) -> RasterReaderPort:
        if self.reader is None:
            raise RuntimeError("RasterReaderPort no configurado")
        return self.reader

    def _require_writer(self) -> RasterWriterPort:
        if self.writer is None:
            raise RuntimeError("RasterWriterPort no configurado")
        return self.writer

    def _require_preproc(self) -> PreprocessingPort:
        if self.preproc is None:
            raise RuntimeError("PreprocessingPort no configurado")
        return self.preproc

    def _maybe_clip(self, r: GeoRaster, roi_g: Optional[GeoJSON], roi_crs: CRSRef) -> GeoRaster:
        if roi_g is None:
            return r
        if not self.clipper:
            raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
        return self.clipper.clip_raster(r, roi_g, roi_crs)
    
    def _nanpercentile(a: np.ndarray, p: float) -> float:
        try:
            return float(np.nanpercentile(a, p, method="linear"))
        except TypeError:
            return float(np.nanpercentile(a, p, interpolation="linear"))

    def _linmap01(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
        span = max(hi - lo, 1e-6)
        out = (a - lo) / span
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _rgb_to_hsl_arrays(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # r,g,b en [0,1], float32; devuelve H,S,L en [0,1]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        l = (maxc + minc) / 2.0
        d = maxc - minc

        s = np.zeros_like(l, dtype=np.float32)
        nonzero = d > 0
        s[nonzero] = d[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0) + 1e-6)

        h = np.zeros_like(l, dtype=np.float32)
        mask = nonzero
        # evitar división por cero
        rd = np.zeros_like(r); gd = np.zeros_like(g); bd = np.zeros_like(b)
        rd[mask] = (((maxc - r) / 6.0) + (d / 2.0))[mask] / (d[mask] + 1e-6)
        gd[mask] = (((maxc - g) / 6.0) + (d / 2.0))[mask] / (d[mask] + 1e-6)
        bd[mask] = (((maxc - b) / 6.0) + (d / 2.0))[mask] / (d[mask] + 1e-6)

        # ramas
        r_is_max = mask & (r == maxc)
        g_is_max = mask & (g == maxc)
        b_is_max = mask & (b == maxc)

        h[r_is_max] = (bd - gd)[r_is_max]
        h[g_is_max] = (1.0/3.0) + (rd - bd)[g_is_max]
        h[b_is_max] = (2.0/3.0) + (gd - rd)[b_is_max]

        # normaliza a [0,1]
        h = (h % 1.0).astype(np.float32, copy=False)
        return h, s.astype(np.float32, copy=False), l.astype(np.float32, copy=False)

    def _profile_float32_from(p: GeoProfile) -> GeoProfile:
        return GeoProfile(
            count=1, dtype="float32", width=p.width, height=p.height,
            transform=p.transform, crs=p.crs, nodata=np.nan)

    # ---------- Casos de uso ----------
    def normalize_single(self, uri: str, spec: NormalizeSingleSpec) -> NormalizedItem:
        reader = self._require_reader()
        pre = self._require_preproc()

        r = reader.read(uri)
        roi_crs = spec.roi_crs or r.profile.crs
        r = self._maybe_clip(r, spec.roi_geojson, roi_crs)
        rn = pre.normalize(r, spec.norm)

        out_path: Optional[Path] = None
        if spec.out_path is not None:
            out_path = Path(spec.out_path)
            writer = self._require_writer()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer.write(str(out_path), rn)

        return NormalizedItem(band=None, raster=rn, out_path=out_path)

    def normalize_many(self, band_uris: Mapping[S2BandName, str], spec: NormalizeManySpec) -> NormalizeManyResult:
        if not band_uris:
            raise ValueError("band_uris vacío")

        reader = self._require_reader()
        pre = self._require_preproc()

        rasters: Dict[S2BandName, GeoRaster] = {}
        ref: Optional[GeoProfile] = None
        roi_crs: Optional[CRSRef] = spec.roi_crs

        for b, uri in band_uris.items():
            r = reader.read(uri)
            # si no se provee roi_crs, clipea en el CRS del raster
            r = self._maybe_clip(r, spec.roi_geojson, roi_crs or r.profile.crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            rasters[b] = pre.normalize(r, spec.norm)

        assert ref is not None
        res_m = self._coerce_resolution_from_profile(ref)
        order = spec.order or self._stable_band_order(band_uris.keys())

        items: list[NormalizedItem] = []
        # Escribe individuales solo si se solicitó y hay out_dir
        if spec.write_individuals and spec.out_dir is not None:
            out_dir = Path(spec.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            writer = self._require_writer()
            for b in order:
                rp = out_dir / f"{b}_norm.tif"
                writer.write(str(rp), rasters[b])
                items.append(NormalizedItem(band=b, raster=rasters[b], out_path=rp))
        else:
            for b in order:
                items.append(NormalizedItem(band=b, raster=rasters[b], out_path=None))

        stack_path: Optional[Path] = None
        if spec.write_stack and spec.out_stack is not None:
            bs = BandSet(resolution_m=res_m, bands=rasters)
            stacked = bs.stack(order)
            stack_path = Path(spec.out_stack)
            stack_path.parent.mkdir(parents=True, exist_ok=True)
            writer = self._require_writer()
            writer.write(str(stack_path), stacked)

        return NormalizeManyResult(
            items=tuple(items),
            band_order=tuple(order),
            stack_path=stack_path,
            resolution_m=res_m,
        )

        def rgb_to_hsl(self, bandset: BandSet, clip: tuple[float,float]=(2.0,98.0), gamma: float=1.0) -> dict[str, GeoRaster]:
            required = ("B02","B03","B04")
            missing = [b for b in required if b not in bandset.bands]
            if missing:
                raise KeyError(f"Faltan bandas RGB: {missing}")

            # B04=R, B03=G, B02=B (Sentinel-2)
            r = bandset.bands["B04"].data.astype(np.float32, copy=False)
            g = bandset.bands["B03"].data.astype(np.float32, copy=False)
            b = bandset.bands["B02"].data.astype(np.float32, copy=False)

            # normaliza cada canal por percentiles (robusto) y aplica gamma si corresponde
            out_r = np.full_like(r, np.nan, dtype=np.float32)
            out_g = np.full_like(g, np.nan, dtype=np.float32)
            out_b = np.full_like(b, np.nan, dtype=np.float32)

            for src, dst in ((r,out_r),(g,out_g),(b,out_b)):
                vals = src[np.isfinite(src)].astype(np.float64, copy=False)
                if vals.size:
                    lo = _nanpercentile(vals, clip[0]); hi = _nanpercentile(vals, clip[1])
                    dst[:] = _linmap01(src.astype(np.float64, copy=False), lo, hi).astype(np.float32, copy=False)

            if gamma and gamma != 1.0:
                invg = 1.0/max(gamma, 1e-6)
                out_r = np.power(out_r, invg, dtype=np.float32, where=np.isfinite(out_r))
                out_g = np.power(out_g, invg, dtype=np.float32, where=np.isfinite(out_g))
                out_b = np.power(out_b, invg, dtype=np.float32, where=np.isfinite(out_b))

            H,S,L = _rgb_to_hsl_arrays(out_r, out_g, out_b)

            prof = _profile_float32_from(list(bandset.bands.values())[0].profile)
            return {
                "H": GeoRaster(H, prof),
                "S": GeoRaster(S, prof),
                "L": GeoRaster(L, prof),
            }

    # ---------- utilitario interno ----------
    @staticmethod
    def _stack_rasters(rasters: Sequence[GeoRaster]) -> GeoRaster:
        if not rasters:
            raise ValueError("No hay rasters para apilar")
        ref = rasters[0].profile
        for r in rasters[1:]:
            validate_profile_compat(ref, r.profile)
        arrs = [r.data for r in rasters]
        d0 = arrs[0].dtype
        if any(a.dtype != d0 for a in arrs[1:]):
            raise ValueError("Los dtypes de datos no coinciden entre capas a apilar")
        data = np.stack(arrs, axis=0).astype(d0, copy=False)
        profile = GeoProfile(
            count=len(arrs),
            dtype=ref.dtype,
            width=ref.width,
            height=ref.height,
            transform=ref.transform,
            crs=ref.crs,
            nodata=ref.nodata,
        )
        return GeoRaster(data=data, profile=profile)


__all__ = [
    "PreprocessingService",
    "NormalizeSingleSpec",
    "NormalizeManySpec",
    "RGB2HSLSpec",
    "NormalizedItem",
    "NormalizeManyResult",
    "RGB2HSLResult",
]
