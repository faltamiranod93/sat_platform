# src/satplatform/services/preprocessing_service.py
from __future__ import annotations

"""
Servicio de preprocesamiento (histogram normalization, RGB→HSL, stack).
Contracts-first: sin Settings, sin cálculo de rutas.

SPRINT 1 — Bugs corregidos:
  - _nanpercentile, _linmap01, _rgb_to_hsl_arrays y _profile_float32_from
    estaban declaradas DENTRO de la clase sin self ni @staticmethod.
    Python las trataba como métodos de instancia con firma incorrecta, y
    al llamarlas como funciones libres dentro de otros métodos lanzaba
    NameError porque no existían en el scope global.

    Solución: se mueven AL MÓDULO como funciones privadas (prefijo _).
    Esto es correcto porque son utilidades numéricas puras sin estado,
    no comportamiento de la clase.

Reglas que se mantienen:
  - Nunca construye paths: si se entrega out_path/out_dir/out_stack se
    escribe; si no, NO escribe.
  - No reproyecta ni resamplea.
  - Si se pasa ROI, requiere ROIClipperPort.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts.core import S2BandName
from ..contracts.geo import (
    GeoProfile,
    GeoRaster,
    CRSRef,
    validate_profile_compat,
)
from ..contracts.products import BandSet
from ..ports.preprocessing import NormalizeSpec, PreprocessingPort
from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort
from ..ports.roi import GeoJSON, ROIClipperPort

# ---------------------------------------------------------------------------
# DTOs de entrada / salida
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizeSingleSpec:
    date:        str                   # YYYYMMDD — informativo, no se usa para paths
    out_path:    Optional[Path] = None # None → no escribe
    norm:        NormalizeSpec  = NormalizeSpec()
    roi_geojson: Optional[GeoJSON]  = None
    roi_crs:     Optional[CRSRef]   = None


@dataclass(frozen=True)
class NormalizeManySpec:
    date:              str
    order:             Optional[Tuple[S2BandName, ...]] = None
    write_individuals: bool            = False
    out_dir:           Optional[Path]  = None  # None → no escribe individuales
    write_stack:       bool            = True
    out_stack:         Optional[Path]  = None  # None → no escribe stack
    norm:              NormalizeSpec   = NormalizeSpec()
    roi_geojson:       Optional[GeoJSON]  = None
    roi_crs:           Optional[CRSRef]   = None


@dataclass(frozen=True)
class RGB2HSLSpec:
    date:        str
    write_individuals: bool           = True
    out_dir:     Optional[Path]       = None
    write_stack: bool                 = False
    out_stack:   Optional[Path]       = None
    roi_geojson: Optional[GeoJSON]    = None
    roi_crs:     Optional[CRSRef]     = None


@dataclass(frozen=True)
class NormalizedItem:
    band:     Optional[S2BandName]  # None para H/S/L
    raster:   GeoRaster
    out_path: Optional[Path]


@dataclass(frozen=True)
class NormalizeManyResult:
    items:       Tuple[NormalizedItem, ...]
    band_order:  Tuple[S2BandName, ...]
    stack_path:  Optional[Path]
    resolution_m: int


@dataclass(frozen=True)
class RGB2HSLResult:
    H:          NormalizedItem
    S:          NormalizedItem
    L:          NormalizedItem
    stack_path: Optional[Path]


# ---------------------------------------------------------------------------
# Utilidades numéricas puras — NIVEL MÓDULO (no dentro de la clase)
#
# ANTES estaban declaradas dentro de PreprocessingService sin self ni
# @staticmethod, lo que causaba NameError al llamarlas desde otros métodos.
# ---------------------------------------------------------------------------

def _nanpercentile(a: np.ndarray, p: float) -> float:
    """Percentil robusto compatible con numpy < 1.22 y >= 1.22."""
    try:
        return float(np.nanpercentile(a, p, method="linear"))   # numpy >= 1.22
    except TypeError:
        return float(np.nanpercentile(a, p, interpolation="linear"))  # compat


def _linmap01(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mapea linealmente a [0, 1] recortando en [lo, hi]."""
    span = max(hi - lo, 1e-6)
    out = (a - lo) / span
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _rgb_to_hsl_arrays(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte arrays RGB float32 en [0,1] a H, S, L en [0,1].
    Vectorizado — sin bucles Python.
    """
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    l    = (maxc + minc) / 2.0
    d    = maxc - minc

    s    = np.zeros_like(l, dtype=np.float32)
    nonz = d > 0
    s[nonz] = d[nonz] / (1.0 - np.abs(2.0 * l[nonz] - 1.0) + 1e-6)

    h = np.zeros_like(l, dtype=np.float32)

    # Deltas auxiliares (solo donde d > 0)
    rd = np.zeros_like(r)
    gd = np.zeros_like(g)
    bd = np.zeros_like(b)
    m  = nonz
    rd[m] = (((maxc - r) / 6.0) + (d / 2.0))[m] / (d[m] + 1e-6)
    gd[m] = (((maxc - g) / 6.0) + (d / 2.0))[m] / (d[m] + 1e-6)
    bd[m] = (((maxc - b) / 6.0) + (d / 2.0))[m] / (d[m] + 1e-6)

    r_max = m & (r == maxc)
    g_max = m & (g == maxc)
    b_max = m & (b == maxc)
    h[r_max] = (bd - gd)[r_max]
    h[g_max] = (1.0 / 3.0) + (rd - bd)[g_max]
    h[b_max] = (2.0 / 3.0) + (gd - rd)[b_max]

    h = (h % 1.0).astype(np.float32, copy=False)
    return h, s.astype(np.float32, copy=False), l.astype(np.float32, copy=False)


def _profile_float32_from(p: GeoProfile) -> GeoProfile:
    """Devuelve un perfil float32 con count=1, preservando geometría y CRS."""
    return GeoProfile(
        count=1,
        dtype="float32",
        width=p.width,
        height=p.height,
        transform=p.transform,
        crs=p.crs,
        nodata=float("nan"),
    )


def _stable_band_order(keys: Iterable[S2BandName]) -> Tuple[S2BandName, ...]:
    """Orden reproducible: B02, B03, …, B12, B8A."""
    def _key(k: str) -> tuple[int, str]:
        try:
            n = int(k[1:3])
        except (ValueError, IndexError):
            n = 99
        suf = k[3:] if len(k) > 3 else ""
        return (n, suf)
    return tuple(sorted(keys, key=_key))


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingService:
    """
    Casos de uso de preprocesamiento:
      - normalize_single  — normaliza un raster individual
      - normalize_many    — normaliza N bandas y opcionalmente stackea
      - rgb_to_hsl        — convierte BandSet RGB a rasters H, S, L

    Todos los puertos son opcionales al construir; se validan al usar.
    """

    reader:  Optional[RasterReaderPort]  = None
    writer:  Optional[RasterWriterPort]  = None
    preproc: Optional[PreprocessingPort] = None
    clipper: Optional[ROIClipperPort]    = None

    # ── Helpers de puertos ───────────────────────────────────────────────────

    def _require_reader(self) -> RasterReaderPort:
        if self.reader is None:
            raise RuntimeError("RasterReaderPort no configurado en PreprocessingService")
        return self.reader

    def _require_writer(self) -> RasterWriterPort:
        if self.writer is None:
            raise RuntimeError("RasterWriterPort no configurado en PreprocessingService")
        return self.writer

    def _require_preproc(self) -> PreprocessingPort:
        if self.preproc is None:
            raise RuntimeError("PreprocessingPort no configurado en PreprocessingService")
        return self.preproc

    def _maybe_clip(
        self,
        r:       GeoRaster,
        roi_g:   Optional[GeoJSON],
        roi_crs: Optional[CRSRef],
    ) -> GeoRaster:
        if roi_g is None:
            return r
        if self.clipper is None:
            raise RuntimeError(
                "Se proporcionó ROI pero no hay ROIClipperPort configurado"
            )
        return self.clipper.clip_raster(r, roi_g, roi_crs or r.profile.crs)

    # ── Helpers geométricos (@staticmethod — sin estado de instancia) ────────

    @staticmethod
    def _coerce_resolution_from_profile(p: GeoProfile) -> int:
        px, py = p.pixel_size()
        gsd = min(abs(px), abs(py))
        return min((10, 20, 60), key=lambda r: abs(gsd - r))

    # ── Casos de uso ─────────────────────────────────────────────────────────

    def normalize_single(self, uri: str, spec: NormalizeSingleSpec) -> NormalizedItem:
        """Lee un raster, lo recorta si hay ROI, lo normaliza y lo escribe si hay path."""
        reader  = self._require_reader()
        preproc = self._require_preproc()

        r       = reader.read(uri)
        roi_crs = spec.roi_crs or r.profile.crs
        r       = self._maybe_clip(r, spec.roi_geojson, roi_crs)
        rn      = preproc.normalize(r, spec.norm)

        out_path: Optional[Path] = None
        if spec.out_path is not None:
            out_path = Path(spec.out_path)
            writer   = self._require_writer()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer.write(str(out_path), rn)

        return NormalizedItem(band=None, raster=rn, out_path=out_path)

    def normalize_many(
        self,
        band_uris: Mapping[S2BandName, str],
        spec:      NormalizeManySpec,
    ) -> NormalizeManyResult:
        """Normaliza N bandas; opcionalmente escribe individuales y/o stack."""
        if not band_uris:
            raise ValueError("band_uris vacío")

        reader  = self._require_reader()
        preproc = self._require_preproc()

        rasters: Dict[S2BandName, GeoRaster] = {}
        ref:     Optional[GeoProfile]        = None
        roi_crs = spec.roi_crs

        for b, uri in band_uris.items():
            r = reader.read(uri)
            r = self._maybe_clip(r, spec.roi_geojson, roi_crs or r.profile.crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            rasters[b] = preproc.normalize(r, spec.norm)

        assert ref is not None
        res_m = self._coerce_resolution_from_profile(ref)
        order = spec.order or _stable_band_order(band_uris.keys())

        items: list[NormalizedItem] = []

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
            bs         = BandSet(resolution_m=res_m, bands=rasters)
            stacked    = bs.stack(order)
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

    def rgb_to_hsl(
        self,
        bandset: BandSet,
        clip:    tuple[float, float] = (2.0, 98.0),
        gamma:   float               = 1.0,
    ) -> dict[str, GeoRaster]:
        """
        Convierte BandSet RGB (B04=R, B03=G, B02=B) a rasters H, S, L.

        Normaliza cada canal por percentiles antes de la conversión
        para que los valores queden en [0, 1].  Aplica corrección gamma
        si gamma != 1.0.
        """
        required = ("B02", "B03", "B04")
        missing  = [b for b in required if b not in bandset.bands]
        if missing:
            raise KeyError(f"Faltan bandas RGB en BandSet: {missing}")

        # B04=R, B03=G, B02=B  (convención Sentinel-2 color natural)
        r_raw = bandset.bands["B04"].data.astype(np.float32, copy=False)
        g_raw = bandset.bands["B03"].data.astype(np.float32, copy=False)
        b_raw = bandset.bands["B02"].data.astype(np.float32, copy=False)

        out_r = np.full_like(r_raw, np.nan, dtype=np.float32)
        out_g = np.full_like(g_raw, np.nan, dtype=np.float32)
        out_b = np.full_like(b_raw, np.nan, dtype=np.float32)

        # Normaliza cada canal por sus propios percentiles (robusto a outliers)
        for src, dst in ((r_raw, out_r), (g_raw, out_g), (b_raw, out_b)):
            vals = src[np.isfinite(src)].astype(np.float64, copy=False)
            if vals.size:
                lo = _nanpercentile(vals, clip[0])
                hi = _nanpercentile(vals, clip[1])
                dst[:] = _linmap01(
                    src.astype(np.float64, copy=False), lo, hi
                ).astype(np.float32, copy=False)

        # Corrección gamma opcional
        if gamma and gamma != 1.0:
            inv_g = 1.0 / max(gamma, 1e-6)
            fin_r = np.isfinite(out_r)
            fin_g = np.isfinite(out_g)
            fin_b = np.isfinite(out_b)
            out_r[fin_r] = np.power(out_r[fin_r], inv_g)
            out_g[fin_g] = np.power(out_g[fin_g], inv_g)
            out_b[fin_b] = np.power(out_b[fin_b], inv_g)

        # Llamada a la función de módulo (no a self)
        H, S, L = _rgb_to_hsl_arrays(out_r, out_g, out_b)

        prof = _profile_float32_from(bandset.bands["B04"].profile)
        return {
            "H": GeoRaster(H, prof),
            "S": GeoRaster(S, prof),
            "L": GeoRaster(L, prof),
        }

    # ── Utilitario para stacking interno ─────────────────────────────────────

    @staticmethod
    def _stack_rasters(rasters: Sequence[GeoRaster]) -> GeoRaster:
        """Apila rasters en un GeoRaster multibanda (F, H, W)."""
        if not rasters:
            raise ValueError("No hay rasters para apilar")
        ref = rasters[0].profile
        for r in rasters[1:]:
            validate_profile_compat(ref, r.profile)
        arrs = [r.data for r in rasters]
        d0   = arrs[0].dtype
        if any(a.dtype != d0 for a in arrs[1:]):
            raise ValueError("Los dtypes no coinciden entre capas a apilar")
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
