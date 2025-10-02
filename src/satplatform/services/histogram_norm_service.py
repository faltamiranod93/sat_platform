# src/satplatform/services/histogram_norm_service.py
from __future__ import annotations

"""
Histogram & Normalization Service (contracts-first, minimal deps)

Objetivo: proveer herramientas deterministas para inspección y
normalización basada en histogramas, y opcionalmente escribir salidas.

Funciones principales:
  • histogram() / histogram_many(): describe distribución (counts, edges,
    stats, percentiles) de una o varias bandas.
  • equalize(): ecualización de histograma (CDF) → [0,1] float32.
  • match_to_reference(): matching de histograma contra raster de
    referencia (CDF→CDF), conservando forma espacial.
  • percent_clip_normalize(): normalización 0..1 por percentiles
    (envolvente robusta), equivalente a NormalizeSpec('percent_clip').

Notas:
  - No realiza reproyecciones ni resampling. Si se requiere ROI, inyecta
    un ROIClipperPort y entrega GeoJSON + CRS.
  - Las escrituras usan Settings.out_path("hist_norm", date=YYYYMMDD)
    salvo que proporciones out_path explícito.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import Settings, get_settings
from ..contracts.core import S2BandName
from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, validate_profile_compat
from ..ports.preprocessing import NormalizeSpec
from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort
from ..ports.roi import ROIClipperPort, GeoJSON


# ----------------------
# DTOs
# ----------------------

@dataclass(frozen=True)
class HistSpec:
    bins: int = 256
    value_range: Optional[Tuple[float, float]] = None  # (min,max) si se desea fijar
    ignore_nodata: bool = True
    p_stats: Tuple[float, float] = (2.0, 98.0)  # percentiles rápidos a reportar

@dataclass(frozen=True)
class HistResult:
    counts: np.ndarray
    bin_edges: np.ndarray
    mean: float
    std: float
    p_low: float
    p_high: float

@dataclass(frozen=True)
class EqualizeSpec:
    date: str
    out_path: Optional[Path] = None
    clip_percentiles: Tuple[float, float] = (0.0, 100.0)  # 0,100 = sin clip previo
    write: bool = True

@dataclass(frozen=True)
class MatchSpec:
    date: str
    out_path: Optional[Path] = None
    clip_percentiles_src: Tuple[float, float] = (0.0, 100.0)
    clip_percentiles_ref: Tuple[float, float] = (0.0, 100.0)
    write: bool = True

@dataclass(frozen=True)
class PercentClipSpec:
    date: str
    norm: NormalizeSpec = NormalizeSpec(method="percent_clip", p_low=2.0, p_high=98.0)
    out_path: Optional[Path] = None
    write: bool = True

@dataclass(frozen=True)
class HistManyResult:
    per_band: Dict[S2BandName, HistResult]
    order: Tuple[S2BandName, ...]


# ----------------------
# Utilidades internas
# ----------------------

def _mask_data(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    if nodata is None:
        return arr.astype(np.float64, copy=False)
    m = arr.astype(np.float64)
    return m[np.isfinite(m) & (m != nodata)]


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(arr, p))


def _histogram_core(arr: np.ndarray, spec: HistSpec, nodata: Optional[float]) -> HistResult:
    data = _mask_data(arr, nodata) if spec.ignore_nodata else arr.astype(np.float64, copy=False)
    if data.size == 0:
        # Vacío: devuelve artefacto estable
        counts = np.zeros(spec.bins, dtype=np.int64)
        edges = np.linspace(0.0, 1.0, spec.bins + 1)
        return HistResult(counts, edges, 0.0, 0.0, 0.0, 0.0)
    vmin = np.nanmin(data) if spec.value_range is None else spec.value_range[0]
    vmax = np.nanmax(data) if spec.value_range is None else spec.value_range[1]
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    counts, edges = np.histogram(data, bins=spec.bins, range=(vmin, vmax))
    mean = float(np.nanmean(data))
    std = float(np.nanstd(data))
    p_low = _percentile(data, spec.p_stats[0])
    p_high = _percentile(data, spec.p_stats[1])
    return HistResult(counts, edges, mean, std, p_low, p_high)


def _cdf_from_hist(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(counts.astype(np.float64))
    if cdf[-1] <= 0:
        cdf = np.linspace(0, 1, len(cdf))
    else:
        cdf = cdf / cdf[-1]
    # valor representativo de cada bin = centro
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, cdf


def _apply_clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def _linmap(arr: np.ndarray, lo: float, hi: float, eps: float = 1e-6) -> np.ndarray:
    return (arr - lo) / (hi - lo + eps)


def _hist_equalize_core(arr: np.ndarray, nodata: Optional[float], pclip: Tuple[float, float]) -> np.ndarray:
    # Clip opcional para robustez
    data = arr.astype(np.float32, copy=True)
    valid_mask = np.ones_like(data, dtype=bool)
    if nodata is not None:
        valid_mask &= np.isfinite(data) & (data != nodata)
    vals = data[valid_mask]
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = np.nanpercentile(vals, pclip[0])
    hi = np.nanpercentile(vals, pclip[1])
    vals = _apply_clip(vals, lo, hi)
    counts, edges = np.histogram(vals, bins=256, range=(vals.min(), vals.max()))
    centers, cdf = _cdf_from_hist(counts, edges)
    # Interpola cada valor al CDF
    data_eq = data.copy()
    data_eq[valid_mask] = np.interp(data[valid_mask], centers, cdf)
    data_eq[~valid_mask] = 0.0
    return data_eq.astype(np.float32)


def _hist_match_core(src: np.ndarray, ref: np.ndarray, nodata_src: Optional[float], nodata_ref: Optional[float], psrc: Tuple[float, float], pref: Tuple[float, float]) -> np.ndarray:
    s = src.astype(np.float32, copy=True)
    r = ref.astype(np.float32, copy=False)
    ms = np.ones_like(s, dtype=bool)
    mr = np.ones_like(r, dtype=bool)
    if nodata_src is not None:
        ms &= np.isfinite(s) & (s != nodata_src)
    if nodata_ref is not None:
        mr &= np.isfinite(r) & (r != nodata_ref)
    if not np.any(ms) or not np.any(mr):
        return np.zeros_like(s, dtype=np.float32)
    s_vals = s[ms]
    r_vals = r[mr]
    s_lo = np.nanpercentile(s_vals, psrc[0]); s_hi = np.nanpercentile(s_vals, psrc[1])
    r_lo = np.nanpercentile(r_vals, pref[0]); r_hi = np.nanpercentile(r_vals, pref[1])
    s_vals = _apply_clip(s_vals, s_lo, s_hi)
    r_vals = _apply_clip(r_vals, r_lo, r_hi)
    s_counts, s_edges = np.histogram(s_vals, bins=512, range=(s_vals.min(), s_vals.max()))
    r_counts, r_edges = np.histogram(r_vals, bins=512, range=(r_vals.min(), r_vals.max()))
    s_centers, s_cdf = _cdf_from_hist(s_counts, s_edges)
    r_centers, r_cdf = _cdf_from_hist(r_counts, r_edges)
    # mapea: valor_src -> cdf_s -> valor_ref (inversa de cdf_ref)
    # Aproximamos inversa de CDF_ref con interpolación
    # Primero obtenemos función x_ref(cdf):
    x_ref = np.interp(s_cdf, r_cdf, r_centers)
    out = s.copy()
    out[ms] = np.interp(s[ms], s_centers, x_ref)
    out[~ms] = 0.0
    return out.astype(np.float32)


# ----------------------
# Servicio
# ----------------------

@dataclass
class HistogramNormService:
    reader: Optional[object] = None
    writer: Optional[object] = None
    clipper: Optional[ROIClipperPort] = None
    settings: Settings = field(default_factory=get_settings)

    # ------ Helpers generales ------
    @staticmethod
    def _coerce_resolution_from_profile(p: GeoProfile) -> int:
        px, py = p.pixel_size()
        gsd = min(abs(px), abs(py))
        return min((10, 20, 60), key=lambda r: abs(gsd - r))

    def _maybe_clip(self, r: GeoRaster, roi_g: Optional[GeoJSON], roi_crs: CRSRef) -> GeoRaster:
        if roi_g is None:
            return r
        if not self.clipper:
            raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
        return self.clipper.clip_raster(r, roi_g, roi_crs)

    # ------ API pública ------
    def histogram(self, uri: str, *, spec: HistSpec = HistSpec(), roi_geojson: Optional[GeoJSON] = None, roi_crs: Optional[CRSRef] = None) -> HistResult:
        r = self.reader.read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs or self.settings.crs_out)
        return _histogram_core(r.data, spec, r.profile.nodata)

    def histogram_many(self, band_uris: Mapping[S2BandName, str], *, spec: HistSpec = HistSpec(), roi_geojson: Optional[GeoJSON] = None, roi_crs: Optional[CRSRef] = None) -> HistManyResult:
        if not band_uris:
            raise ValueError("band_uris vacío")
        ref: Optional[GeoProfile] = None
        per: Dict[S2BandName, HistResult] = {}
        for b, uri in band_uris.items():
            r = self.reader.read(uri)
            r = self._maybe_clip(r, roi_geojson, roi_crs or self.settings.crs_out)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            per[b] = _histogram_core(r.data, spec, r.profile.nodata)
        order = self._stable_band_order(band_uris.keys())
        return HistManyResult(per_band=per, order=order)

    def equalize(self, uri: str, eq: EqualizeSpec, *, roi_geojson: Optional[GeoJSON] = None, roi_crs: Optional[CRSRef] = None) -> Tuple[GeoRaster, Optional[Path]]:
        r = self.reader.read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs or self.settings.crs_out)
        eq_arr = _hist_equalize_core(r.data, r.profile.nodata, eq.clip_percentiles)
        out = GeoRaster(eq_arr, r.profile._replace(dtype="float32")) if hasattr(r.profile, "_replace") else GeoRaster(eq_arr, r.profile)  # tolerant with namedtuple-like
        out_path: Optional[Path] = None
        if eq.write:
            out_path = Path(eq.out_path) if eq.out_path else self.settings.out_path("hist_norm", date=eq.date)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(out_path), out)
        return out, out_path

    def match_to_reference(self, src_uri: str, ref_uri: str, ms: MatchSpec, *, roi_geojson: Optional[GeoJSON] = None, roi_crs: Optional[CRSRef] = None) -> Tuple[GeoRaster, Optional[Path]]:
        src = self.reader.read(src_uri)
        ref = self.reader.read(ref_uri)
        # No obligamos compatibilidad geométrica; matching usa solo histograma de ref.
        src = self._maybe_clip(src, roi_geojson, roi_crs or self.settings.crs_out)
        matched = _hist_match_core(src.data, ref.data, src.profile.nodata, ref.profile.nodata, ms.clip_percentiles_src, ms.clip_percentiles_ref)
        out = GeoRaster(matched, src.profile._replace(dtype="float32")) if hasattr(src.profile, "_replace") else GeoRaster(matched, src.profile)
        out_path: Optional[Path] = None
        if ms.write:
            out_path = Path(ms.out_path) if ms.out_path else self.settings.out_path("hist_norm", date=ms.date)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(out_path), out)
        return out, out_path

    def percent_clip_normalize(self, uri: str, ps: PercentClipSpec, *, roi_geojson: Optional[GeoJSON] = None, roi_crs: Optional[CRSRef] = None) -> Tuple[GeoRaster, Optional[Path]]:
        r = self.reader.read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs or self.settings.crs_out)
        # Implementación directa (sin depender de PreprocessingPort)
        arr = r.data.astype(np.float32)
        valid = np.ones_like(arr, dtype=bool)
        if r.profile.nodata is not None:
            valid &= np.isfinite(arr) & (arr != r.profile.nodata)
        vals = arr[valid]
        if vals.size == 0:
            out_arr = np.zeros_like(arr, dtype=np.float32)
        else:
            lo = np.nanpercentile(vals, ps.norm.p_low)
            hi = np.nanpercentile(vals, ps.norm.p_high)
            vals = _apply_clip(vals, lo, hi)
            out_arr = np.zeros_like(arr, dtype=np.float32)
            out_arr[valid] = _linmap(arr[valid], lo, hi)
            out_arr[~valid] = 0.0
        out = GeoRaster(out_arr, r.profile._replace(dtype="float32")) if hasattr(r.profile, "_replace") else GeoRaster(out_arr, r.profile)
        out_path: Optional[Path] = None
        if ps.write:
            out_path = Path(ps.out_path) if ps.out_path else self.settings.out_path("hist_norm", date=ps.date)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(out_path), out)
        return out, out_path

    @staticmethod
    def hist_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        """
        Devuelve stats robustas para normalización (min, max o percentiles).
        El test 'contract' suele validar tipos y rangos.
        """
        if mask is not None:
            data = arr[mask]
        else:
            data = arr
        data = data[np.isfinite(data)]
        if data.size == 0:
            return {"p2": 0.0, "p98": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}

        p2, p98 = np.percentile(data, [2, 98])
        return {
            "p2": float(p2),
            "p98": float(p98),
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
        }

    @staticmethod
    def normalize_band(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """
        Normaliza a [0,1] recortando en [lo,hi], sin NaNs.
        """
        eps = 1e-12
        out = (arr - lo) / max(hi - lo, eps)
        out = np.clip(out, 0.0, 1.0)
        return out.astype("float32", copy=False)

    def normalize_bandset(self, bandset /*: BandSet*/, order: Sequence[str]) /*-> BandSet o GeoRaster*/:
        """
        Aplica normalize_band por banda según stats por banda (p2/p98).
        Debe devolver un objeto de dominio (no de adapter).
        """
        # ejemplo sin depender de reader/writer:
        # 1) extrae arrays en el orden solicitado
        # 2) calcula stats por banda
        # 3) normaliza y apila
        # 4) arma perfil y devuelve raster/bandset normalizado
        raise NotImplementedError

    # -------- orden estable --------
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


__all__ = [
    "HistogramNormService",
    "HistSpec",
    "HistResult",
    "EqualizeSpec",
    "MatchSpec",
    "PercentClipSpec",
    "HistManyResult",
]
