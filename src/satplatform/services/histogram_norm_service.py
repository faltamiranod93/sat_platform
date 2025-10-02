# src/satplatform/services/histogram_norm_service.py
from __future__ import annotations

"""
Histogram & Normalization Service (contracts-first, minimal deps)

Objetivo: herramientas deterministas para inspección y normalización
basada en histogramas. Sin Settings ni cálculo de rutas; solo escribe
si se proveen paths explícitos en los specs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

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
# Utilidades internas numéricas (robustas con NaN)
# ----------------------

def _nanpercentile(a: np.ndarray, p: float) -> float:
    try:
        return float(np.nanpercentile(a, p, method="linear"))  # numpy >= 1.22
    except TypeError:
        return float(np.nanpercentile(a, p, interpolation="linear"))  # compat

def _mask_valid(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Devuelve SOLO los valores válidos como vector float64 (para cómputo)."""
    data = arr.astype(np.float64, copy=False)
    m = np.isfinite(data)
    if nodata is not None:
        m &= data != nodata
    return data[m]

def _histogram_core(arr: np.ndarray, spec: HistSpec, nodata: Optional[float]) -> HistResult:
    data = _mask_valid(arr, nodata) if spec.ignore_nodata else arr.astype(np.float64, copy=False)
    if data.size == 0:
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
    p_low = _nanpercentile(data, spec.p_stats[0])
    p_high = _nanpercentile(data, spec.p_stats[1])
    return HistResult(counts, edges, mean, std, p_low, p_high)

def _cdf_from_hist(counts: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(counts.astype(np.float64))
    if cdf[-1] <= 0:
        cdf = np.linspace(0.0, 1.0, len(cdf))
    else:
        cdf = cdf / cdf[-1]
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, cdf

def _apply_clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)

def _linmap_to_01(arr: np.ndarray, lo: float, hi: float, eps: float = 1e-6) -> np.ndarray:
    span = max(hi - lo, eps)
    out = (arr - lo) / span
    np.clip(out, 0.0, 1.0, out=out)
    return out

def _profile_float32_from(p: GeoProfile, *, count: int | None = None) -> GeoProfile:
    """Construye un perfil float32, manteniendo geom/CRS; nodata se expone como NaN."""
    return GeoProfile(
        count=count if count is not None else p.count,
        dtype="float32",
        width=p.width,
        height=p.height,
        transform=p.transform,
        crs=p.crs,
        nodata=np.nan,
    )

def _hist_equalize_core(arr: np.ndarray, nodata: Optional[float], pclip: Tuple[float, float]) -> np.ndarray:
    data = arr.astype(np.float32, copy=False)
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    if not np.any(valid):
        out = np.full_like(data, np.nan, dtype=np.float32)
        return out

    vals = data[valid].astype(np.float64, copy=False)
    lo = _nanpercentile(vals, pclip[0])
    hi = _nanpercentile(vals, pclip[1])
    vals = _apply_clip(vals, lo, hi)

    if not np.isfinite(vals).any() or np.isclose(vals.max(), vals.min()):
        # Degenerado: todo igual → mapa constante 0.5
        out = np.full_like(data, np.nan, dtype=np.float32)
        out[valid] = 0.5
        return out

    counts, edges = np.histogram(vals, bins=256, range=(vals.min(), vals.max()))
    centers, cdf = _cdf_from_hist(counts, edges)

    out = np.full_like(data, np.nan, dtype=np.float32)
    out[valid] = np.interp(data[valid], centers, cdf).astype(np.float32, copy=False)
    np.clip(out, 0.0, 1.0, out=out)
    return out

def _hist_match_core(
    src: np.ndarray,
    ref: np.ndarray,
    nodata_src: Optional[float],
    nodata_ref: Optional[float],
    psrc: Tuple[float, float],
    pref: Tuple[float, float],
) -> np.ndarray:
    s = src.astype(np.float32, copy=False)
    r = ref.astype(np.float32, copy=False)
    ms = np.isfinite(s)
    mr = np.isfinite(r)
    if nodata_src is not None:
        ms &= s != nodata_src
    if nodata_ref is not None:
        mr &= r != nodata_ref

    out = np.full_like(s, np.nan, dtype=np.float32)
    if not np.any(ms) or not np.any(mr):
        return out

    s_vals = s[ms].astype(np.float64, copy=False)
    r_vals = r[mr].astype(np.float64, copy=False)

    s_lo = _nanpercentile(s_vals, psrc[0]); s_hi = _nanpercentile(s_vals, psrc[1])
    r_lo = _nanpercentile(r_vals, pref[0]); r_hi = _nanpercentile(r_vals, pref[1])

    s_vals = _apply_clip(s_vals, s_lo, s_hi)
    r_vals = _apply_clip(r_vals, r_lo, r_hi)

    if np.isclose(s_vals.max(), s_vals.min()) or np.isclose(r_vals.max(), r_vals.min()):
        out[ms] = s_vals.mean(dtype=np.float64).astype(np.float32) if s_vals.size else np.nan
        return out

    s_counts, s_edges = np.histogram(s_vals, bins=512, range=(s_vals.min(), s_vals.max()))
    r_counts, r_edges = np.histogram(r_vals, bins=512, range=(r_vals.min(), r_vals.max()))
    s_centers, s_cdf = _cdf_from_hist(s_counts, s_edges)
    r_centers, r_cdf = _cdf_from_hist(r_counts, r_edges)

    # valor_src -> cdf_s -> valor_ref (inversa aprox de cdf_ref)
    x_ref = np.interp(s_cdf, r_cdf, r_centers)  # x(cdf)
    out[ms] = np.interp(s[ms], s_centers, x_ref).astype(np.float32, copy=False)
    return out


# ----------------------
# Servicio
# ----------------------

@dataclass
class HistogramNormService:
    reader: Optional[RasterReaderPort] = None
    writer: Optional[RasterWriterPort] = None
    clipper: Optional[ROIClipperPort] = None

    # ------ Helpers de puertos ------
    def _require_reader(self) -> RasterReaderPort:
        if self.reader is None:
            raise RuntimeError("RasterReaderPort no configurado")
        return self.reader

    def _require_writer(self) -> RasterWriterPort:
        if self.writer is None:
            raise RuntimeError("RasterWriterPort no configurado")
        return self.writer

    def _maybe_clip(self, r: GeoRaster, roi_g: Optional[GeoJSON], roi_crs: Optional[CRSRef]) -> GeoRaster:
        if roi_g is None:
            return r
        if not self.clipper:
            raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
        # si roi_crs es None, clipea en el CRS del raster
        return self.clipper.clip_raster(r, roi_g, roi_crs or r.profile.crs)

    # ------ API pública ------
    def histogram(
        self,
        uri: str,
        *,
        spec: HistSpec = HistSpec(),
        roi_geojson: Optional[GeoJSON] = None,
        roi_crs: Optional[CRSRef] = None,
    ) -> HistResult:
        r = self._require_reader().read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs)
        return _histogram_core(r.data, spec, r.profile.nodata)

    def histogram_many(
        self,
        band_uris: Mapping[S2BandName, str],
        *,
        spec: HistSpec = HistSpec(),
        roi_geojson: Optional[GeoJSON] = None,
        roi_crs: Optional[CRSRef] = None,
    ) -> HistManyResult:
        if not band_uris:
            raise ValueError("band_uris vacío")
        ref: Optional[GeoProfile] = None
        per: Dict[S2BandName, HistResult] = {}
        reader = self._require_reader()
        for b, uri in band_uris.items():
            r = reader.read(uri)
            r = self._maybe_clip(r, roi_geojson, roi_crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            per[b] = _histogram_core(r.data, spec, r.profile.nodata)
        order = self._stable_band_order(band_uris.keys())
        return HistManyResult(per_band=per, order=order)

    def equalize(
        self,
        uri: str,
        eq: EqualizeSpec,
        *,
        roi_geojson: Optional[GeoJSON] = None,
        roi_crs: Optional[CRSRef] = None,
    ) -> Tuple[GeoRaster, Optional[Path]]:
        r = self._require_reader().read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs)
        eq_arr = _hist_equalize_core(r.data, r.profile.nodata, eq.clip_percentiles).astype(np.float32, copy=False)
        out = GeoRaster(eq_arr, _profile_float32_from(r.profile, count=1))
        out_path: Optional[Path] = None
        if eq.write and eq.out_path is not None:
            out_path = Path(eq.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._require_writer().write(str(out_path), out)
        return out, out_path

    def match_to_reference(
        self,
        src_uri: str,
        ref_uri: str,
        ms: MatchSpec,
        *,
        roi_geojson: Optional[GeoJSON] = None,
        roi_crs: Optional[CRSRef] = None,
    ) -> Tuple[GeoRaster, Optional[Path]]:
        reader = self._require_reader()
        src = reader.read(src_uri)
        ref = reader.read(ref_uri)
        src = self._maybe_clip(src, roi_geojson, roi_crs)
        # matching usa histograma de ref; no exige compat geométrica
        matched = _hist_match_core(
            src.data, ref.data, src.profile.nodata, ref.profile.nodata, ms.clip_percentiles_src, ms.clip_percentiles_ref
        )
        out = GeoRaster(matched.astype(np.float32, copy=False), _profile_float32_from(src.profile, count=1))
        out_path: Optional[Path] = None
        if ms.write and ms.out_path is not None:
            out_path = Path(ms.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._require_writer().write(str(out_path), out)
        return out, out_path

    def percent_clip_normalize(
        self,
        uri: str,
        ps: PercentClipSpec,
        *,
        roi_geojson: Optional[GeoJSON] = None,
        roi_crs: Optional[CRSRef] = None,
    ) -> Tuple[GeoRaster, Optional[Path]]:
        r = self._require_reader().read(uri)
        r = self._maybe_clip(r, roi_geojson, roi_crs)

        arr = r.data.astype(np.float32, copy=False)
        valid = np.isfinite(arr)
        if r.profile.nodata is not None:
            valid &= arr != r.profile.nodata

        out_arr = np.full_like(arr, np.nan, dtype=np.float32)
        if np.any(valid):
            vals = arr[valid].astype(np.float64, copy=False)
            lo = _nanpercentile(vals, ps.norm.p_low)
            hi = _nanpercentile(vals, ps.norm.p_high)
            mapped = _linmap_to_01(arr[valid].astype(np.float64, copy=False), lo, hi).astype(np.float32, copy=False)
            np.clip(mapped, 0.0, 1.0, out=mapped)
            out_arr[valid] = mapped

        out = GeoRaster(out_arr, _profile_float32_from(r.profile, count=1))
        out_path: Optional[Path] = None
        if ps.write and ps.out_path is not None:
            out_path = Path(ps.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._require_writer().write(str(out_path), out)
        return out, out_path

    @staticmethod
    def hist_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        """Stats robustas (usa nanpercentile/mean)."""
        data = arr if mask is None else arr[mask]
        data = data[np.isfinite(data)]
        if data.size == 0:
            return {"p2": 0.0, "p98": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
        p2 = _nanpercentile(data, 2.0)
        p98 = _nanpercentile(data, 98.0)
        return {"p2": float(p2), "p98": float(p98), "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)), "mean": float(np.nanmean(data))}

    @staticmethod
    def normalize_band(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """
        Normaliza a [0,1] recortando en [lo,hi], preservando NaNs.
        Devuelve float32.
        """
        out = ((arr.astype(np.float64, copy=False) - lo) / max(hi - lo, 1e-6)).astype(np.float32, copy=False)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def normalize_bandset(self, bandset: "BandSet", order: Sequence[S2BandName]) -> "BandSet":
        """
        Normaliza cada banda de un BandSet a [0,1] usando percentiles (p2/p98).
        - Devuelve un BandSet nuevo con float32 y NaN en inválidos (no se escribe).
        - Requiere que todas las bandas de 'order' existan; el resto también se normaliza.
        """
        # valida que el order exista en el bandset
        missing = [b for b in order if b not in bandset.bands]
        if missing:
            raise KeyError(f"Faltan bandas en BandSet: {missing}")

        out_map: dict[S2BandName, GeoRaster] = {}

        for name, ras in bandset.bands.items():
            arr = ras.data.astype(np.float32, copy=False)
            valid = np.isfinite(arr)
            if ras.profile.nodata is not None:
                valid &= arr != ras.profile.nodata

            # salida con NaN por defecto
            out_arr = np.full_like(arr, np.nan, dtype=np.float32)

            if np.any(valid):
                vals = arr[valid].astype(np.float64, copy=False)
                lo = _nanpercentile(vals, 2.0)
                hi = _nanpercentile(vals, 98.0)
                mapped = _linmap_to_01(
                    arr[valid].astype(np.float64, copy=False), lo, hi
                ).astype(np.float32, copy=False)
                # asegura [0,1] con tolerancia numérica
                np.clip(mapped, 0.0, 1.0, out=mapped)
                out_arr[valid] = mapped

            out_map[name] = GeoRaster(out_arr, _profile_float32_from(ras.profile, count=1))

        # construye nuevo BandSet inmutable con mismas props geométricas
        return BandSet(resolution_m=bandset.resolution_m, bands=out_map)
        #raise NotImplementedError

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
