# src/satplatform/services/classmap_service.py
from __future__ import annotations

"""
Servicio de construcción de CLASSMAP (inspiración "v5"), contracts-first.
Pipeline determinista:
  LOAD → ALIGN → (ROI) → (PRE) → INFER → POST (stats/palette) → EXPORT

No asume backends concretos: todo va vía *ports*.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import Settings, get_settings
from ..contracts.core import ClassLabel, S2BandName, SceneId, Stage
from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, validate_profile_compat
from ..contracts.products import BandSet

from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort
from ..ports.roi import ROIClipperPort, GeoJSON
from ..ports.preprocessing import PreprocessingPort, NormalizeSpec
from ..ports.pixel_class import PixelClassifierPort
from ..ports.class_map import ClassMapPort, ClassMap
from ..ports.exporters import QuicklookExporterPort, ReportExporterPort, QuicklookSpec


# ----------------------
# Especificaciones / DTOs
# ----------------------

@dataclass(frozen=True)
class ClassMapInputs:
    band_uris: Dict[S2BandName, str]
    scene: Optional[SceneId] = None
    classes: Optional[Sequence[ClassLabel]] = None


@dataclass(frozen=True)
class ClassMapSpec:
    date: str  # YYYYMMDD para rutas
    order: Optional[Tuple[S2BandName, ...]] = None
    normalize: bool = False
    norm_spec: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None
    out_tif: Optional[Path] = None
    out_png: Optional[Path] = None
    make_report: bool = False
    report_template: str = "classmap_summary"


@dataclass(frozen=True)
class ClassMapResult:
    labels_tif: Path
    quicklook_png: Optional[Path]
    counts: Mapping[int, int]
    percents: Mapping[int, float]
    palette: Mapping[int, Tuple[int, int, int]]
    band_order: Tuple[S2BandName, ...]
    resolution_m: int


# ----------------------
# Servicio
# ----------------------

@dataclass
class ClassMapService:
    reader: RasterReaderPort
    writer: RasterWriterPort
    classifier: PixelClassifierPort
    cmapper: ClassMapPort
    preproc: Optional[PreprocessingPort] = None
    clipper: Optional[ROIClipperPort] = None
    ql_exporter: Optional[QuicklookExporterPort] = None
    reporter: Optional[ReportExporterPort] = None
    settings: Settings = field(default_factory=get_settings)

    # --------- API principal ---------
    def run(self, inputs: ClassMapInputs, spec: ClassMapSpec) -> ClassMapResult:
        # 1) LOAD + ALIGN (+ ROI) + (PRE)
        rasters, ref_profile = self._load_align(inputs.band_uris, spec)
        if spec.normalize:
            if not self.preproc:
                raise RuntimeError("normalize=True requiere un PreprocessingPort configurado")
            rasters = {k: self.preproc.normalize(v, spec.norm_spec) for k, v in rasters.items()}  # type: ignore[union-attr]

        # 2) BANDSET determinista
        band_order = spec.order or tuple(self._stable_band_order(inputs.band_uris.keys()))
        res_m = self._coerce_resolution_from_profile(ref_profile)
        bandset = BandSet(resolution_m=res_m, bands=rasters)

        # 3) INFER
        labels = self.classifier.predict(bandset)

        # 4) POST → ClassMap (counts / palette)
        classes = tuple(inputs.classes or self.classifier.classes())
        cmap = self.cmapper.from_labels(labels, classes)
        perc = self._to_percents(cmap.counts, total=labels.data.size)

        # 5) EXPORT (TIFF + quicklook opcional + reporte opcional)
        out_tif, out_png = self._export(labels, cmap, spec)

        return ClassMapResult(
            labels_tif=out_tif,
            quicklook_png=out_png,
            counts=cmap.counts,
            percents=perc,
            palette=cmap.palette,
            band_order=band_order,
            resolution_m=res_m,
        )

    # --------- Fases internas ---------
    def _load_align(self, band_uris: Mapping[S2BandName, str], spec: ClassMapSpec) -> Tuple[Dict[S2BandName, GeoRaster], GeoProfile]:
        if not band_uris:
            raise ValueError("band_uris vacío")
        rasters: Dict[S2BandName, GeoRaster] = {}
        ref: Optional[GeoProfile] = None
        roi_g = spec.roi_geojson
        roi_crs = spec.roi_crs or self.settings.crs_out

        for b, uri in band_uris.items():
            r = self.reader.read(uri)
            if roi_g is not None:
                if not self.clipper:
                    raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
                r = self.clipper.clip_raster(r, roi_g, roi_crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            rasters[b] = r
        assert ref is not None
        return rasters, ref

    @staticmethod
    def _stable_band_order(keys: Iterable[S2BandName]) -> Tuple[S2BandName, ...]:
        # Orden reproducible: orden numérico por índice de banda (B02, B03, ... B12, B8A, etc.)
        def _key(k: str) -> Tuple[int, str]:
            # B8A como 8 + sufijo A para que quede después de B08
            try:
                n = int(k[1:3])
            except Exception:
                n = 99
            suf = k[3:] if len(k) > 3 else ""
            return (n, suf)
        return tuple(sorted(keys, key=_key))

    @staticmethod
    def _coerce_resolution_from_profile(p: GeoProfile) -> int:
        px, py = p.pixel_size()
        gsd = min(abs(px), abs(py))
        return min((10, 20, 60), key=lambda r: abs(gsd - r))

    @staticmethod
    def _to_percents(counts: Mapping[int, int], *, total: int) -> Mapping[int, float]:
        if total <= 0:
            return {int(k): 0.0 for k in counts}
        return {int(k): (v / float(total)) * 100.0 for k, v in counts.items()}

    # --------- Export ---------
    def _export(self, labels: GeoRaster, cmap: ClassMap, spec: ClassMapSpec) -> Tuple[Path, Optional[Path]]:
        # GeoTIFF
        out_tif = Path(spec.out_tif) if spec.out_tif else self.settings.out_path("classmap", date=spec.date)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        self.writer.write(str(out_tif), labels)

        # Quicklook opcional
        out_png: Optional[Path] = None
        if spec.out_png or self.ql_exporter:
            out_png = Path(spec.out_png) if spec.out_png else out_tif.with_suffix(".png")
            if self.ql_exporter:
                self.ql_exporter.export_classmap(labels, tuple(self.classifier.classes()), str(out_png), QuicklookSpec())
            else:
                # Fallback mínimo si no hay exporter
                self._save_png_inline(labels, cmap.palette, out_png)

        # Report opcional
        if spec.make_report and self.reporter:
            ctx = {
                "date": spec.date,
                "labels_tif": str(out_tif),
                "quicklook_png": str(out_png) if out_png else None,
                "counts": dict(cmap.counts),
                "percents": dict(self._to_percents(cmap.counts, total=labels.data.size)),
                "palette": {int(k): tuple(v) for k, v in cmap.palette.items()},
                "classes": [c.__dict__ for c in self.classifier.classes()],
            }
            report_path = out_tif.with_suffix(".csv")  # por defecto CSV; cambia a PDF/HTML con otro adapter
            self.reporter.render(spec.report_template, ctx, str(report_path))
        return out_tif, out_png

    @staticmethod
    def _save_png_inline(labels: GeoRaster, palette: Mapping[int, Tuple[int, int, int]], out_path: Path) -> None:
        h, w = labels.data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        ids = np.unique(labels.data)
        for i in ids:
            rgb[labels.data == i] = palette.get(int(i), (0, 0, 0))
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(rgb, mode="RGB").save(out_path)
        except Exception:
            import matplotlib.image as mpimg  # type: ignore
            mpimg.imsave(out_path, rgb)


__all__ = [
    "ClassMapInputs",
    "ClassMapSpec",
    "ClassMapResult",
    "ClassMapService",
]
