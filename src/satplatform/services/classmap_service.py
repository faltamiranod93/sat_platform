# src/satplatform/services/classmap_service.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Mapping as TMapping

import numpy as np

from ..contracts.core import ClassLabel, S2BandName, SceneId
from ..contracts.geo import GeoRaster, GeoProfile, CRSRef, validate_profile_compat
from ..contracts.products import BandSet

from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort
from ..ports.roi import ROIClipperPort, GeoJSON
from ..ports.preprocessing import PreprocessingPort, NormalizeSpec
from ..ports.pixel_class import PixelClassifierPort
from ..ports.class_map import ClassMapPort, ClassMap
from ..ports.exporters import QuicklookExporterPort, ReportExporterPort, QuicklookSpec


"""
Servicio de construcción de CLASSMAP (inspiración "v5"), contracts-first.
Pipeline determinista:
  LOAD → ALIGN → (ROI) → (PRE) → INFER → POST (stats/palette) → (EXPORT opcional)

No asume backends concretos: todo va vía *ports*. No usa Settings ni calcula rutas.
"""

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
    date: str  # YYYYMMDD (informativo; NO se usa para construir rutas)
    order: Optional[Tuple[S2BandName, ...]] = None
    normalize: bool = False
    norm_spec: NormalizeSpec = NormalizeSpec()
    roi_geojson: Optional[GeoJSON] = None
    roi_crs: Optional[CRSRef] = None
    out_tif: Optional[Path] = None       # si None -> no escribe GeoTIFF
    out_png: Optional[Path] = None       # si None -> no escribe quicklook
    make_report: bool = False            # si True y hay reporter, escribe reporte junto a out_tif (si existe)
    report_template: str = "classmap_summary"

@dataclass(frozen=True)
class ClassMapResult:
    labels_tif: Optional[Path]                 # None si no se escribió
    quicklook_png: Optional[Path]              # None si no se escribió
    counts: TMapping[int, int]
    percents: TMapping[int, float]
    palette: TMapping[int, Tuple[int, int, int]]
    band_order: Tuple[S2BandName, ...]
    resolution_m: int

# ----------------------
# Servicio
# ----------------------

@dataclass
class ClassMapService:
    reader: Optional[RasterReaderPort] = None
    classifier: Optional[PixelClassifierPort] = None
    cmapper: Optional[ClassMapPort] = None
    writer: Optional[RasterWriterPort] = None
    preproc: Optional[PreprocessingPort] = None
    clipper: Optional[ROIClipperPort] = None
    ql_exporter: Optional[QuicklookExporterPort] = None
    reporter: Optional[ReportExporterPort] = None

    def __init__(
        self,
        classmap_port: Optional[ClassMapPort] = None,
        pixel_port: Optional[PixelClassifierPort] = None,
        *,
        reader: Optional[RasterReaderPort] = None,
        writer: Optional[RasterWriterPort] = None,
        classifier: Optional[PixelClassifierPort] = None,
        cmapper: Optional[ClassMapPort] = None,
        preproc: Optional[PreprocessingPort] = None,
        clipper: Optional[ROIClipperPort] = None,
        ql_exporter: Optional[QuicklookExporterPort] = None,
        reporter: Optional[ReportExporterPort] = None,
    ):
        self.reader = reader
        self.writer = writer
        self.classifier = classifier or pixel_port
        self.cmapper = cmapper or classmap_port
        self.preproc = preproc
        self.clipper = clipper
        self.ql_exporter = ql_exporter
        self.reporter = reporter

    # --------- API principal ---------
    def run(self, inputs: ClassMapInputs, spec: ClassMapSpec) -> ClassMapResult:
        if self.reader is None:
            raise RuntimeError("RasterReaderPort no configurado")
        if self.classifier is None:
            raise RuntimeError("PixelClassifierPort no configurado")
        if self.cmapper is None:
            raise RuntimeError("ClassMapPort no configurado")
        # 1) LOAD + ALIGN (+ ROI) + (PRE)
        rasters, ref_profile = self._load_align(inputs.band_uris, spec)
        if spec.normalize:
            if not self.preproc:
                raise RuntimeError("normalize=True requiere un PreprocessingPort configurado")
            rasters = {k: self.preproc.normalize(v, spec.norm_spec) for k, v in rasters.items()}  # type: ignore[union-attr]

        # 2) BANDSET determinista
        band_order = spec.order or self._stable_band_order(inputs.band_uris.keys())
        res_m = self._coerce_resolution_from_profile(ref_profile)
        bandset = BandSet(resolution_m=res_m, bands=rasters)

        # 3) INFER
        labels: GeoRaster = self.classifier.predict(bandset)

        # 4) POST → ClassMap (counts / palette)
        classes = tuple(inputs.classes or self.classifier.classes())
        cmap: ClassMap = self.cmapper.from_labels(labels, classes)
        perc = self._to_percents(cmap.counts, total=int(labels.data.size))

        # 5) EXPORT opcional
        out_tif, out_png = self._export(labels, cmap, spec)

        return ClassMapResult(
            labels_tif=out_tif,
            quicklook_png=out_png,
            counts=cmap.counts,
            percents=perc,
            palette=cmap.palette,
            band_order=tuple(band_order),
            resolution_m=res_m,
        )

    # --------- Fases internas ---------
    def _load_align(
        self,
        band_uris: Mapping[S2BandName, str],
        spec: ClassMapSpec
    ) -> Tuple[Dict[S2BandName, GeoRaster], GeoProfile]:
        if not band_uris:
            raise ValueError("band_uris vacío")
        rasters: Dict[S2BandName, GeoRaster] = {}
        ref: Optional[GeoProfile] = None
        roi_g = spec.roi_geojson

        for b, uri in band_uris.items():
            r = self.reader.read(uri)
            if roi_g is not None:
                if not self.clipper:
                    raise RuntimeError("Se proporcionó ROI pero no hay ROIClipperPort configurado")
                # si no entregan roi_crs, clipea en el CRS del raster
                r = self.clipper.clip_raster(r, roi_g, spec.roi_crs or r.profile.crs)
            if ref is None:
                ref = r.profile
            else:
                validate_profile_compat(ref, r.profile)
            rasters[b] = r
        assert ref is not None
        return rasters, ref

    @staticmethod
    def _stable_band_order(keys: Iterable[S2BandName]) -> Tuple[S2BandName, ...]:
        # Orden reproducible: B02, B03, ..., B12, B8A
        def _key(k: str) -> Tuple[int, str]:
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

    # --------- Export (opcional y sin rutas implícitas) ---------
    def _export(self, labels: GeoRaster, cmap: ClassMap, spec: ClassMapSpec) -> Tuple[Optional[Path], Optional[Path]]:
        out_tif: Optional[Path] = None
        out_png: Optional[Path] = None

        # GeoTIFF (solo si se provee path y existe writer)
        if spec.out_tif is not None and self.writer is not None:
            out_tif = Path(spec.out_tif)
            out_tif.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(out_tif), labels)

        # Quicklook PNG (solo si se provee path)
        if spec.out_png is not None:
            out_png = Path(spec.out_png)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            if self.ql_exporter:
                self.ql_exporter.export_classmap(labels, tuple(self.classifier.classes()), str(out_png), QuicklookSpec())
            else:
                self._save_png_inline(labels, cmap.palette, out_png)

        # Reporte (solo si el llamador quiere y hay reporter; requiere out_tif para ubicar carpeta)
        if spec.make_report and self.reporter and out_tif is not None:
            ctx = {
                "date": spec.date,
                "labels_tif": str(out_tif),
                "quicklook_png": str(out_png) if out_png else None,
                "counts": dict(cmap.counts),
                "percents": dict(self._to_percents(cmap.counts, total=int(labels.data.size))),
                "palette": {int(k): tuple(v) for k, v in cmap.palette.items()},
                "classes": [c.__dict__ for c in self.classifier.classes()],
            }
            report_path = out_tif.with_suffix(".csv")
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
