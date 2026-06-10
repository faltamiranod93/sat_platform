"""Orquestador de clasificación batch multi-escena con comparación de clasificadores.

Para cada escena (TIFF multibanda Sentinel Hub, georef corregida al vuelo por el
reader decorado) corre N clasificadores ya entrenados, exporta un classmap por
clasificador (GeoTIFF + PNG) y mide el acuerdo pixel-a-pixel entre ellos.

Los clasificadores llegan entrenados (fit una sola vez); aquí solo se infiere.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..contracts.core import ClassLabel
from ..ports.class_map import ClassMapPort
from ..ports.pixel_class import PixelClassifierPort
from ..ports.raster_read import RasterReaderPort, URI
from ..ports.raster_write import RasterWriterPort
from .classmap_service import ClassMapService
from .multiband_loader import load_multiband_bandset

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifierSpec:
    key: str                         # "maha" | "cos" | "euc"
    adapter: PixelClassifierPort     # ya entrenado


@dataclass(frozen=True)
class SceneResult:
    date: str                        # YYYYMMDD — clave de trazabilidad
    counts_by_classifier: Dict[str, Dict[int, int]]
    agreement: Dict[str, float]      # "maha-cos" → % de píxeles iguales


@dataclass(frozen=True)
class BatchResult:
    scenes: Tuple[SceneResult, ...]
    failed: Tuple[Tuple[str, str], ...] = ()   # (date, mensaje de error)


@dataclass
class BatchClassifyService:
    """Orquesta N clasificadores por escena. Las rutas de salida llegan como
    resolvers inyectados (el composition root los construye desde el contrato
    output_patterns), así el servicio no depende de Settings."""

    reader: RasterReaderPort
    writer: RasterWriterPort
    cmapper: ClassMapPort
    classifiers: Tuple[ClassifierSpec, ...]
    classmap_path: Callable[[str, str], Path]   # (date, classifier) -> .tif
    vis_path: Callable[[str, str], Path]        # (date, classifier) -> .png
    summary_path: Callable[[str], Path]         # (name) -> .csv

    def run(
        self,
        scenes: Mapping[str, URI],               # {date(YYYYMMDD): uri}, 1 por fecha
        classes: Sequence[ClassLabel],
    ) -> BatchResult:
        results: list[SceneResult] = []
        failed: list[Tuple[str, str]] = []
        items = sorted(scenes.items())
        n = len(items)

        for i, (date, uri) in enumerate(items):
            _log.info("[%d/%d] %s", i + 1, n, date)
            try:
                results.append(self._classify_scene(date, uri, classes))
            except Exception as e:  # una escena corrupta/ilegible no debe tumbar el batch
                _log.warning("[%d/%d] FALLÓ %s: %s", i + 1, n, date, e)
                failed.append((date, str(e)))

        batch = BatchResult(scenes=tuple(results), failed=tuple(failed))
        self._write_summary(batch, classes)
        return batch

    def _classify_scene(self, date: str, uri: str, classes) -> SceneResult:
        bandset = load_multiband_bandset(self.reader, uri)
        label_arrays: Dict[str, np.ndarray] = {}
        counts: Dict[str, Dict[int, int]] = {}

        for spec in self.classifiers:
            labels = spec.adapter.predict(bandset)
            cmap = self.cmapper.from_labels(labels, tuple(classes))
            tif = self.classmap_path(date, spec.key)
            png = self.vis_path(date, spec.key)
            tif.parent.mkdir(parents=True, exist_ok=True)
            png.parent.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(tif), labels)
            ClassMapService._save_png_inline(labels, cmap.palette, png)
            label_arrays[spec.key] = labels.data
            counts[spec.key] = {int(k): int(v) for k, v in cmap.counts.items()}

        return SceneResult(
            date=date,
            counts_by_classifier=counts,
            agreement=self._agreement(label_arrays),
        )

    @staticmethod
    def _agreement(label_arrays: Mapping[str, np.ndarray]) -> Dict[str, float]:
        keys = sorted(label_arrays)
        out: Dict[str, float] = {}
        for a, b in itertools.combinations(keys, 2):
            out[f"{a}-{b}"] = float((label_arrays[a] == label_arrays[b]).mean()) * 100.0
        return out

    def _write_summary(self, batch: BatchResult, classes: Sequence[ClassLabel]) -> None:
        name_by_id = {int(c.id): c.name for c in classes}

        count_rows = []
        for sr in batch.scenes:
            for key, counts in sr.counts_by_classifier.items():
                total = sum(counts.values()) or 1
                for cid, px in sorted(counts.items()):
                    count_rows.append({
                        "date": sr.date, "classifier": key, "class_id": cid,
                        "class_name": name_by_id.get(cid, str(cid)),
                        "px": px, "pct": round(px / total * 100.0, 4),
                    })
        agr_rows = [
            {"date": sr.date, "pair": pair, "pct_agreement": round(pct, 4)}
            for sr in batch.scenes for pair, pct in sr.agreement.items()
        ]

        counts_path = self.summary_path("counts")
        agr_path = self.summary_path("agreement")
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        agr_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(count_rows).to_csv(counts_path, index=False)
        pd.DataFrame(agr_rows).to_csv(agr_path, index=False)


__all__ = ["BatchClassifyService", "ClassifierSpec", "SceneResult", "BatchResult"]
