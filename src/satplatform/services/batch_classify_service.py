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
from typing import Dict, Mapping, Sequence, Tuple

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
    scene_id: str
    counts_by_classifier: Dict[str, Dict[int, int]]
    agreement: Dict[str, float]      # "maha-cos" → % de píxeles iguales


@dataclass(frozen=True)
class BatchResult:
    scenes: Tuple[SceneResult, ...]
    failed: Tuple[Tuple[str, str], ...] = ()   # (scene_id, mensaje de error)


@dataclass
class BatchClassifyService:
    reader: RasterReaderPort
    writer: RasterWriterPort
    cmapper: ClassMapPort
    classifiers: Tuple[ClassifierSpec, ...]

    def run(
        self,
        scene_uris: Sequence[URI],
        classes: Sequence[ClassLabel],
        out_root: str | Path,
    ) -> BatchResult:
        out_root = Path(out_root)
        results: list[SceneResult] = []
        failed: list[Tuple[str, str]] = []
        n = len(scene_uris)

        for i, uri in enumerate(scene_uris):
            scene_id = Path(uri).stem
            _log.info("[%d/%d] %s", i + 1, n, scene_id)
            try:
                results.append(self._classify_scene(uri, scene_id, classes, out_root))
            except Exception as e:  # una escena corrupta/ilegible no debe tumbar el batch
                _log.warning("[%d/%d] FALLÓ %s: %s", i + 1, n, scene_id, e)
                failed.append((scene_id, str(e)))

        batch = BatchResult(scenes=tuple(results), failed=tuple(failed))
        self._write_summary(batch, classes, out_root)
        return batch

    def _classify_scene(self, uri, scene_id, classes, out_root: Path) -> SceneResult:
        bandset = load_multiband_bandset(self.reader, uri)
        scene_dir = out_root / scene_id
        label_arrays: Dict[str, np.ndarray] = {}
        counts: Dict[str, Dict[int, int]] = {}

        for spec in self.classifiers:
            labels = spec.adapter.predict(bandset)
            cmap = self.cmapper.from_labels(labels, tuple(classes))
            scene_dir.mkdir(parents=True, exist_ok=True)
            self.writer.write(str(scene_dir / f"classmap_{spec.key}.tif"), labels)
            ClassMapService._save_png_inline(
                labels, cmap.palette, scene_dir / f"classmap_{spec.key}.png"
            )
            label_arrays[spec.key] = labels.data
            counts[spec.key] = {int(k): int(v) for k, v in cmap.counts.items()}

        return SceneResult(
            scene_id=scene_id,
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

    @staticmethod
    def _write_summary(batch: BatchResult, classes: Sequence[ClassLabel], out_root: Path) -> None:
        summary_dir = out_root / "_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        name_by_id = {int(c.id): c.name for c in classes}

        count_rows = []
        for sr in batch.scenes:
            for key, counts in sr.counts_by_classifier.items():
                total = sum(counts.values()) or 1
                for cid, px in sorted(counts.items()):
                    count_rows.append({
                        "scene": sr.scene_id, "classifier": key, "class_id": cid,
                        "class_name": name_by_id.get(cid, str(cid)),
                        "px": px, "pct": round(px / total * 100.0, 4),
                    })
        pd.DataFrame(count_rows).to_csv(summary_dir / "counts.csv", index=False)

        agr_rows = [
            {"scene": sr.scene_id, "pair": pair, "pct_agreement": round(pct, 4)}
            for sr in batch.scenes for pair, pct in sr.agreement.items()
        ]
        pd.DataFrame(agr_rows).to_csv(summary_dir / "agreement.csv", index=False)


__all__ = ["BatchClassifyService", "ClassifierSpec", "SceneResult", "BatchResult"]
