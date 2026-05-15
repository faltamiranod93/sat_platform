"""Tests del contrato de ClassMapService usando fakes de ports.

La API real es:
  ClassMapService.run(inputs: ClassMapInputs, spec: ClassMapSpec) -> ClassMapResult

donde inputs.band_uris se resuelve vía RasterReaderPort y la inferencia es
delegada a PixelClassifierPort. El mapeo de labels→counts/palette ocurre en
ClassMapPort.from_labels().
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pytest

csvc = pytest.importorskip(
    "satplatform.services.classmap_service", reason="ClassMapService no encontrado"
)

from satplatform.contracts.core import ClassLabel, MacroClass, RGB8
from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef
from satplatform.contracts.products import BandSet
from satplatform.ports.class_map import ClassMap


# --------- utilidades sintéticas ---------

def _prof(w=20, h=15, px=10.0, dtype="float32"):
    return GeoProfile(
        count=1, dtype=dtype, width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px),
        crs=CRSRef.from_epsg(32719),
    )


def _geo_band(value=0.0, w=20, h=15, dtype=np.float32):
    arr = np.full((h, w), float(value), dtype=dtype)
    return GeoRaster(arr, _prof(w, h, dtype=str(np.dtype(dtype))))


def _labels():
    return (
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA, color=RGB8(r=31, g=119, b=180)),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE, color=RGB8(r=214, g=39, b=40)),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO, color=RGB8(r=152, g=223, b=138)),
    )


# --------- fakes de puertos ---------

class FakeReader:
    """Devuelve bandas sintéticas indexadas por URI."""
    def __init__(self, uri_to_value: Mapping[str, float]):
        self._map = dict(uri_to_value)

    def read(self, uri: str) -> GeoRaster:
        v = self._map.get(uri, 0.5)
        return _geo_band(value=v)


class FakeClassifier:
    """Clasifica usando una banda fija: B04 > 0.6 → 2 (relave), else 1 (agua)."""
    def __init__(self, classes_def: Sequence[ClassLabel]):
        self._classes = tuple(classes_def)

    def name(self) -> str:
        return "fake-classifier"

    def classes(self) -> Sequence[ClassLabel]:
        return self._classes

    def predict(self, bands: BandSet, *, calibration_id=None) -> GeoRaster:
        b04 = bands.bands["B04"].data
        out = np.where(b04 > 0.6, 2, 1).astype(np.uint8)
        p = bands.bands["B04"].profile
        prof = GeoProfile(
            count=1, dtype="uint8", width=p.width, height=p.height,
            transform=p.transform, crs=p.crs, nodata=0,
        )
        return GeoRaster(out, prof)


class FakeClassMapper:
    """Cuenta píxeles y construye paleta desde las labels."""
    def from_labels(self, labels: GeoRaster, classes: Sequence[ClassLabel]) -> ClassMap:
        data = labels.data
        ids, counts = np.unique(data, return_counts=True)
        counts_map = {int(i): int(c) for i, c in zip(ids, counts)}
        palette = {int(c.id): (c.color.r, c.color.g, c.color.b) for c in classes}
        return ClassMap(labels=labels, counts=counts_map, palette=palette)


# --------- tests ---------

def test_classmap_service_orchestrates_ports_and_returns_result():
    classes = _labels()
    # B04 alto → relave; otras bandas bajas
    reader = FakeReader({
        "B03.tif": 0.2,
        "B04.tif": 0.8,
        "B08.tif": 0.3,
        "B11.tif": 0.5,
    })
    classifier = FakeClassifier(classes_def=classes)
    cmapper = FakeClassMapper()

    svc = csvc.ClassMapService(
        reader=reader, classifier=classifier, cmapper=cmapper,
    )

    inputs = csvc.ClassMapInputs(
        band_uris={"B03": "B03.tif", "B04": "B04.tif", "B08": "B08.tif", "B11": "B11.tif"},
        classes=classes,
    )
    spec = csvc.ClassMapSpec(date="20250115")

    result = svc.run(inputs, spec)

    assert isinstance(result, csvc.ClassMapResult)
    assert result.band_order == ("B03", "B04", "B08", "B11")
    assert result.resolution_m == 10
    # Todos los píxeles deberían ser clase 2 (relave) porque B04=0.8 > 0.6
    assert result.counts.get(2, 0) > 0
    assert 1 not in result.counts or result.counts[1] == 0
    # Sin out_tif/out_png → no se escribe nada
    assert result.labels_tif is None
    assert result.quicklook_png is None


def test_classmap_service_requires_reader_classifier_cmapper():
    svc = csvc.ClassMapService()
    inputs = csvc.ClassMapInputs(band_uris={"B04": "x.tif"})
    spec = csvc.ClassMapSpec(date="20250115")
    with pytest.raises(RuntimeError, match="RasterReaderPort"):
        svc.run(inputs, spec)


def test_classmap_inputs_empty_band_uris_raises():
    classes = _labels()
    reader = FakeReader({})
    classifier = FakeClassifier(classes_def=classes)
    cmapper = FakeClassMapper()

    svc = csvc.ClassMapService(reader=reader, classifier=classifier, cmapper=cmapper)
    inputs = csvc.ClassMapInputs(band_uris={}, classes=classes)
    spec = csvc.ClassMapSpec(date="20250115")
    with pytest.raises(ValueError, match="band_uris vacío"):
        svc.run(inputs, spec)
