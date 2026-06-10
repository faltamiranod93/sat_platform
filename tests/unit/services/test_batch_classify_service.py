"""Unit tests para BatchClassifyService (sin I/O raster real; PNG/CSV en tmp_path)."""
import numpy as np
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.ports.class_map import ClassMap
from satplatform.services.batch_classify_service import (
    BatchClassifyService,
    ClassifierSpec,
)

_BAND_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06",
               "B07", "B08", "B8A", "B09", "B11", "B12")


def _profile(h=4, w=4):
    return GeoProfile(
        count=1, dtype="int16", width=w, height=h,
        transform=(479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


def _classes():
    return (
        ClassLabel(id=1, name="agua", macro=MacroClass.AGUA, color={"r": 0, "g": 0, "b": 200}),
        ClassLabel(id=2, name="relave", macro=MacroClass.RELAVE, color={"r": 200, "g": 0, "b": 0}),
    )


class _FakeReader:
    def read(self, uri, band_index=None):
        return GeoRaster(data=np.full((4, 4), band_index, dtype=np.uint16), profile=_profile())


class _FakeClassifier:
    """Devuelve labels constantes = value; cuenta llamadas a predict/fit."""

    def __init__(self, value):
        self.value = value
        self.predict_calls = 0

    def predict(self, bandset, *, calibration_id=None):
        self.predict_calls += 1
        first = next(iter(bandset.bands.values()))
        H, W = first.profile.height, first.profile.width
        return GeoRaster(data=np.full((H, W), self.value, dtype=np.int16), profile=_profile(H, W))


class _FakeCmapper:
    def from_labels(self, labels, classes):
        ids, cnts = np.unique(labels.data, return_counts=True)
        counts = {int(i): int(c) for i, c in zip(ids, cnts)}
        palette = {int(c.id): c.color.as_tuple() for c in classes}
        return ClassMap(labels=labels, counts=counts, palette=palette)


class _FakeWriter:
    def __init__(self):
        self.writes = []

    def write(self, uri, raster, *, compress=None, tiled=True):
        self.writes.append(uri)
        return uri


def _service():
    return BatchClassifyService(
        reader=_FakeReader(),
        writer=_FakeWriter(),
        cmapper=_FakeCmapper(),
        classifiers=(
            ClassifierSpec("maha", _FakeClassifier(1)),
            ClassifierSpec("cos", _FakeClassifier(1)),
            ClassifierSpec("euc", _FakeClassifier(2)),
        ),
    )


class TestRun:
    def test_writes_tif_and_png_per_classifier(self, tmp_path):
        svc = _service()
        svc.run(["/x/SCENE_A.tif"], _classes(), tmp_path)
        # 3 tif vía writer
        assert len(svc.writer.writes) == 3
        assert {p.split("/")[-1] for p in svc.writer.writes} == {
            "classmap_maha.tif", "classmap_cos.tif", "classmap_euc.tif"
        }
        # 3 png en disco
        d = tmp_path / "SCENE_A"
        for key in ("maha", "cos", "euc"):
            assert (d / f"classmap_{key}.png").exists()

    def test_agreement(self, tmp_path):
        svc = _service()
        res = svc.run(["/x/SCENE_A.tif"], _classes(), tmp_path)
        agr = res.scenes[0].agreement
        assert agr["cos-maha"] == pytest.approx(100.0)   # ambos = 1
        assert agr["euc-maha"] == pytest.approx(0.0)      # 2 vs 1
        assert agr["cos-euc"] == pytest.approx(0.0)

    def test_counts(self, tmp_path):
        svc = _service()
        res = svc.run(["/x/SCENE_A.tif"], _classes(), tmp_path)
        counts = res.scenes[0].counts_by_classifier
        assert counts["maha"] == {1: 16}   # 4×4 todo clase 1
        assert counts["euc"] == {2: 16}

    def test_no_refit_per_scene(self, tmp_path):
        """Los clasificadores llegan entrenados: predict 1×/escena, sin reentrenar."""
        svc = _service()
        svc.run(["/x/A.tif", "/x/B.tif"], _classes(), tmp_path)
        for spec in svc.classifiers:
            assert spec.adapter.predict_calls == 2  # 2 escenas

    def test_summary_csvs_written(self, tmp_path):
        svc = _service()
        svc.run(["/x/A.tif", "/x/B.tif"], _classes(), tmp_path)
        assert (tmp_path / "_summary" / "counts.csv").exists()
        assert (tmp_path / "_summary" / "agreement.csv").exists()


class _FailingClassifier:
    """Levanta excepción en predict, para probar robustez por escena."""
    def predict(self, bandset, *, calibration_id=None):
        raise RuntimeError("no recognized as raster")


class TestRobustness:
    def test_failing_scene_is_skipped_not_aborted(self, tmp_path):
        from satplatform.services.batch_classify_service import BatchClassifyService, ClassifierSpec
        # primer clasificador OK, pero usamos un reader que falla en una escena:
        class _ReaderFailsOnB(_FakeReader):
            def read(self, uri, band_index=None):
                if "BAD" in uri:
                    raise RuntimeError("not recognized as being in a supported file format")
                return super().read(uri, band_index)

        svc = BatchClassifyService(
            reader=_ReaderFailsOnB(),
            writer=_FakeWriter(),
            cmapper=_FakeCmapper(),
            classifiers=(ClassifierSpec("maha", _FakeClassifier(1)),),
        )
        res = svc.run(["/x/GOOD_A.tif", "/x/BAD_B.tif", "/x/GOOD_C.tif"], _classes(), tmp_path)
        # 2 OK, 1 fallida — el batch no aborta
        assert len(res.scenes) == 2
        assert len(res.failed) == 1
        assert res.failed[0][0] == "BAD_B"
        # summary igual se escribe
        assert (tmp_path / "_summary" / "counts.csv").exists()
