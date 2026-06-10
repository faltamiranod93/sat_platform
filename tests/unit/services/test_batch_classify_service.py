"""Unit tests para BatchClassifyService (resolvers inyectados; PNG/CSV en tmp_path)."""
import numpy as np
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.ports.class_map import ClassMap
from satplatform.services.batch_classify_service import (
    BatchClassifyService,
    ClassifierSpec,
)


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


def _service(root, classifiers=None):
    """Resolvers que reproducen el layout del contrato dentro de tmp_path."""
    return BatchClassifyService(
        reader=_FakeReader(),
        writer=_FakeWriter(),
        cmapper=_FakeCmapper(),
        classifiers=classifiers or (
            ClassifierSpec("maha", _FakeClassifier(1)),
            ClassifierSpec("cos", _FakeClassifier(1)),
            ClassifierSpec("euc", _FakeClassifier(2)),
        ),
        classmap_path=lambda d, c: root / "03-Products" / "CLASSMAP" / d / f"classmap_{c}.tif",
        vis_path=lambda d, c: root / "03-Products" / "VIS" / d / f"classmap_{c}.png",
        summary_path=lambda name: root / "04-Analysis" / "CLASSMAP-COMPARE" / f"{name}.csv",
    )


class TestRun:
    def test_writes_tif_and_png_in_schema(self, tmp_path):
        svc = _service(tmp_path)
        svc.run({"20240123": "/x/SCENE_A.tif"}, _classes())
        # tif vía writer en CLASSMAP/{date}/classmap_{clf}.tif
        assert sorted(p.replace(str(tmp_path), "") for p in svc.writer.writes) == [
            "/03-Products/CLASSMAP/20240123/classmap_cos.tif",
            "/03-Products/CLASSMAP/20240123/classmap_euc.tif",
            "/03-Products/CLASSMAP/20240123/classmap_maha.tif",
        ]
        # png en VIS/{date}/
        for key in ("maha", "cos", "euc"):
            assert (tmp_path / "03-Products" / "VIS" / "20240123" / f"classmap_{key}.png").exists()

    def test_scene_result_keyed_by_date(self, tmp_path):
        svc = _service(tmp_path)
        res = svc.run({"20240123": "/x/A.tif"}, _classes())
        assert res.scenes[0].date == "20240123"

    def test_agreement_and_counts(self, tmp_path):
        svc = _service(tmp_path)
        res = svc.run({"20240123": "/x/A.tif"}, _classes())
        agr = res.scenes[0].agreement
        assert agr["cos-maha"] == pytest.approx(100.0)
        assert agr["euc-maha"] == pytest.approx(0.0)
        assert res.scenes[0].counts_by_classifier["maha"] == {1: 16}

    def test_no_refit_per_scene(self, tmp_path):
        svc = _service(tmp_path)
        svc.run({"20240123": "/x/A.tif", "20240128": "/x/B.tif"}, _classes())
        for spec in svc.classifiers:
            assert spec.adapter.predict_calls == 2

    def test_summary_csvs_in_analysis(self, tmp_path):
        svc = _service(tmp_path)
        svc.run({"20240123": "/x/A.tif"}, _classes())
        base = tmp_path / "04-Analysis" / "CLASSMAP-COMPARE"
        assert (base / "counts.csv").exists()
        assert (base / "agreement.csv").exists()


class TestRobustness:
    def test_failing_scene_is_skipped_not_aborted(self, tmp_path):
        class _ReaderFailsOnBad(_FakeReader):
            def read(self, uri, band_index=None):
                if "BAD" in uri:
                    raise RuntimeError("not recognized as being in a supported file format")
                return super().read(uri, band_index)

        svc = _service(tmp_path, classifiers=(ClassifierSpec("maha", _FakeClassifier(1)),))
        svc.reader = _ReaderFailsOnBad()
        res = svc.run(
            {"20240103": "/x/GOOD_A.tif", "20240108": "/x/BAD_B.tif", "20240113": "/x/GOOD_C.tif"},
            _classes(),
        )
        assert len(res.scenes) == 2
        assert len(res.failed) == 1
        assert res.failed[0][0] == "20240108"
        assert (tmp_path / "04-Analysis" / "CLASSMAP-COMPARE" / "counts.csv").exists()
