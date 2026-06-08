"""Unit tests para MahalanobisClassifierAdapter (sin I/O)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.adapters.mahalanobis_classifier import MahalanobisClassifierAdapter
from tests.factories import make_profile, make_raster


def _classes():
    return [
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    ]


def _mcal_df(n_per_class: int = 20) -> pd.DataFrame:
    """DataFrame Mcal sintético con 3 clases bien separadas en espacio espectral."""
    rng = np.random.default_rng(42)
    rows = []
    centers = {1: 1000.0, 2: 5000.0, 3: 9000.0}
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for ng, center in centers.items():
        for _ in range(n_per_class):
            vals = rng.normal(center, 50, len(bands))
            row = {"Ng": ng}
            for b, v in zip(bands, vals):
                row[b] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_bandset(value: float):
    """BandSet 4×4 con valor uniforme en todas las bandas."""
    from satplatform.contracts.products import BandSet
    band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    bands_map = {}
    for b in band_names:
        arr = np.full((4, 4), value, dtype=np.float32)
        from satplatform.contracts.geo import GeoRaster
        bands_map[b] = GeoRaster(data=arr, profile=make_profile(w=4, h=4, dtype="float32"))
    return BandSet(resolution_m=10, bands=bands_map)


class TestMahalanobisClassifierFit:
    def test_name_without_hsl(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        assert clf.name() == "mahalanobis"

    def test_name_with_hsl(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False  # evitar SpectralService en unit test
        )
        clf2 = MahalanobisClassifierAdapter(
            _classes=clf._classes,
            _models=clf._models,
            _band_filter=clf._band_filter,
            _include_hsl=True,
        )
        assert clf2.name() == "mahalanobis_hsl"

    def test_classes_returned(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        assert len(clf.classes()) == 3
        assert {c.id for c in clf.classes()} == {1, 2, 3}

    def test_models_fitted_for_all_classes(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        assert set(clf._models.keys()) == {1, 2, 3}
        for cls_id, (mean, prec) in clf._models.items():
            assert mean.ndim == 1
            assert prec.ndim == 2
            assert prec.shape[0] == prec.shape[1] == len(mean)


class TestMahalanobisClassifierPredict:
    def test_output_shape(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        bs = _make_bandset(1000.0)
        result = clf.predict(bs)
        assert result.data.shape == (4, 4)

    def test_output_dtype_is_int16(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        result = clf.predict(_make_bandset(5000.0))
        assert result.data.dtype == np.int16

    def test_correct_label_class1(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        result = clf.predict(_make_bandset(1000.0))
        assert np.all(result.data == 1)

    def test_correct_label_class2(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        result = clf.predict(_make_bandset(5000.0))
        assert np.all(result.data == 2)

    def test_correct_label_class3(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        result = clf.predict(_make_bandset(9000.0))
        assert np.all(result.data == 3)

    def test_output_profile_geometry_matches_input(self):
        clf = MahalanobisClassifierAdapter.fit(
            _mcal_df(), _classes(), include_hsl=False
        )
        bs = _make_bandset(5000.0)
        result = clf.predict(bs)
        assert result.profile.width == 4
        assert result.profile.height == 4
        assert result.profile.count == 1
