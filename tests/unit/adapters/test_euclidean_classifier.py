"""Unit tests para EuclideanClassifierAdapter (sin I/O)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.contracts.geo import GeoRaster
from satplatform.contracts.products import BandSet
from satplatform.adapters.euclidean_classifier import EuclideanClassifierAdapter
from tests.factories import make_profile


def _classes():
    return [
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    ]


BANDS = ["B02", "B03", "B04", "B05"]


def _mcal_df() -> pd.DataFrame:
    """3 clases con centroides bien separados en 4 bandas."""
    rng = np.random.default_rng(0)
    centers = {1: 500.0, 2: 5000.0, 3: 9500.0}
    rows = []
    for ng, center in centers.items():
        for _ in range(10):
            vals = rng.normal(center, 30.0, len(BANDS))
            row = {"Ng": ng}
            for b, v in zip(BANDS, vals):
                row[b] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_bandset(value: float):
    bands_map = {}
    for b in BANDS:
        arr = np.full((5, 5), value, dtype=np.float32)
        bands_map[b] = GeoRaster(data=arr, profile=make_profile(w=5, h=5, dtype="float32"))
    return BandSet(resolution_m=10, bands=bands_map)


class TestEuclideanClassifierFit:
    def test_name(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        assert clf.name() == "euclidean"

    def test_classes_returned(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        assert {c.id for c in clf.classes()} == {1, 2, 3}

    def test_reference_shape(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        assert clf._reference.shape == (3, len(BANDS))


class TestEuclideanClassifierPredict:
    def test_output_shape(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(500.0))
        assert result.data.shape == (5, 5)

    def test_output_dtype(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(5000.0))
        assert result.data.dtype == np.int16

    def test_correct_label_class1(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(500.0))
        assert np.all(result.data == 1)

    def test_correct_label_class2(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(5000.0))
        assert np.all(result.data == 2)

    def test_correct_label_class3(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(9500.0))
        assert np.all(result.data == 3)

    def test_output_profile_matches_input(self):
        clf = EuclideanClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        result = clf.predict(_make_bandset(5000.0))
        assert result.profile.width == 5
        assert result.profile.height == 5
        assert result.profile.count == 1
