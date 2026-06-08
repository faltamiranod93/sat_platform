"""Unit tests para CosineClassifierAdapter (sin I/O)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.contracts.geo import GeoRaster
from satplatform.contracts.products import BandSet
from satplatform.adapters.cosine_classifier import CosineClassifierAdapter
from tests.factories import make_profile


def _classes():
    return [
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    ]


BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def _mcal_df() -> pd.DataFrame:
    """DataFrame con 3 clases que difieren en forma espectral (diferente ratio de bandas)."""
    rng = np.random.default_rng(7)
    # Clases separadas en dirección coseno: distintos perfiles espectrales
    profiles = {
        1: np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # azul dominante
        2: np.array([0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]),  # VNIR dominante
        3: np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0]),  # SWIR dominante
    }
    rows = []
    for ng, profile in profiles.items():
        scale = 5000.0
        for _ in range(10):
            vals = profile * scale + rng.normal(0, 50, len(BANDS))
            row = {"Ng": ng}
            for b, v in zip(BANDS, vals):
                row[b] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_bandset(values: np.ndarray):
    """BandSet 4×4 con perfil espectral dado (values es array de len(BANDS))."""
    bands_map = {}
    for b, v in zip(BANDS, values):
        arr = np.full((4, 4), v, dtype=np.float32)
        bands_map[b] = GeoRaster(data=arr, profile=make_profile(w=4, h=4, dtype="float32"))
    return BandSet(resolution_m=10, bands=bands_map)


class TestCosineClassifierFit:
    def test_name_single_stage(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False, two_stage=False
        )
        assert clf.name() == "cosine"

    def test_name_two_stage(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False, two_stage=True
        )
        assert clf.name() == "cosine_twostage"

    def test_classes_count(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        assert len(clf.classes()) == 3

    def test_reference_is_normalized(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        norms = np.linalg.norm(clf._reference, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestCosineClassifierPredict:
    def test_output_shape(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        profile1 = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 5000.0
        result = clf.predict(_make_bandset(profile1))
        assert result.data.shape == (4, 4)

    def test_output_dtype(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        profile1 = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 5000.0
        result = clf.predict(_make_bandset(profile1))
        assert result.data.dtype == np.int16

    def test_correct_class1_cosine(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        profile1 = np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 5000.0
        result = clf.predict(_make_bandset(profile1))
        assert np.all(result.data == 1)

    def test_correct_class3_cosine(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False
        )
        profile3 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0]) * 5000.0
        result = clf.predict(_make_bandset(profile3))
        assert np.all(result.data == 3)

    def test_two_stage_same_shape(self):
        clf = CosineClassifierAdapter.fit(
            _mcal_df(), _classes(), band_filter=BANDS, include_hsl=False, two_stage=True,
            stage2_class_ids=[2, 3]
        )
        profile2 = np.array([0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]) * 5000.0
        result = clf.predict(_make_bandset(profile2))
        assert result.data.shape == (4, 4)
        assert result.data.dtype == np.int16
