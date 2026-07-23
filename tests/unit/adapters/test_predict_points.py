"""predict_points() debe coincidir con predict() raster en los mismos píxeles."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.contracts.geo import GeoRaster
from satplatform.contracts.products import BandSet
from satplatform.adapters.mahalanobis_classifier import MahalanobisClassifierAdapter
from satplatform.adapters.euclidean_classifier import EuclideanClassifierAdapter
from satplatform.adapters.cosine_classifier import CosineClassifierAdapter
from satplatform.adapters.legacy_pixelclass_adapter import LegacyPixelClassifier
from tests.factories import make_profile

BANDS = ["B02", "B03", "B04", "B05"]


def _classes():
    return [
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    ]


def _mcal_df():
    rng = np.random.default_rng(0)
    centers = {1: 500.0, 2: 5000.0, 3: 9500.0}
    rows = []
    for ng, c in centers.items():
        for _ in range(15):
            row = {"Ng": ng}
            for b in BANDS:
                row[b] = float(rng.normal(c, 40.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _bandset_and_df(seed=1, h=6, w=7):
    """BandSet con píxeles variados + DataFrame con los MISMOS valores (orden C/row-major)."""
    rng = np.random.default_rng(seed)
    bands_map = {}
    cols = {}
    for b in BANDS:
        arr = rng.uniform(0, 10000, size=(h, w)).astype(np.float32)
        bands_map[b] = GeoRaster(data=arr, profile=make_profile(w=w, h=h, dtype="float32"))
        cols[b] = arr.ravel()  # C-order, igual que from_bandset
    bs = BandSet(resolution_m=10, bands=bands_map)
    df = pd.DataFrame(cols)
    return bs, df


@pytest.mark.parametrize("factory", [
    lambda: MahalanobisClassifierAdapter.fit(_mcal_df(), _classes(), band_filter=BANDS, include_hsl=False),
    lambda: EuclideanClassifierAdapter.fit(_mcal_df(), _classes(), band_filter=BANDS, include_hsl=False),
    lambda: CosineClassifierAdapter.fit(_mcal_df(), _classes(), band_filter=BANDS, include_hsl=False),
    lambda: CosineClassifierAdapter.fit(_mcal_df(), _classes(), band_filter=BANDS, include_hsl=False, two_stage=True),
])
def test_predict_points_matches_raster(factory):
    clf = factory()
    bs, df = _bandset_and_df()
    raster = clf.predict(bs).data.ravel()  # C-order
    points = clf.predict_points(df)
    assert points.shape == raster.shape
    assert np.array_equal(points, raster)


def test_legacy_predict_points_not_implemented():
    clf = LegacyPixelClassifier(classes_def=_classes())
    with pytest.raises(NotImplementedError):
        clf.predict_points(pd.DataFrame({"B03": [1.0], "B04": [1.0], "B08": [1.0], "B11": [1.0]}))
