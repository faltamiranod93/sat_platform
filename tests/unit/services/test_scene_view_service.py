"""Unit tests para SceneViewService (dominio puro)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.contracts.products import BandSet
from satplatform.services.scene_view_service import SceneViewService

_GT = (479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0)


def _profile(h=5, w=5):
    return GeoProfile(count=1, dtype="uint16", width=w, height=h,
                      transform=_GT, crs=CRSRef.from_epsg(32719), nodata=None)


def _bandset(h=5, w=5):
    # cada banda = valor constante distinto, para verificar extracción
    vals = {"B02": 100, "B03": 200, "B04": 300, "B08": 800, "B11": 1100, "B12": 1200}
    bands = {b: GeoRaster(data=np.full((h, w), v, dtype=np.uint16), profile=_profile(h, w))
             for b, v in vals.items()}
    return BandSet(resolution_m=10, bands=bands)


@pytest.fixture
def svc():
    return SceneViewService()


class TestRgb:
    def test_shape_and_dtype(self, svc):
        rgb = svc.rgb_composite(_bandset(), "TrueColor")
        assert rgb.shape == (5, 5, 3)
        assert rgb.dtype == np.uint8

    def test_presets(self, svc):
        for p in svc.presets():
            assert svc.rgb_composite(_bandset(), p).shape == (5, 5, 3)

    def test_bad_preset_raises(self, svc):
        with pytest.raises(ValueError):
            svc.rgb_composite(_bandset(), "NoExiste")


class TestPointsToPixels:
    def test_world_to_pixel_projection(self, svc):
        # origen (479556, 7306103), pixel 10 → (479561, 7306098) ≈ col 0.5 row 0.5
        utm = pd.DataFrame({"UTM_E": [479561.0], "UTM_N": [7306098.0], "Ng": [1]})
        out = svc.points_to_pixels(utm, _profile())
        assert out["col"].iloc[0] == pytest.approx(0.5, abs=1e-6)
        assert out["row"].iloc[0] == pytest.approx(0.5, abs=1e-6)


class TestPixelSignature:
    def test_extracts_all_bands(self, svc):
        sig = svc.pixel_signature(_bandset(), col=2, row=2)
        assert sig["B08"] == 800
        assert sig["B12"] == 1200
        assert set(sig) == {"B02", "B03", "B04", "B08", "B11", "B12"}

    def test_out_of_bounds_empty(self, svc):
        assert svc.pixel_signature(_bandset(h=5, w=5), col=99, row=99) == {}
