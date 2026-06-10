"""Unit tests para FeatureService — consistencia train↔predict del vector de features."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.contracts.products import BandSet
from satplatform.services.feature_service import FeatureService

_BANDS = ("B02", "B03", "B04", "B08", "B11")


def _profile(h, w):
    return GeoProfile(
        count=1, dtype="float32", width=w, height=h,
        transform=(479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


def _bandset_from_values(values: dict[str, float], h=2, w=3) -> BandSet:
    bands = {
        name: GeoRaster(data=np.full((h, w), v, dtype=np.float32), profile=_profile(h, w))
        for name, v in values.items()
    }
    return BandSet(resolution_m=10, bands=bands)


def _df_from_values(values: dict[str, float], n=6) -> pd.DataFrame:
    return pd.DataFrame({k: [v] * n for k, v in values.items()})


@pytest.fixture
def svc():
    return FeatureService()


class TestFeatureNames:
    def test_order_bands_hsl_indices(self, svc):
        names = svc.feature_names(("B02", "B03"), include_hsl=True, indices=("NDVI",))
        assert names == ("B02", "B03", "H", "S", "L", "NDVI")

    def test_no_hsl_no_indices(self, svc):
        assert svc.feature_names(("B04", "B08"), False, ()) == ("B04", "B08")


class TestConsistency:
    """from_bandset y from_dataframe deben dar el MISMO vector para los mismos valores."""

    def test_bands_only(self, svc):
        vals = {"B02": 1500.0, "B04": 3000.0, "B08": 4000.0}
        Xb, nb = svc.from_bandset(_bandset_from_values(vals), ("B02", "B04", "B08"))
        Xd, nd = svc.from_dataframe(_df_from_values(vals), ("B02", "B04", "B08"))
        assert nb == nd
        # todos los pixeles/filas iguales → comparar primera fila
        np.testing.assert_allclose(Xb[0], Xd[0], rtol=1e-6)

    def test_with_hsl(self, svc):
        vals = {"B02": 1500.0, "B03": 2500.0, "B04": 3500.0}
        Xb, _ = svc.from_bandset(_bandset_from_values(vals), ("B02", "B03", "B04"), include_hsl=True)
        Xd, names = svc.from_dataframe(_df_from_values(vals), ("B02", "B03", "B04"), include_hsl=True)
        assert names[-3:] == ("H", "S", "L")
        np.testing.assert_allclose(Xb[0], Xd[0], rtol=1e-6)

    def test_with_indices(self, svc):
        vals = {"B02": 1500.0, "B03": 2500.0, "B04": 3500.0, "B08": 6000.0, "B11": 4000.0}
        idx = ("NDVI", "NDWI", "MNDWI", "BSI")
        Xb, nb = svc.from_bandset(_bandset_from_values(vals), _BANDS, include_hsl=False, indices=idx)
        Xd, nd = svc.from_dataframe(_df_from_values(vals), _BANDS, include_hsl=False, indices=idx)
        assert nb == nd == _BANDS + idx
        np.testing.assert_allclose(Xb[0], Xd[0], rtol=1e-6)

    def test_ndvi_value_matches_manual(self, svc):
        # NDVI = (B08-B04)/(B08+B04) tras /10000
        vals = {"B04": 3000.0, "B08": 6000.0}
        Xd, names = svc.from_dataframe(_df_from_values(vals), ("B04",), indices=("NDVI",))
        b08, b04 = 0.6, 0.3
        expected = (b08 - b04) / (b08 + b04 + 1e-12)
        assert names == ("B04", "NDVI")
        assert Xd[0, 1] == pytest.approx(expected, rel=1e-5)

    def test_ndsi_equals_mndwi(self, svc):
        vals = {"B03": 2500.0, "B11": 4000.0}
        Xd, names = svc.from_dataframe(_df_from_values(vals), ("B03",), indices=("MNDWI", "NDSI"))
        # columnas MNDWI y NDSI deben ser idénticas
        assert names == ("B03", "MNDWI", "NDSI")
        assert Xd[0, 1] == pytest.approx(Xd[0, 2])


class TestShape:
    def test_bandset_rows_equal_pixels(self, svc):
        X, _ = svc.from_bandset(_bandset_from_values({"B04": 1.0, "B08": 2.0}, h=2, w=3), ("B04", "B08"))
        assert X.shape == (6, 2)  # 2*3 pixeles, 2 features

    def test_missing_band_for_index_raises(self, svc):
        with pytest.raises(KeyError):
            svc.from_dataframe(_df_from_values({"B04": 1.0}), ("B04",), indices=("NDVI",))  # falta B08
