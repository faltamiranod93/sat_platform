"""Unit tests para McalGeorefService (sin I/O)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, pixel_to_world, world_to_pixel
from satplatform.services.mcal_georef_service import McalGeorefService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(w: int = 10, h: int = 12, x0: float = 482000.0, y0: float = 7305000.0, px: float = 10.0) -> GeoProfile:
    """GeoProfile sintético con GeoTransform estándar norte-arriba."""
    return GeoProfile(
        count=1, dtype="uint16", width=w, height=h,
        transform=(x0, px, 0.0, y0, 0.0, -px),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


def _mcal(rows: int = 5, i_base: int = 2, j_base: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "Fecha": ["2020-04-03"] * rows,
        "i": [i_base + k for k in range(rows)],
        "j": [j_base + k for k in range(rows)],
        "Ng": [1, 2, 3, 4, 5][:rows],
        "B02": [1000] * rows,
        "B04": [2000] * rows,
    })


# ---------------------------------------------------------------------------
# add_utm
# ---------------------------------------------------------------------------

class TestAddUtm:
    def test_columns_added(self):
        svc = McalGeorefService()
        out = svc.add_utm(_mcal(), _profile())
        assert "UTM_E" in out.columns
        assert "UTM_N" in out.columns
        assert "EPSG" in out.columns

    def test_row_count_preserved(self):
        svc = McalGeorefService()
        df = _mcal(5)
        out = svc.add_utm(df, _profile())
        assert len(out) == len(df)

    def test_epsg_value(self):
        svc = McalGeorefService()
        out = svc.add_utm(_mcal(), _profile(), epsg=32719)
        assert (out["EPSG"] == 32719).all()

    def test_utm_values_correct(self):
        """UTM_E/N deben coincidir con pixel_to_world aplicado manualmente."""
        svc = McalGeorefService()
        prof = _profile(x0=482000.0, y0=7305000.0, px=10.0)
        df = _mcal(3, i_base=5, j_base=10)
        out = svc.add_utm(df, prof)

        gt = prof.transform
        for _, row in out.iterrows():
            expected_x, expected_y = pixel_to_world(col=int(row["j"]), row=int(row["i"]), gt=gt)
            assert abs(row["UTM_E"] - expected_x) < 1e-3
            assert abs(row["UTM_N"] - expected_y) < 1e-3

    def test_utm_e_in_roi_bounds(self):
        """Con ROI de Laguna-Seca (482000–492000 m E), UTM_E debe estar en ese rango."""
        svc = McalGeorefService()
        prof = _profile(w=1000, h=1200, x0=482000.0, y0=7305000.0, px=10.0)
        df = pd.DataFrame({"Fecha": ["2020-04-03"], "i": [100], "j": [200], "Ng": [1]})
        out = svc.add_utm(df, prof)
        assert 482000.0 <= out["UTM_E"].iloc[0] <= 492000.0
        assert 7293000.0 <= out["UTM_N"].iloc[0] <= 7305000.0

    def test_roundtrip_pixel_utm_pixel(self):
        """add_utm seguido de world_to_pixel debe recuperar (i, j) original."""
        svc = McalGeorefService()
        prof = _profile(x0=482000.0, y0=7305000.0, px=10.0)
        df = _mcal(5, i_base=1, j_base=2)
        out = svc.add_utm(df, prof)

        gt = prof.transform
        for _, row in out.iterrows():
            col_f, row_f = world_to_pixel(x=row["UTM_E"], y=row["UTM_N"], gt=gt)
            assert abs(col_f - row["j"]) < 0.5
            assert abs(row_f - row["i"]) < 0.5


# ---------------------------------------------------------------------------
# extract_at_utm_points
# ---------------------------------------------------------------------------

class TestExtractAtUtmPoints:

    def _band_arrays(self, h: int = 5, w: int = 5) -> dict:
        rng = np.random.default_rng(0)
        return {
            "B02": rng.integers(500, 3000, (h, w), dtype=np.uint16).astype(float),
            "B04": rng.integers(500, 3000, (h, w), dtype=np.uint16).astype(float),
        }

    def _utm_df_from_pixels(self, pixels: list[tuple[int, int]], prof: GeoProfile) -> pd.DataFrame:
        gt = prof.transform
        rows = []
        for i, j in pixels:
            x, y = pixel_to_world(col=j, row=i, gt=gt)
            rows.append({"Fecha": "2020-04-03", "UTM_E": x, "UTM_N": y, "Ng": 1})
        return pd.DataFrame(rows)

    def test_output_row_count(self):
        svc = McalGeorefService()
        prof = _profile(w=5, h=5)
        utm_df = self._utm_df_from_pixels([(0, 0), (1, 2), (3, 4)], prof)
        out = svc.extract_at_utm_points(utm_df, prof, self._band_arrays(), ["B02", "B04"])
        assert len(out) == 3

    def test_output_band_columns(self):
        svc = McalGeorefService()
        prof = _profile(w=5, h=5)
        utm_df = self._utm_df_from_pixels([(2, 2)], prof)
        out = svc.extract_at_utm_points(utm_df, prof, self._band_arrays(), ["B02", "B04"])
        assert "B02" in out.columns
        assert "B04" in out.columns

    def test_correct_spectral_values(self):
        """El valor extraído debe coincidir exactamente con el array en (i, j)."""
        svc = McalGeorefService()
        prof = _profile(w=5, h=5)
        bands = self._band_arrays()
        target_i, target_j = 2, 3
        utm_df = self._utm_df_from_pixels([(target_i, target_j)], prof)
        out = svc.extract_at_utm_points(utm_df, prof, bands, ["B02"])
        assert out["B02"].iloc[0] == bands["B02"][target_i, target_j]

    def test_out_of_bounds_discarded(self):
        """Puntos fuera del ROI deben descartarse sin excepción."""
        svc = McalGeorefService()
        prof = _profile(w=5, h=5, x0=482000.0, y0=7305000.0, px=10.0)
        out_of_bounds_utm = pd.DataFrame([{
            "Fecha": "2020-04-03",
            "UTM_E": 999999.0,   # muy lejos del ROI
            "UTM_N": 9999999.0,
            "Ng": 1,
        }])
        out = svc.extract_at_utm_points(out_of_bounds_utm, prof, self._band_arrays(), ["B02"])
        assert len(out) == 0

    def test_empty_input_returns_empty_df(self):
        svc = McalGeorefService()
        prof = _profile(w=5, h=5)
        empty = pd.DataFrame(columns=["Fecha", "UTM_E", "UTM_N", "Ng"])
        out = svc.extract_at_utm_points(empty, prof, self._band_arrays(), ["B02"])
        assert len(out) == 0
        assert "B02" in out.columns

    def test_mixed_in_and_out_of_bounds(self):
        """Solo los puntos dentro del ROI deben aparecer en el output."""
        svc = McalGeorefService()
        prof = _profile(w=5, h=5, x0=482000.0, y0=7305000.0, px=10.0)
        bands = self._band_arrays()
        in_bounds = self._utm_df_from_pixels([(1, 1), (2, 2)], prof)
        out_of_bounds = pd.DataFrame([{
            "Fecha": "2020-04-03", "UTM_E": 0.0, "UTM_N": 0.0, "Ng": 1,
        }])
        utm_df = pd.concat([in_bounds, out_of_bounds], ignore_index=True)
        out = svc.extract_at_utm_points(utm_df, prof, bands, ["B02"])
        assert len(out) == 2
