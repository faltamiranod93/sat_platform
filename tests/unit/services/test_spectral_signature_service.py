"""Unit tests para SpectralSignatureService (dominio puro)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.services.spectral_signature_service import SpectralSignatureService

_BANDS = ("B02", "B03", "B04", "B08", "B11")


def _df(n_per_class=30, seed=0):
    """3 clases bien separadas en el espacio espectral."""
    rng = np.random.default_rng(seed)
    rows = []
    centers = {1: 1000.0, 2: 5000.0, 3: 9000.0}
    fechas = ["20240123", "20240128"]
    for ng, c in centers.items():
        for k in range(n_per_class):
            row = {"Ng": ng, "Fecha": fechas[k % 2]}
            for b in _BANDS:
                row[b] = float(rng.normal(c, 50))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def svc():
    return SpectralSignatureService()


class TestSignaturesByClass:
    def test_shape_and_columns(self, svc):
        out = svc.signatures_by_class(_df(), _BANDS)
        assert list(out.columns) == ["class_id", "feature", "mean", "std", "n"]
        # 3 clases × 5 bandas
        assert len(out) == 15
        assert set(out["class_id"]) == {1, 2, 3}

    def test_means_track_class_center(self, svc):
        out = svc.signatures_by_class(_df(), _BANDS)
        m1 = out[(out.class_id == 1) & (out.feature == "B04")]["mean"].iloc[0]
        m3 = out[(out.class_id == 3) & (out.feature == "B04")]["mean"].iloc[0]
        assert m1 == pytest.approx(1000, abs=80)
        assert m3 == pytest.approx(9000, abs=80)

    def test_with_indices_and_hsl(self, svc):
        out = svc.signatures_by_class(_df(), _BANDS, include_hsl=True, indices=("NDVI",))
        feats = set(out["feature"])
        assert {"H", "S", "L", "NDVI"} <= feats


class TestSeparability:
    def test_symmetric_zero_diagonal(self, svc):
        M = svc.separability_matrix(_df(), _BANDS)
        assert M.shape == (3, 3)
        np.testing.assert_allclose(np.diag(M.values), 0.0, atol=1e-9)
        np.testing.assert_allclose(M.values, M.values.T, rtol=1e-9)

    def test_far_classes_more_separable(self, svc):
        M = svc.separability_matrix(_df(), _BANDS)
        # clase 1 vs 3 (centros 1000 vs 9000) más lejos que 1 vs 2 (1000 vs 5000)
        assert M.loc[1, 3] > M.loc[1, 2]

    def test_mahalanobis_metric(self, svc):
        M = svc.separability_matrix(_df(), _BANDS, metric="mahalanobis")
        assert M.shape == (3, 3)
        np.testing.assert_allclose(np.diag(M.values), 0.0, atol=1e-6)

    def test_bad_metric_raises(self, svc):
        with pytest.raises(ValueError):
            svc.separability_matrix(_df(), _BANDS, metric="coseno")


class TestPCA:
    def test_two_components(self, svc):
        out = svc.pca_2d(_df(), _BANDS)
        assert list(out.columns) == ["class_id", "pc1", "pc2"]
        assert len(out) == 90  # 3×30


class TestTemporal:
    def test_means_by_date(self, svc):
        out = svc.temporal_means(_df(), class_id=1, band_filter=_BANDS)
        assert set(out["Fecha"]) == {"20240123", "20240128"}
        assert "mean" in out.columns

    def test_missing_class_returns_empty(self, svc):
        out = svc.temporal_means(_df(), class_id=99, band_filter=_BANDS)
        assert len(out) == 0
