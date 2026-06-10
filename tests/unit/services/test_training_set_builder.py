"""Unit tests para TrainingSetBuilder (match fecha+ubicación, sin I/O raster real)."""
import json

import numpy as np
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.services.training_set_builder import (
    TrainingSetBuilder,
    scene_index_from_uris,
)

# Profile UTM 19S 10×10 px, origen conocido.
_GT = (479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0)


def _profile():
    return GeoProfile(
        count=1, dtype="uint16", width=10, height=10,
        transform=_GT, crs=CRSRef.from_epsg(32719), nodata=None,
    )


class _FakeReader:
    """Devuelve, por band_index, un array 10×10 constante = band_index."""

    def read(self, uri, band_index=None):
        data = np.full((10, 10), band_index, dtype=np.uint16)
        return GeoRaster(data=data, profile=_profile())

    def profile(self, uri):
        return _profile()

    def size(self, uri):
        return (10, 10)

    def exists(self, uri):
        return True


def _write_geojson(path):
    # 2 puntos dentro del raster en 2024-01-23, 1 punto en una fecha sin escena.
    feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [479600.0, 7306050.0]},
         "properties": {"Ng": 1, "Fecha": "2024-01-23"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [479640.0, 7306020.0]},
         "properties": {"Ng": 4, "Fecha": "2024-01-23"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [479600.0, 7306050.0]},
         "properties": {"Ng": 2, "Fecha": "2099-09-09"}},
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return path


class TestSceneIndex:
    def test_parses_date_from_name(self):
        idx = scene_index_from_uris([
            "/x/S2A_MSIL2A_20240123T143741_N0510_R096_T19KDP_20240123T201348.tif",
            "/x/S2B_MSIL2A_20250102T143739_N0511_R096_T19KDP_20250102T203919.tif",
        ])
        assert idx["2024-01-23"].endswith("20240123T201348.tif")
        assert idx["2025-01-02"].endswith("20250102T203919.tif")

    def test_ignores_unparseable(self):
        assert scene_index_from_uris(["/x/notascene.tif"]) == {}


class TestBuild:
    def test_only_dates_with_scene_used(self, tmp_path):
        gj = _write_geojson(tmp_path / "pts.geojson")
        builder = TrainingSetBuilder(reader=_FakeReader())
        res = builder.build(gj, scene_index={"2024-01-23": "scene_20240123.tif"})
        assert res.used_by_date == {"2024-01-23": 2}
        assert res.omitted_by_date == {"2099-09-09": 1}
        assert res.n_used == 2
        assert res.n_omitted == 1

    def test_extracted_columns(self, tmp_path):
        gj = _write_geojson(tmp_path / "pts.geojson")
        builder = TrainingSetBuilder(reader=_FakeReader())
        res = builder.build(gj, scene_index={"2024-01-23": "scene.tif"})
        assert len(res.df) == 2
        for col in ("Fecha", "UTM_E", "UTM_N", "Ng"):
            assert col in res.df.columns
        # bandas extraídas; B8A leída con band_index=9 → valor 9
        assert "B8A" in res.df.columns
        assert (res.df["B8A"] == 9).all()
        assert set(res.df["Ng"]) == {1, 4}

    def test_no_matching_scene_returns_empty(self, tmp_path):
        gj = _write_geojson(tmp_path / "pts.geojson")
        builder = TrainingSetBuilder(reader=_FakeReader())
        res = builder.build(gj, scene_index={})
        assert len(res.df) == 0
        assert res.n_used == 0
        assert res.n_omitted == 3
