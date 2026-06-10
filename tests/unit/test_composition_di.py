"""Smoke tests del composition root.

Valida que el wiring devuelve servicios con sus puertos cableados.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from satplatform.composition import di
from satplatform.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, crs_out="EPSG:32719")


def test_resolve_classes_falls_back_to_defaults(settings: Settings):
    classes = di.resolve_classes(settings)
    assert len(classes) == 3
    names = {c.name for c in classes}
    assert names == {"Agua", "Relave", "Terreno"}


def test_build_classmap_service_wires_ports(settings: Settings):
    svc = di.build_classmap_service(settings)
    assert svc.reader is not None
    assert svc.writer is not None
    assert svc.classifier is not None
    assert svc.cmapper is not None
    assert svc.preproc is not None
    assert svc.clipper is not None


def test_build_preprocessing_service_wires_ports(settings: Settings):
    svc = di.build_preprocessing_service(settings)
    assert svc.reader is not None
    assert svc.writer is not None
    assert svc.preproc is not None


def test_build_histogram_norm_service_wires_ports(settings: Settings):
    svc = di.build_histogram_norm_service(settings)
    assert svc.reader is not None
    assert svc.writer is not None


def test_build_spectral_and_training_services_pure_domain():
    spec = di.build_spectral_service()
    tr = di.build_training_service()
    # Servicios puros: no requieren puertos
    assert spec is not None
    assert tr is not None


def test_pixel_classifier_uses_settings_classes(settings: Settings):
    clf = di.build_pixel_classifier(settings)
    classes = clf.classes()
    assert len(classes) == 3
    assert {c.id for c in classes} == {1, 2, 3}


def test_build_raster_reader_fix_georef_wraps_decorator():
    from satplatform.adapters.georef_fixing_raster_reader import GeorefFixingRasterReader
    from satplatform.adapters.gdal_raster_reader import GdalRasterReader
    plain = di.build_raster_reader()
    fixing = di.build_raster_reader(fix_georef=True)
    assert isinstance(plain, GdalRasterReader)
    assert isinstance(fixing, GeorefFixingRasterReader)
    assert isinstance(fixing.base, GdalRasterReader)
    assert fixing.target_epsg == 32719


def test_build_batch_classify_service_trains_three(settings: Settings):
    import numpy as np
    import pandas as pd
    # train_df sintético con las 3 clases default y las 12 bandas
    bands = ["B01", "B02", "B03", "B04", "B05", "B06",
             "B07", "B08", "B8A", "B09", "B11", "B12"]
    rng = np.random.default_rng(0)
    rows = []
    for ng, center in {1: 1000.0, 2: 5000.0, 3: 9000.0}.items():
        for _ in range(10):
            row = {"Ng": ng}
            for b in bands:
                row[b] = float(rng.normal(center, 30))
            rows.append(row)
    train_df = pd.DataFrame(rows)
    svc = di.build_batch_classify_service(settings, train_df, indices=("NDVI",))
    assert len(svc.classifiers) == 3
    assert {c.key for c in svc.classifiers} == {"maha", "cos", "euc"}
    from satplatform.adapters.georef_fixing_raster_reader import GeorefFixingRasterReader
    assert isinstance(svc.reader, GeorefFixingRasterReader)
