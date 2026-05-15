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
