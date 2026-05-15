"""Composition root — wiring de adapters↔ports↔services.

Esta es la ÚNICA capa que conoce simultáneamente:
  - los adapters concretos (GDAL, legacy_*, csv_*)
  - los services de dominio (ClassMapService, PreprocessingService, …)
  - la configuración (Settings)

`cli.py` debe delegar aquí; los services nunca importan de aquí.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from ..config import Settings
from ..contracts.core import ClassLabel, MacroClass, RGB8


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def load_settings_from_yaml(path: Path) -> Settings:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(**data)


def load_class_labels(path: Path) -> tuple[ClassLabel, ...]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out: list[ClassLabel] = []
    for it in items:
        out.append(
            ClassLabel(
                id=int(it["id"]),
                name=str(it["name"]),
                macro=MacroClass(it["macro"]),
                color=RGB8(**it.get("color", {})),
            )
        )
    return tuple(out)


def build_settings(project_root: Path) -> Settings:
    """Carga settings.yaml y, si no hay clases, fusiona class_labels.json."""
    cfg = (project_root / "00-Config" / "settings.yaml").resolve()
    st = load_settings_from_yaml(cfg)
    if not st.classes:
        labels_json = (project_root / "00-Config" / "class_labels.json").resolve()
        if labels_json.exists():
            st = st.model_copy(update={"classes": load_class_labels(labels_json)})
    return st


def default_classes() -> tuple[ClassLabel, ...]:
    """Clases por defecto (Agua/Relave/Terreno) cuando settings.classes está vacío."""
    return (
        ClassLabel(id=1, name="Agua",    macro=MacroClass.AGUA,
                   color=RGB8(r=31, g=119, b=180)),
        ClassLabel(id=2, name="Relave",  macro=MacroClass.RELAVE,
                   color=RGB8(r=214, g=39, b=40)),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO,
                   color=RGB8(r=152, g=223, b=138)),
    )


def resolve_classes(settings: Settings) -> tuple[ClassLabel, ...]:
    return tuple(settings.classes) if settings.classes else default_classes()


# ---------------------------------------------------------------------------
# Adapters base (sin dependencia de Settings)
# ---------------------------------------------------------------------------

def build_raster_reader():
    from ..adapters.gdal_raster_reader import GdalRasterReader
    return GdalRasterReader()


def build_raster_writer():
    from ..adapters.gdal_raster_writer import GdalRasterWriter
    return GdalRasterWriter()


def build_clipper(settings: Settings):
    from ..adapters.gdalwarp_cli import GdalWarpClipper
    reader = build_raster_reader()
    writer = build_raster_writer()
    return GdalWarpClipper(
        raster_reader=reader,
        raster_writer=writer,
        gdalwarp_exe=str(settings.gdalwarp_exe) if settings.gdalwarp_exe else None,
    )


def build_preprocessing_adapter():
    from ..adapters.legacy_histnorm_adapter import LegacyHistNormAdapter
    return LegacyHistNormAdapter()


def build_pixel_classifier(settings: Settings):
    from ..adapters.legacy_pixelclass_adapter import LegacyPixelClassifier
    return LegacyPixelClassifier(classes_def=resolve_classes(settings))


def build_class_mapper():
    from ..adapters.legacy_classmap_adapter import LegacyClassMapAdapter
    return LegacyClassMapAdapter()


# ---------------------------------------------------------------------------
# Services (cableados)
# ---------------------------------------------------------------------------

def build_classmap_service(settings: Settings):
    """Pipeline LOAD→ALIGN→PRE→INFER→EXPORT con clasificador legacy."""
    from ..services.classmap_service import ClassMapService
    return ClassMapService(
        reader=build_raster_reader(),
        writer=build_raster_writer(),
        clipper=build_clipper(settings),
        preproc=build_preprocessing_adapter(),
        classifier=build_pixel_classifier(settings),
        cmapper=build_class_mapper(),
    )


def build_preprocessing_service(settings: Settings):
    from ..services.preprocessing_service import PreprocessingService
    return PreprocessingService(
        reader=build_raster_reader(),
        writer=build_raster_writer(),
        preproc=build_preprocessing_adapter(),
        clipper=build_clipper(settings),
    )


def build_histogram_norm_service(settings: Settings):
    from ..services.histogram_norm_service import HistogramNormService
    return HistogramNormService(
        reader=build_raster_reader(),
        writer=build_raster_writer(),
        clipper=build_clipper(settings),
    )


def build_spectral_service():
    """SpectralService es puro dominio — no requiere wiring."""
    from ..services.spectral_service import SpectralService
    return SpectralService()


def build_training_service():
    """TrainingService es puro dominio — no requiere wiring."""
    from ..services.training_service import TrainingService
    return TrainingService()


__all__ = [
    # Settings helpers
    "load_settings_from_yaml",
    "load_class_labels",
    "build_settings",
    "default_classes",
    "resolve_classes",
    # Adapters
    "build_raster_reader",
    "build_raster_writer",
    "build_clipper",
    "build_preprocessing_adapter",
    "build_pixel_classifier",
    "build_class_mapper",
    # Services
    "build_classmap_service",
    "build_preprocessing_service",
    "build_histogram_norm_service",
    "build_spectral_service",
    "build_training_service",
]
