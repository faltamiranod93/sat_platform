# src/satplatform/composition/di.py
from __future__ import annotations
import json, yaml
from pathlib import Path

from ..config import Settings
from ..adapters.gdal_raster_reader import GdalRasterReader
from ..adapters.gdal_raster_writer import GdalRasterWriter
from ..adapters.gdalwarp_cli import GdalWarpClipper
from ..adapters.legacy_histnorm_adapter import LegacyHistNormAdapter
from ..adapters.legacy_pixelclass_adapter import LegacyPixelClassifier
from ..adapters.legacy_classmap_adapter import LegacyClassMapAdapter

from ..services.preprocessing_service import PreprocessingService
from ..services.classmap_service import ClassMapService

def load_settings(config_path: str | Path = "00-Config/settings.yaml") -> Settings:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # construye Settings; pydantic validará tipos/placeholders
    return Settings(**data)

def load_classes_json(p: str | Path = "00-Config/class_labels.json"):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_services():
    settings = load_settings()
    reader = GdalRasterReader()
    writer = GdalRasterWriter()
    clipper = GdalWarpClipper(raster_reader=reader, raster_writer=writer, gdalwarp_exe=(str(settings.gdalwarp_exe) if settings.gdalwarp_exe else None))
    preproc_adapter = LegacyHistNormAdapter()

    # Pixel-classifier y classmap adapter
    classes = load_classes_json()
    classifier = LegacyPixelClassifier(classes_def=tuple(
        __import__("satplatform.contracts.core", fromlist=["ClassLabel","RGB8","MacroClass"]).core.ClassLabel(**c)  # opcional: o construye con pydantic
        for c in classes
    ))  # si prefieres, conviértelo explícitamente

    cmapper = LegacyClassMapAdapter()

    preproc_svc = PreprocessingService(reader=reader, writer=writer, preproc=preproc_adapter, clipper=clipper, settings=settings)
    classmap_svc = ClassMapService(reader=reader, writer=writer, classifier=classifier, cmapper=cmapper, preproc=preproc_adapter, clipper=clipper, settings=settings)
    return settings, preproc_svc, classmap_svc
