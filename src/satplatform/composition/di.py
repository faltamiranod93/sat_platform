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

def build_raster_reader(*, fix_georef: bool = False, target_epsg: int = 32719):
    """Lector de raster. Con fix_georef=True lo envuelve en el decorator que
    corrige al vuelo la georef geográfica-mal-etiquetada → UTM (escenas Sentinel
    Hub). Default False para no alterar a los consumidores existentes.
    """
    from ..adapters.gdal_raster_reader import GdalRasterReader
    base = GdalRasterReader()
    if not fix_georef:
        return base
    from ..adapters.georef_fixing_raster_reader import GeorefFixingRasterReader
    return GeorefFixingRasterReader(
        base=base,
        fix_service=build_georef_fix_service(),
        target_epsg=target_epsg,
    )


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
# Classifier adapters nativos (Mahalanobis / Cosine / Euclidean)
# ---------------------------------------------------------------------------

def build_mahalanobis_classifier(
    mcal_path: Path,
    classes: "tuple[ClassLabel, ...]",
    include_hsl: bool = True,
    band_filter: "tuple[str, ...] | None" = None,
    diag_reg: float = 1e-6,
):
    """Ajusta un MahalanobisClassifierAdapter desde un CSV Mcal.

    include_hsl=True  → v9p2 (activo, usa H S L físico)
    include_hsl=False → v9   (solo bandas espectrales)
    """
    import pandas as pd
    from ..adapters.mahalanobis_classifier import (
        MahalanobisClassifierAdapter,
        DEFAULT_BAND_FILTER,
    )
    mcal_df = pd.read_csv(mcal_path)
    bf = band_filter if band_filter is not None else DEFAULT_BAND_FILTER
    return MahalanobisClassifierAdapter.fit(
        mcal_df=mcal_df,
        classes=classes,
        band_filter=bf,
        include_hsl=include_hsl,
        diag_reg=diag_reg,
    )


def build_cosine_classifier(
    mcal_path: Path,
    classes: "tuple[ClassLabel, ...]",
    band_filter: "tuple[str, ...]",
    include_hsl: bool = False,
    two_stage: bool = False,
    stage2_class_ids: "tuple[int, ...] | None" = None,
):
    """Ajusta un CosineClassifierAdapter desde un CSV Mcal.

    two_stage=False → v4
    two_stage=True  → v5.0 / v5.1
    """
    import pandas as pd
    from ..adapters.cosine_classifier import CosineClassifierAdapter
    mcal_df = pd.read_csv(mcal_path)
    s2ids: tuple[int, ...] = stage2_class_ids if stage2_class_ids is not None else (3, 4, 5)
    return CosineClassifierAdapter.fit(
        mcal_df=mcal_df,
        classes=classes,
        band_filter=band_filter,
        include_hsl=include_hsl,
        two_stage=two_stage,
        stage2_class_ids=s2ids,
    )


def build_euclidean_classifier(
    mcal_path: Path,
    classes: "tuple[ClassLabel, ...]",
    band_filter: "tuple[str, ...]",
    include_hsl: bool = False,
):
    """Ajusta un EuclideanClassifierAdapter desde un CSV Mcal (v3)."""
    import pandas as pd
    from ..adapters.euclidean_classifier import EuclideanClassifierAdapter
    mcal_df = pd.read_csv(mcal_path)
    return EuclideanClassifierAdapter.fit(
        mcal_df=mcal_df,
        classes=classes,
        band_filter=band_filter,
        include_hsl=include_hsl,
    )


def build_classmap_service_with_mahalanobis(
    settings: "Settings",
    mcal_path: Path,
    include_hsl: bool = True,
):
    """Pipeline completo usando MahalanobisClassifierAdapter (v9p2 por defecto)."""
    from ..services.classmap_service import ClassMapService
    classes = resolve_classes(settings)
    classifier = build_mahalanobis_classifier(
        mcal_path=mcal_path,
        classes=classes,
        include_hsl=include_hsl,
    )
    return ClassMapService(
        reader=build_raster_reader(),
        writer=build_raster_writer(),
        clipper=build_clipper(settings),
        preproc=build_preprocessing_adapter(),
        classifier=classifier,
        cmapper=build_class_mapper(),
    )


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


def build_mcal_georef_service():
    """McalGeorefService es puro dominio — no requiere wiring."""
    from ..services.mcal_georef_service import McalGeorefService
    return McalGeorefService()


def build_crs_transform():
    """Adapter de reproyección de coordenadas (pyproj)."""
    from ..adapters.pyproj_crs_transform import PyprojCrsTransform
    return PyprojCrsTransform()


def build_georef_fix_service():
    """GeorefFixService: corrige georef geográfica-mal-etiquetada → UTM."""
    from ..services.georef_fix_service import GeorefFixService
    return GeorefFixService(crs_transform=build_crs_transform())


def build_training_set(
    geojson_path: Path,
    scenes_glob: str,
    *,
    target_epsg: int = 32719,
):
    """Construye el training set por match fecha+ubicación.

    Re-extrae las bandas de las escenas (georef corregida al vuelo) en los puntos
    UTM del GeoJSON v7. Devuelve un TrainingSetResult (df + resumen used/omitted).
    """
    import glob as _glob
    from ..services.training_set_builder import TrainingSetBuilder, scene_index_from_uris

    reader = build_raster_reader(fix_georef=True, target_epsg=target_epsg)
    scene_uris = sorted(_glob.glob(scenes_glob))
    scene_index = scene_index_from_uris(scene_uris)
    builder = TrainingSetBuilder(reader=reader)
    return builder.build(geojson_path, scene_index)


def build_batch_classify_service(
    settings: "Settings",
    train_df,
    *,
    indices: "tuple[str, ...]" = (),
    target_epsg: int = 32719,
):
    """Entrena los 3 clasificadores (Mahalanobis/Cosine/Euclidean) desde el
    train_df y arma el BatchClassifyService con el reader decorado.
    """
    from ..adapters.mahalanobis_classifier import MahalanobisClassifierAdapter, DEFAULT_BAND_FILTER
    from ..adapters.cosine_classifier import CosineClassifierAdapter
    from ..adapters.euclidean_classifier import EuclideanClassifierAdapter
    from ..services.batch_classify_service import BatchClassifyService, ClassifierSpec

    classes = resolve_classes(settings)
    maha = MahalanobisClassifierAdapter.fit(
        train_df, classes, include_hsl=True, indices=indices
    )
    cos = CosineClassifierAdapter.fit(
        train_df, classes, band_filter=DEFAULT_BAND_FILTER, include_hsl=False, indices=indices
    )
    euc = EuclideanClassifierAdapter.fit(
        train_df, classes, band_filter=DEFAULT_BAND_FILTER, include_hsl=False, indices=indices
    )
    return BatchClassifyService(
        reader=build_raster_reader(fix_georef=True, target_epsg=target_epsg),
        writer=build_raster_writer(),
        cmapper=build_class_mapper(),
        classifiers=(
            ClassifierSpec("maha", maha),
            ClassifierSpec("cos", cos),
            ClassifierSpec("euc", euc),
        ),
        # Resolvers de ruta desde el contrato output_patterns (services no usan Settings).
        classmap_path=lambda d, c: settings.out_path("classmap", date=d, classifier=c),
        vis_path=lambda d, c: settings.out_path("classmap_vis", date=d, classifier=c),
        summary_path=lambda name: settings.out_path("compare_summary", name=name),
    )


def build_evaluation_service(seed: int = 42):
    """EvaluationService es puro dominio (Milestone 1) — no requiere wiring."""
    from ..services.evaluation_service import EvaluationService
    return EvaluationService(seed=seed)


def default_eval_configs(indices: "tuple[str, ...]" = ()):
    """Configs de evaluación: Mahalanobis producción (B02–B12 + HSL) y ablación (sin HSL)."""
    from ..services.evaluation_service import EvalConfig
    from ..adapters.mahalanobis_classifier import (
        MahalanobisClassifierAdapter,
        DEFAULT_BAND_FILTER,
    )

    def _mk(include_hsl: bool):
        def make(train_df, classes):
            return MahalanobisClassifierAdapter.fit(
                train_df, classes, band_filter=DEFAULT_BAND_FILTER,
                include_hsl=include_hsl, indices=indices,
            )
        return make

    return [EvalConfig("prod", _mk(True)), EvalConfig("ablation", _mk(False))]


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
    # Classifier adapters nativos
    "build_mahalanobis_classifier",
    "build_cosine_classifier",
    "build_euclidean_classifier",
    "build_classmap_service_with_mahalanobis",
    # Services
    "build_classmap_service",
    "build_preprocessing_service",
    "build_histogram_norm_service",
    "build_spectral_service",
    "build_training_service",
    "build_mcal_georef_service",
    "build_crs_transform",
    "build_georef_fix_service",
    "build_training_set",
    "build_batch_classify_service",
    "build_evaluation_service",
    "default_eval_configs",
]
