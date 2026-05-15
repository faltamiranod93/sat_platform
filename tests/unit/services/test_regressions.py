"""Tests de regresión para los 3 bugs bloqueantes corregidos.

Cada test asegura que NO se reintroduzca el bug:
  - SyntaxError en preprocessing_service (módulo no importable)
  - bands.has() en legacy_pixelclass_adapter (AttributeError)
  - BandSet no importado en histogram_norm_service (NameError)
"""
from __future__ import annotations

import numpy as np
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass, RGB8
from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef
from satplatform.contracts.products import BandSet


def _ras(value=1000, w=12, h=8, dtype=np.uint16):
    prof = GeoProfile(
        count=1, dtype=str(np.dtype(dtype)), width=w, height=h,
        transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719),
    )
    arr = np.full((h, w), value, dtype=dtype)
    return GeoRaster(arr, prof)


def _ras_gradient(base=1000, scale=100, w=12, h=8, dtype=np.uint16):
    prof = GeoProfile(
        count=1, dtype=str(np.dtype(dtype)), width=w, height=h,
        transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719),
    )
    y, x = np.mgrid[0:h, 0:w]
    arr = (base + scale * (x + y)).astype(dtype)
    return GeoRaster(arr, prof)


# ---------- Regression #1: preprocessing_service importable ----------

def test_preprocessing_service_imports_cleanly():
    """Antes había un __all__ duplicado que generaba SyntaxError."""
    import importlib
    mod = importlib.import_module("satplatform.services.preprocessing_service")
    assert hasattr(mod, "PreprocessingService")
    assert hasattr(mod, "NormalizeSingleSpec")
    assert hasattr(mod, "NormalizeManySpec")
    assert "PreprocessingService" in mod.__all__


# ---------- Regression #2: legacy_pixelclass no usa bands.has() ----------

def test_legacy_pixelclass_predict_runs():
    """Antes llamaba bands.has() (método inexistente en BandSet) → AttributeError."""
    from satplatform.adapters.legacy_pixelclass_adapter import LegacyPixelClassifier

    classes = (
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA, color=RGB8(r=0, g=0, b=255)),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE, color=RGB8(r=255, g=0, b=0)),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO, color=RGB8(r=0, g=255, b=0)),
    )
    clf = LegacyPixelClassifier(classes_def=classes)

    # 4 bandas obligatorias presentes
    bs = BandSet(
        resolution_m=10,
        bands={
            "B03": _ras_gradient(base=500, scale=50),
            "B04": _ras_gradient(base=4000, scale=100),
            "B08": _ras_gradient(base=300, scale=30),
            "B11": _ras_gradient(base=5000, scale=200),
        },
    )
    result = clf.predict(bs)
    assert isinstance(result, GeoRaster)
    assert result.data.dtype == np.uint8
    assert result.profile.width == 12 and result.profile.height == 8
    # Cada píxel asignado a alguna de las 3 clases
    assert set(np.unique(result.data).tolist()).issubset({1, 2, 3})


def test_legacy_pixelclass_predict_with_subset_of_bands():
    """Con 3 bandas (mínimo requerido) también debe correr."""
    from satplatform.adapters.legacy_pixelclass_adapter import LegacyPixelClassifier

    classes = (
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA, color=RGB8(r=0, g=0, b=255)),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE, color=RGB8(r=255, g=0, b=0)),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO, color=RGB8(r=0, g=255, b=0)),
    )
    clf = LegacyPixelClassifier(classes_def=classes)

    bs = BandSet(
        resolution_m=10,
        bands={
            "B03": _ras_gradient(),
            "B04": _ras_gradient(base=4000),
            "B08": _ras_gradient(base=300),
        },
    )
    result = clf.predict(bs)
    assert isinstance(result, GeoRaster)


def test_legacy_pixelclass_fails_with_too_few_bands():
    from satplatform.adapters.legacy_pixelclass_adapter import LegacyPixelClassifier

    classes = (
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA, color=RGB8(r=0, g=0, b=255)),
    )
    clf = LegacyPixelClassifier(classes_def=classes)

    bs = BandSet(
        resolution_m=10,
        bands={"B03": _ras(), "B04": _ras()},  # solo 2 de las requeridas
    )
    with pytest.raises(ValueError, match="al menos 3 bandas"):
        clf.predict(bs)


# ---------- Regression #3: histogram_norm_service.normalize_bandset ----------

def test_histogram_normalize_bandset_runs():
    """Antes lanzaba NameError porque BandSet no estaba importado."""
    from satplatform.services.histogram_norm_service import HistogramNormService

    svc = HistogramNormService()
    bs = BandSet(
        resolution_m=10,
        bands={
            "B03": _ras_gradient(base=100, scale=20),
            "B04": _ras_gradient(base=2000, scale=80),
        },
    )
    out = svc.normalize_bandset(bs, order=("B03", "B04"))
    assert isinstance(out, BandSet)
    assert set(out.bands.keys()) == {"B03", "B04"}
    for name, ras in out.bands.items():
        assert ras.data.dtype == np.float32
        finite = np.isfinite(ras.data)
        assert finite.any(), f"{name} sin valores finitos"
        vals = ras.data[finite]
        assert (vals >= -1e-6).all() and (vals <= 1 + 1e-6).all()


def test_histogram_normalize_bandset_missing_band_raises():
    from satplatform.services.histogram_norm_service import HistogramNormService

    svc = HistogramNormService()
    bs = BandSet(resolution_m=10, bands={"B03": _ras_gradient()})
    with pytest.raises(KeyError, match="Faltan bandas"):
        svc.normalize_bandset(bs, order=("B03", "B08"))


# ---------- Regression #4: legacy_histnorm RGB→HSL preserva cromaticidad ----------

def test_legacy_histnorm_rgb_to_hsl_preserves_chromaticity():
    """Antes: cada canal se normalizaba por su propio max, destruyendo el color.
    Ahora: escalar común. Un gris (R=G=B) debe seguir siendo gris (S≈0).
    """
    from satplatform.adapters.legacy_histnorm_adapter import LegacyHistNormAdapter

    # gris uniforme: R=G=B=4000 (en escala reflectancia)
    gray = _ras(value=4000)
    adapter = LegacyHistNormAdapter()
    H, S, L = adapter.rgb_to_hsl(gray, gray, gray)

    # Para un gris perfecto: saturación ≈ 0
    assert np.all(S.data < 1e-3), f"Gris no debería tener saturación, max S = {S.data.max()}"
    # Luminosidad coherente con el valor de entrada (4000 / 4000 ≈ 1.0 tras escala)
    assert L.data.mean() > 0.5, f"L de gris brillante debería ser alto, got {L.data.mean()}"


def test_legacy_histnorm_rgb_to_hsl_red_dominant():
    """Rojo dominante (R alto, G y B bajos) → H cerca de 0 (rojo en HSL)."""
    from satplatform.adapters.legacy_histnorm_adapter import LegacyHistNormAdapter

    red = _ras(value=8000)
    low = _ras(value=500)
    adapter = LegacyHistNormAdapter()
    H, S, L = adapter.rgb_to_hsl(red, low, low)

    # Rojo dominante → saturación alta
    assert S.data.mean() > 0.3, f"S esperado > 0.3, got {S.data.mean()}"
    # H cerca de 0 o de 1 (rojo en círculo de matiz)
    h_mean = H.data.mean()
    assert h_mean < 0.1 or h_mean > 0.9, f"H esperado cerca de rojo, got {h_mean}"


def test_legacy_histnorm_rgb_to_hsl_outputs_in_range():
    """Todos los canales H,S,L deben estar en [0,1]."""
    from satplatform.adapters.legacy_histnorm_adapter import LegacyHistNormAdapter

    r = _ras_gradient(base=3000, scale=80)
    g = _ras_gradient(base=2000, scale=60)
    b = _ras_gradient(base=1000, scale=40)
    adapter = LegacyHistNormAdapter()
    H, S, L = adapter.rgb_to_hsl(r, g, b)

    for name, ch in [("H", H), ("S", S), ("L", L)]:
        finite = np.isfinite(ch.data)
        assert finite.all(), f"{name} contiene NaN/inf"
        assert ch.data.dtype == np.float32
        assert (ch.data >= -1e-6).all() and (ch.data <= 1 + 1e-6).all(), \
            f"{name} fuera de [0,1]: min={ch.data.min()}, max={ch.data.max()}"
