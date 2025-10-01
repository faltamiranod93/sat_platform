import numpy as np
import pytest
from types import SimpleNamespace

csvc = pytest.importorskip("satplatform.services.classmap_service", reason="ClassMapService no encontrado")

from satplatform.contracts.core import CalibrationSpec, ClassLabel, MacroClass, SceneId
from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef
from satplatform.contracts.products import BandSet


# --------- utilidades sintéticas ---------
def _prof(w=20, h=15, px=10.0):
    return GeoProfile(count=1, dtype="float32", width=w, height=h,
                      transform=(0.0, px, 0.0, 0.0, 0.0, -px),
                      crs=CRSRef.from_epsg(32719))

def _geo_float(value=0.0, w=20, h=15):
    arr = np.full((h, w), float(value), dtype=np.float32)
    return GeoRaster(arr, _prof(w, h))

def _roi_mod_hsl():
    return BandSet(resolution_m=10, bands={"H": _geo_float(0.1), "S": _geo_float(0.5), "L": _geo_float(0.8)})

def _labels():
    return (
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
    )

# --------- fakes de puertos (no I/O) ---------
class FakeClassMapPort:
    def apply(self, hsl: BandSet, calib: CalibrationSpec):
        # devuelve un clasificado multiclase dummy (1 donde L>0.6, si no 2)
        L = hsl.bands["L"].data
        cm = np.where(L > 0.6, 1, 2).astype(np.uint8)
        prof = list(hsl.bands.values())[0].profile
        cm_prof = GeoProfile(count=1, dtype="uint8", width=prof.width, height=prof.height,
                             transform=prof.transform, crs=prof.crs, nodata=0)
        return GeoRaster(cm, cm_prof)

class FakePixelClassifierPort:
    def masks(self, hsl: BandSet, threshold_map):
        # genera máscaras por clase usando thresholds ficticios
        H = hsl.bands["H"].data
        prof = list(hsl.bands.values())[0].profile
        mask1 = GeoRaster((H < 0.2).astype(np.uint8), prof.with_crs(prof.crs))
        mask2 = GeoRaster((H >= 0.2).astype(np.uint8), prof.with_crs(prof.crs))
        return {1: mask1, 2: mask2}

# --------------------------------------------

def test_classmap_service_orchestrates_ports_and_returns_domain_objects(monkeypatch):
    # ensambla el servicio con puertos falsos (inyección)
    svc = csvc.ClassMapService(classmap_port=FakeClassMapPort(), pixel_port=FakePixelClassifierPort())

    roi_mod = _roi_mod_hsl()
    calib = CalibrationSpec(schema_version="1.0.0", ref_date=SceneId(date=pytest.datetime.date(2025,1,1)).date())  # ref simple
    labels = _labels()

    result = svc.run(roi_mod=roi_mod, calibration=calib, labels=labels)

    # contrato mínimo: dict con classmap y opcionalmente masks/metrics/meta
    assert isinstance(result, dict)
    assert "classmap" in result
    classmap = result["classmap"]
    assert isinstance(classmap, GeoRaster)
    assert classmap.profile.crs.equals(list(roi_mod.bands.values())[0].profile.crs)
    assert classmap.data.dtype in (np.uint8, np.int16, np.int32)

    # máscaras por clase (si se exponen)
    if "masks" in result and result["masks"] is not None:
        masks = result["masks"]
        assert set(masks.keys()) == {1, 2}
        for mid, mras in masks.items():
            assert isinstance(mras, GeoRaster)
            assert mras.profile.width == classmap.profile.width
            assert mras.data.dtype in (np.uint8, np.bool_)

    # métricas opcionales: si hay, deben ser números reales
    if "metrics" in result and result["metrics"] is not None:
        metrics = result["metrics"]
        for k, v in metrics.items():
            assert isinstance(v, (int, float))
