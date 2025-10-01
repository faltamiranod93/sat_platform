import numpy as np
import pytest

psvc = pytest.importorskip("satplatform.services.preprocessing_service", reason="PreprocessingService no encontrado")

from satplatform.contracts.products import BandSet
from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef


def _ras(w=16, h=12, px=10.0, value=0, dtype=np.uint16):
    prof = GeoProfile(
        count=1, dtype="uint16", width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px), crs=CRSRef.from_epsg(32719)
    )
    arr = np.full((h, w), value, dtype=dtype)
    return GeoRaster(arr, prof)


def _bandset_rgb():
    # patrones simples para detectar cambios tras clip/gamma
    b04 = _ras(value=4000)  # R
    b03 = _ras(value=2000)  # G
    b02 = _ras(value=1000)  # B
    return BandSet(resolution_m=10, bands={"B04": b04, "B03": b03, "B02": b02})


def test_rgb_to_hsl_contract_and_ranges():
    svc = psvc.PreprocessingService()
    bs = _bandset_rgb()

    # asume API rgb_to_hsl(bandset, clip=(2,98), gamma=1.0) -> dict[str, GeoRaster]
    out = svc.rgb_to_hsl(bs, clip=(2, 98), gamma=1.0)

    # contrato mínimo
    assert isinstance(out, dict)
    for k in ("H", "S", "L"):
        assert k in out, f"Falta canal {k}"
        assert isinstance(out[k], GeoRaster)
        # mismo perfil (shape/CRS/transform)
        assert out[k].profile.width == bs.bands["B04"].profile.width
        assert out[k].profile.height == bs.bands["B04"].profile.height
        assert out[k].profile.crs.equals(bs.bands["B04"].profile.crs)

        # rango [0,1] y dtype flotante (flexible en margen)
        arr = out[k].data
        assert arr.dtype.kind in ("f",), f"{k} debería ser float; es {arr.dtype}"
        assert np.isfinite(arr).all()
        assert (arr >= -1e-6).all() and (arr <= 1 + 1e-6).all()

    # L debe responder a cambios de intensidad (heurística simple)
    # si subimos B04, L debería subir
    bs2 = BandSet(resolution_m=10, bands={"B04": _ras(value=8000), "B03": _ras(value=2000), "B02": _ras(value=1000)})
    out2 = svc.rgb_to_hsl(bs2, clip=(2, 98), gamma=1.0)
    assert out2["L"].data.mean() > out["L"].data.mean()


def test_rgb_to_hsl_requires_bands():
    svc = psvc.PreprocessingService()
    # falta B03
    bs_bad = BandSet(resolution_m=10, bands={"B04": _ras(), "B02": _ras()})
    with pytest.raises((KeyError, ValueError)):
        svc.rgb_to_hsl(bs_bad, clip=(2, 98), gamma=1.0)
