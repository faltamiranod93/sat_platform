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


def _ras_gradient(w=16, h=12, px=10.0, base=0, scale=1, dtype=np.uint16):
    """Raster con gradiente lineal para que las percentiles no colapsen."""
    prof = GeoProfile(
        count=1, dtype="uint16", width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px), crs=CRSRef.from_epsg(32719)
    )
    y, x = np.mgrid[0:h, 0:w]
    arr = (base + scale * (x + y)).astype(dtype)
    return GeoRaster(arr, prof)


def _bandset_rgb_gradient(intensity: float = 1.0):
    """RGB con gradientes escalados — permite probar respuesta de L a cambios de brillo."""
    b04 = _ras_gradient(base=int(2000 * intensity), scale=int(50 * intensity))
    b03 = _ras_gradient(base=int(1500 * intensity), scale=int(40 * intensity))
    b02 = _ras_gradient(base=int(1000 * intensity), scale=int(30 * intensity))
    return BandSet(resolution_m=10, bands={"B04": b04, "B03": b03, "B02": b02})


def test_rgb_to_hsl_contract_and_ranges():
    svc = psvc.PreprocessingService()
    bs = _bandset_rgb_gradient(intensity=1.0)

    out = svc.rgb_to_hsl(bs, clip=(2, 98), gamma=1.0)

    assert isinstance(out, dict)
    for k in ("H", "S", "L"):
        assert k in out, f"Falta canal {k}"
        assert isinstance(out[k], GeoRaster)
        assert out[k].profile.width == bs.bands["B04"].profile.width
        assert out[k].profile.height == bs.bands["B04"].profile.height
        assert out[k].profile.crs.equals(bs.bands["B04"].profile.crs)

        arr = out[k].data
        assert arr.dtype.kind in ("f",), f"{k} debería ser float; es {arr.dtype}"
        finite = np.isfinite(arr)
        assert finite.any(), f"{k} no tiene valores finitos"
        vals = arr[finite]
        assert (vals >= -1e-6).all() and (vals <= 1 + 1e-6).all()


def test_rgb_to_hsl_gamma_changes_l():
    """Gamma != 1 debe modificar la distribución de L (corrección sRGB-like, inv_gamma)."""
    svc = psvc.PreprocessingService()
    bs = _bandset_rgb_gradient(intensity=1.0)
    out = svc.rgb_to_hsl(bs, clip=(2, 98), gamma=1.0)
    out2 = svc.rgb_to_hsl(bs, clip=(2, 98), gamma=2.2)

    l_mid = np.nanmedian(out["L"].data)
    l2_mid = np.nanmedian(out2["L"].data)
    assert abs(l2_mid - l_mid) > 1e-3, f"gamma=2.2 no cambió L (mid {l_mid} vs {l2_mid})"
    # Convención: el código aplica inv_gamma = 1/gamma → gamma>1 sube L (sRGB encoding)
    assert l2_mid > l_mid, f"gamma=2.2 debería subir L (inv_gamma<1); got {l2_mid} vs {l_mid}"


def test_rgb_to_hsl_requires_bands():
    svc = psvc.PreprocessingService()
    bs_bad = BandSet(resolution_m=10, bands={"B04": _ras(), "B02": _ras()})
    with pytest.raises((KeyError, ValueError)):
        svc.rgb_to_hsl(bs_bad, clip=(2, 98), gamma=1.0)
