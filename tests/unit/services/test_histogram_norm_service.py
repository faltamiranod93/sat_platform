import numpy as np
import pytest

hsvc = pytest.importorskip("satplatform.services.histogram_norm_service", reason="HistogramNormService no encontrado")

from satplatform.contracts.products import BandSet
from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef


def _ras_rand(w=32, h=24, px=10.0, seed=0):
    rng = np.random.default_rng(seed)
    prof = GeoProfile(
        count=1, dtype="uint16", width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px), crs=CRSRef.from_epsg(32719)
    )
    arr = rng.integers(0, 10000, size=(h, w), dtype=np.uint16)
    return GeoRaster(arr, prof)


def test_hist_stats_contract():
    svc = hsvc.HistogramNormService()
    r = _ras_rand()
    stats = svc.hist(r, bins=256)

    # contrato mínimo de salida de histograma
    for k in ("hist", "bin_edges", "min", "max", "mean", "std", "p2", "p98"):
        assert k in stats, f"Falta clave {k}"

    assert len(stats["hist"]) == 256
    assert len(stats["bin_edges"]) == 257
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_normalize_bandset_contract_and_range():
    svc = hsvc.HistogramNormService()
    bs = BandSet(resolution_m=10, bands={"B04": _ras_rand(seed=1), "B03": _ras_rand(seed=2), "B02": _ras_rand(seed=3)})
    norm = svc.normalize(bs, clip=(2, 98))  # -> BandSet o dict de GeoRaster

    # acepta dos variantes: devuelve BandSet o dict-like {band: GeoRaster}
    if isinstance(norm, BandSet):
        out_map = norm.bands
        prof_ref = list(bs.bands.values())[0].profile
        # perfiles iguales
        for n, ras in out_map.items():
            assert ras.profile.width == prof_ref.width
            assert ras.profile.height == prof_ref.height
            assert ras.profile.crs.equals(prof_ref.crs)
            # rango [0,1] flotante
            assert ras.data.dtype.kind == "f"
            assert (ras.data >= -1e-6).all() and (ras.data <= 1 + 1e-6).all()
    else:
        out_map = norm
        assert isinstance(out_map, dict)
        for n, ras in out_map.items():
            assert isinstance(ras, GeoRaster)
            assert ras.data.dtype.kind == "f"
            assert (ras.data >= -1e-6).all() and (ras.data <= 1 + 1e-6).all()
