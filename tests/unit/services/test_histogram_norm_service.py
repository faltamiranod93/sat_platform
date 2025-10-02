import numpy as np
import pytest

hsvc = pytest.importorskip(
    "satplatform.services.histogram_norm_service",
    reason="HistogramNormService no encontrado",
)

from satplatform.contracts.geo import GeoRaster, GeoProfile, CRSRef
from satplatform.contracts.core import S2BandName


def _ras_rand(w=32, h=24, px=10.0, seed=0) -> GeoRaster:
    rng = np.random.default_rng(seed)
    prof = GeoProfile(
        count=1, dtype="uint16", width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px), crs=CRSRef.from_epsg(32719)
    )
    arr = rng.integers(0, 10000, size=(h, w), dtype=np.uint16)
    return GeoRaster(arr, prof)


class _DummyReader:
    """Reader mínimo que entrega rasters por URI."""
    def __init__(self, rasters_by_uri: dict[str, GeoRaster]):
        self._map = dict(rasters_by_uri)

    def read(self, uri: str) -> GeoRaster:
        return self._map[uri]


def test_histogram_contract_and_stats():
    # Prepara raster y servicio con reader en memoria
    r = _ras_rand()
    reader = _DummyReader({"mem://r": r})
    svc = hsvc.HistogramNormService(reader=reader)

    # Histograma (256 bins) sin ROI
    spec = hsvc.HistSpec(bins=256)
    res = svc.histogram("mem://r", spec=spec)

    # Contrato mínimo del histograma
    assert isinstance(res.counts, np.ndarray)
    assert isinstance(res.bin_edges, np.ndarray)
    assert len(res.counts) == 256
    assert len(res.bin_edges) == 257

    # Coherencia simple de stats
    arr = r.data.astype(np.float64, copy=False)
    assert np.nanmin(arr) <= res.mean <= np.nanmax(arr)
    assert res.p_low <= res.p_high
    # Los conteos deben sumar al número de píxeles (no hay nodata en este test)
    assert int(res.counts.sum()) == arr.size


def test_percent_clip_normalize_contract_and_range():
    # Prepara tres bandas y URIs
    bmap: dict[S2BandName, GeoRaster] = {
        "B04": _ras_rand(seed=1),
        "B03": _ras_rand(seed=2),
        "B02": _ras_rand(seed=3),
    }
    uri_map = {band: f"mem://{band}" for band in bmap}
    reader = _DummyReader({uri_map[k]: v for k, v in bmap.items()})
    svc = hsvc.HistogramNormService(reader=reader)

    out_map: dict[S2BandName, GeoRaster] = {}
    for band, uri in uri_map.items():
        # Normalización por percentiles (2–98), sin escribir a disco
        spec = hsvc.PercentClipSpec(date="20240101", out_path=None, write=False)
        out_raster, out_path = svc.percent_clip_normalize(uri, spec)
        assert out_path is None  # no escritura si no se entrega path

        # Verificaciones de contrato
        assert out_raster.data.dtype.kind == "f"  # float (float32)
        finite = np.isfinite(out_raster.data)
        assert finite.any()  # hay datos válidos
        # Rango [0,1] (permitimos tolerancia numérica)
        assert np.nanmin(out_raster.data) >= -1e-6
        assert np.nanmax(out_raster.data) <= 1.0 + 1e-6

        # Perfil coherente (mismas dims y CRS; dtype puede cambiar a float32)
        src_prof = bmap[band].profile
        dst_prof = out_raster.profile
        assert dst_prof.width == src_prof.width
        assert dst_prof.height == src_prof.height
        assert dst_prof.crs.equals(src_prof.crs)

        out_map[band] = out_raster

    # Asegura que tenemos las tres bandas normalizadas
    assert set(out_map.keys()) == {"B02", "B03", "B04"}
