"""Unit tests para multiband_loader (sin I/O real)."""
import numpy as np
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.services.multiband_loader import (
    SENTINEL_HUB_BAND_ORDER,
    bandset_to_arrays,
    load_multiband_bandset,
)


def _profile():
    return GeoProfile(
        count=1, dtype="uint16", width=4, height=3,
        transform=(479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


class _FakeReader:
    """Devuelve un array constante = band_index, para verificar el mapeo posición→nombre."""

    def __init__(self):
        self.calls = []

    def read(self, uri, band_index=None):
        self.calls.append(band_index)
        data = np.full((3, 4), band_index, dtype=np.uint16)
        return GeoRaster(data=data, profile=_profile())

    def profile(self, uri):
        return _profile()

    def size(self, uri):
        return (4, 3)

    def exists(self, uri):
        return True


class TestBandOrder:
    def test_order_indices_are_one_based_and_positional(self):
        # El bug clásico: B8A y B11 NO son las posiciones 11/12.
        order = SENTINEL_HUB_BAND_ORDER
        assert order.index("B01") + 1 == 1
        assert order.index("B08") + 1 == 8
        assert order.index("B8A") + 1 == 9   # ← B8A es la 9, no la 10
        assert order.index("B09") + 1 == 10
        assert order.index("B11") + 1 == 11  # ← B11 es la 11, no la 12
        assert order.index("B12") + 1 == 12


class TestLoad:
    def test_reads_all_12_bands(self):
        reader = _FakeReader()
        bs = load_multiband_bandset(reader, "scene.tif")
        assert len(bs.bands) == 12
        assert reader.calls == list(range(1, 13))

    def test_band_value_matches_position(self):
        reader = _FakeReader()
        bs = load_multiband_bandset(reader, "scene.tif")
        # B8A leído con band_index=9 → array constante 9
        assert bs.bands["B8A"].data[0, 0] == 9
        assert bs.bands["B11"].data[0, 0] == 11

    def test_wanted_filters_bands(self):
        reader = _FakeReader()
        bs = load_multiband_bandset(reader, "scene.tif", wanted=("B04", "B03", "B02"))
        assert set(bs.bands.keys()) == {"B04", "B03", "B02"}
        # solo 3 lecturas, con los índices correctos
        assert sorted(reader.calls) == [2, 3, 4]

    def test_unknown_band_raises(self):
        reader = _FakeReader()
        with pytest.raises(KeyError):
            load_multiband_bandset(reader, "scene.tif", wanted=("B99",))

    def test_bandset_to_arrays(self):
        reader = _FakeReader()
        bs = load_multiband_bandset(reader, "scene.tif", wanted=("B04", "B08"))
        arrs = bandset_to_arrays(bs)
        assert set(arrs.keys()) == {"B04", "B08"}
        assert arrs["B04"].shape == (3, 4)
        assert arrs["B08"][0, 0] == 8
