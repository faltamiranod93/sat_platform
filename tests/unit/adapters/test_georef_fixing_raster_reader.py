"""Unit tests para GeorefFixingRasterReader (decorator de georef al vuelo)."""
import numpy as np
import pytest

from satplatform.adapters.georef_fixing_raster_reader import GeorefFixingRasterReader
from satplatform.contracts.geo import CRSRef, GeoProfile, GeoRaster
from satplatform.services.georef_fix_service import GeorefFixService


# Reutiliza el patrón de fakes/perfiles del test de GeorefFixService.
class _FakeCrsTransform:
    KNOWN_IN = (-69.2015622401477, -24.3581726864337)
    KNOWN_OUT = (479556.00, 7306103.00)

    def transform_xy(self, x, y, src_epsg, dst_epsg):
        if (round(x, 9), round(y, 9)) == (round(self.KNOWN_IN[0], 9), round(self.KNOWN_IN[1], 9)):
            return self.KNOWN_OUT
        return (x + 500000.0, y + 7000000.0)


def _broken_profile() -> GeoProfile:
    lon, lat = _FakeCrsTransform.KNOWN_IN
    return GeoProfile(
        count=1, dtype="uint16", width=4, height=3,
        transform=(lon, 10.0, 0.0, lat, 0.0, -10.0),
        crs=CRSRef.from_epsg(4326), nodata=None,
    )


def _good_profile() -> GeoProfile:
    return GeoProfile(
        count=1, dtype="uint16", width=4, height=3,
        transform=(479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


class _FakeReader:
    """Reader base fake: devuelve un GeoRaster con el profile dado."""

    def __init__(self, profile):
        self._profile = profile
        self._data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        self.read_calls = []
        self.size_calls = 0
        self.exists_calls = 0

    def read(self, uri, band_index=None):
        self.read_calls.append((uri, band_index))
        return GeoRaster(data=self._data, profile=self._profile)

    def profile(self, uri):
        return self._profile

    def size(self, uri):
        self.size_calls += 1
        return (self._profile.width, self._profile.height)

    def exists(self, uri):
        self.exists_calls += 1
        return True


@pytest.fixture
def fix_service():
    return GeorefFixService(crs_transform=_FakeCrsTransform())


class TestRead:
    def test_corrects_broken_profile(self, fix_service):
        base = _FakeReader(_broken_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        out = dec.read("scene.tif", band_index=2)
        assert out.profile.crs.epsg == 32719
        assert out.profile.transform[0] == pytest.approx(479556.0)
        assert out.profile.transform[3] == pytest.approx(7306103.0)
        # delega band_index al base
        assert base.read_calls == [("scene.tif", 2)]

    def test_preserves_data(self, fix_service):
        base = _FakeReader(_broken_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        out = dec.read("scene.tif")
        np.testing.assert_array_equal(out.data, base._data)

    def test_passthrough_on_correct_profile(self, fix_service):
        base = _FakeReader(_good_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        out = dec.read("scene.tif")
        # mismo profile (no se reconstruye)
        assert out.profile is base._profile


class TestProfileSizeExists:
    def test_profile_fixed(self, fix_service):
        base = _FakeReader(_broken_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        assert dec.profile("scene.tif").crs.epsg == 32719

    def test_size_delegates(self, fix_service):
        base = _FakeReader(_broken_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        assert dec.size("scene.tif") == (4, 3)
        assert base.size_calls == 1

    def test_exists_delegates(self, fix_service):
        base = _FakeReader(_broken_profile())
        dec = GeorefFixingRasterReader(base=base, fix_service=fix_service)
        assert dec.exists("scene.tif") is True
        assert base.exists_calls == 1
