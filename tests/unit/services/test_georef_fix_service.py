"""Unit tests para GeorefFixService (dominio puro, sin pyproj ni I/O)."""
import pytest

from satplatform.contracts.geo import CRSRef, GeoProfile, geotransform_bounds, world_to_pixel
from satplatform.services.georef_fix_service import GeorefFixService


# ---------------------------------------------------------------------------
# Fake CrsTransformPort — mapeo lon/lat→UTM determinista, sin pyproj.
# ---------------------------------------------------------------------------

class _FakeCrsTransform:
    """Conversión fija basada en el caso real de Laguna Seca.

    Mapea el origen geográfico del TIFF Sentinel Hub a su equivalente UTM 19S
    medido con pyproj (E=479556.00, N=7306103.00). Cualquier otra entrada se
    desplaza de forma determinista para poder verificar el cableado.
    """

    KNOWN_IN = (-69.2015622401477, -24.3581726864337)
    KNOWN_OUT = (479556.00, 7306103.00)

    def __init__(self):
        self.calls = []

    def transform_xy(self, x, y, src_epsg, dst_epsg):
        self.calls.append((x, y, src_epsg, dst_epsg))
        if (round(x, 9), round(y, 9)) == (round(self.KNOWN_IN[0], 9), round(self.KNOWN_IN[1], 9)):
            return self.KNOWN_OUT
        # Fallback determinista: traslada al cuadrante UTM 19S.
        return (x + 500000.0, y + 7000000.0)


def _broken_profile(px: float = 10.0, w: int = 1426, h: int = 1587) -> GeoProfile:
    """Perfil con el bug: CRS 4326 declarado pero origen lon/lat y píxel métrico."""
    lon, lat = _FakeCrsTransform.KNOWN_IN
    return GeoProfile(
        count=12, dtype="uint16", width=w, height=h,
        transform=(lon, px, 0.0, lat, 0.0, -px),
        crs=CRSRef.from_epsg(4326), nodata=None,
    )


def _good_profile() -> GeoProfile:
    """Perfil UTM 19S correcto (no necesita corrección)."""
    return GeoProfile(
        count=12, dtype="uint16", width=1426, height=1587,
        transform=(479556.0, 10.0, 0.0, 7306103.0, 0.0, -10.0),
        crs=CRSRef.from_epsg(32719), nodata=None,
    )


@pytest.fixture
def svc():
    return GeorefFixService(crs_transform=_FakeCrsTransform())


# ---------------------------------------------------------------------------
# needs_fix
# ---------------------------------------------------------------------------

class TestNeedsFix:
    def test_detects_geographic_with_metric_pixel(self, svc):
        assert svc.needs_fix(_broken_profile()) is True

    def test_correct_utm_profile_not_flagged(self, svc):
        assert svc.needs_fix(_good_profile()) is False

    def test_real_geographic_pixel_not_flagged(self, svc):
        """Un raster geográfico legítimo (píxel en grados, ~1e-4) no es bug."""
        prof = GeoProfile(
            count=1, dtype="uint16", width=100, height=100,
            transform=(-69.2, 9.0e-5, 0.0, -24.3, 0.0, -9.0e-5),
            crs=CRSRef.from_epsg(4326), nodata=None,
        )
        assert svc.needs_fix(prof) is False

    def test_projected_crs_never_flagged(self, svc):
        prof = _broken_profile()
        prof = prof.with_crs(CRSRef.from_epsg(32719))
        assert svc.needs_fix(prof) is False


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------

class TestFix:
    def test_crs_set_to_target(self, svc):
        out = svc.fix(_broken_profile())
        assert out.crs.epsg == 32719

    def test_origin_converted_to_utm(self, svc):
        out = svc.fix(_broken_profile())
        x0, _, _, y0, _, _ = out.transform
        assert x0 == pytest.approx(479556.0)
        assert y0 == pytest.approx(7306103.0)

    def test_pixel_size_preserved(self, svc):
        out = svc.fix(_broken_profile(px=10.0))
        px, py = out.pixel_size()
        assert px == pytest.approx(10.0)
        assert py == pytest.approx(-10.0)

    def test_dimensions_preserved(self, svc):
        prof = _broken_profile()
        out = svc.fix(prof)
        assert (out.width, out.height) == (prof.width, prof.height)

    def test_transform_called_with_origin(self, svc):
        prof = _broken_profile()
        svc.fix(prof)
        x, y, src, dst = svc.crs_transform.calls[0]
        assert (x, y) == (prof.transform[0], prof.transform[3])
        assert (src, dst) == (4326, 32719)

    def test_custom_target_epsg(self, svc):
        out = svc.fix(_broken_profile(), target_epsg=32718)
        assert out.crs.epsg == 32718
        assert svc.crs_transform.calls[0][3] == 32718

    def test_raises_on_already_correct_profile(self, svc):
        with pytest.raises(ValueError):
            svc.fix(_good_profile())

    def test_corrected_bounds_contain_known_points(self, svc):
        """Verificación end-to-end: con el perfil corregido, puntos UTM reales
        del GeoJSON Mcal v7 caen dentro de la grilla."""
        out = svc.fix(_broken_profile())
        b = geotransform_bounds(out.transform, out.width, out.height)
        # bbox real de los 552 puntos del 2024-01-23 (EPSG:32719)
        for e, n in [(482890.0, 7303720.0), (491880.0, 7293130.0), (485870.0, 7302350.0)]:
            assert b.minx <= e <= b.maxx
            assert b.miny <= n <= b.maxy
            col, row = world_to_pixel(e, n, out.transform)
            assert 0 <= int(round(col)) < out.width
            assert 0 <= int(round(row)) < out.height


# ---------------------------------------------------------------------------
# fix_if_needed
# ---------------------------------------------------------------------------

class TestFixIfNeeded:
    def test_applies_when_broken(self, svc):
        out = svc.fix_if_needed(_broken_profile())
        assert out.crs.epsg == 32719

    def test_passthrough_when_correct(self, svc):
        prof = _good_profile()
        out = svc.fix_if_needed(prof)
        assert out is prof
