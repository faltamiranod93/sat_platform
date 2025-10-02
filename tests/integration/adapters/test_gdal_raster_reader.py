# tests/integration/adapters/test_gdal_raster_reader.py
import pytest
from pathlib import Path
from satplatform.adapters.gdal_raster_reader import GdalRasterReader

pytestmark = pytest.mark.gdal  # corre sólo si hay GDAL

def test_read_small_geotiff(tmp_path: Path):
    # prepara un geotiff tiny (o incluye uno en tests/data)
    path = Path(__file__).parent.parent / "data" / "tiny.tif"
    reader = GdalRasterReader()
    rast = reader.read(path)
    assert rast.profile.width > 0
    assert rast.profile.crs is not None
