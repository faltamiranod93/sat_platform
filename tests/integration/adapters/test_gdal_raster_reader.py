"""Integration test del GdalRasterReader.

Usa la cadena de fallback rasterio → GDAL → tifffile. Genera su propio fixture
TIFF para no depender de archivos versionados.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gdal


@pytest.fixture
def tiny_tif(tmp_path: Path) -> Path:
    """Crea un TIFF pequeño usando tifffile (fallback puro). Sin georreferencia."""
    tifffile = pytest.importorskip("tifffile", reason="tifffile no disponible")
    arr = np.arange(25, dtype=np.uint16).reshape(5, 5)
    out = tmp_path / "tiny.tif"
    tifffile.imwrite(out, arr)
    return out


def test_read_small_geotiff(tiny_tif: Path):
    from satplatform.adapters.gdal_raster_reader import GdalRasterReader

    reader = GdalRasterReader()
    rast = reader.read(tiny_tif)
    assert rast.profile.width == 5
    assert rast.profile.height == 5
    assert rast.data.shape == (5, 5) or rast.data.shape == (1, 5, 5)
