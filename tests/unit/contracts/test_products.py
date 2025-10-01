import pytest
from satplatform.contracts.products import BandSet
from tests.factories import make_raster

def test_bandset_stack_and_resolution_ok():
    ras = {"B04": make_raster(), "B03": make_raster(), "B02": make_raster()}
    bs = BandSet(resolution_m=10, bands=ras)
    stacked = bs.stack(["B04","B03","B02"])
    assert stacked.data.shape[0] == 3

def test_bandset_resolution_mismatch_fails():
    ras = {"B04": make_raster(px=20.0)}
    with pytest.raises(ValueError):
        BandSet(resolution_m=10, bands=ras)
