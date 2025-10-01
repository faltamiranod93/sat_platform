import numpy as np
import pytest
from satplatform.contracts.geo import GeoProfile, CRSRef, GeoRaster, validate_profile_compat

def test_georaster_immutable_buffer():
    p = GeoProfile(count=1, dtype="uint16", width=4, height=3,
                   transform=(0,10,0,0,0,-10), crs=CRSRef.from_epsg(32719))
    r = GeoRaster(np.zeros((3,4), dtype=np.uint16), p)
    with pytest.raises((ValueError, RuntimeError)):
        r.data[...] = 1

def test_validate_profile_tolerance():
    a = GeoProfile(1,"uint16",2,2,(0,10,0,0,0,-10),CRSRef.from_epsg(32719))
    b = GeoProfile(1,"uint16",2,2,(1e-7,10,0,0,0,-10),CRSRef.from_epsg(32719))
    validate_profile_compat(a,b)  # no lanza
