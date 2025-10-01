import numpy as np
from satplatform.contracts.geo import GeoProfile, CRSRef, GeoRaster

def make_profile(w=10, h=10, px=10.0, epsg=32719, dtype="uint16"):
    return GeoProfile(
        count=1, dtype=dtype, width=w, height=h,
        transform=(0.0, px, 0.0, 0.0, 0.0, -px),
        crs=CRSRef.from_epsg(epsg), nodata=None
    )

def make_raster(w=10, h=10, px=10.0, value=0, dtype=np.uint16):
    prof = make_profile(w, h, px, dtype="uint16" if dtype==np.uint16 else "float32")
    arr = np.full((h, w), value, dtype=dtype)
    return GeoRaster(data=arr, profile=prof)

def make_rgb_bandset():
    from satplatform.contracts.products import BandSet
    ras = {n: make_raster() for n in ["B04","B03","B02"]}
    return BandSet(resolution_m=10, bands=ras)
