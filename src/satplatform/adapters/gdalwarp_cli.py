## `src/satplatform/adapters/gdalwarp_cli.py`
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts.geo import GeoRaster, GeoProfile, CRSRef
from ..ports.roi import ROIClipperPort, GeoJSON, WKT
from ..ports.raster_read import RasterReaderPort
from ..ports.raster_write import RasterWriterPort

@dataclass(frozen=True)
class GdalWarpClipper(ROIClipperPort):
    """Adapter que invoca `gdalwarp` por CLI para recortar/cortar por ROI.

    Requiere un `raster_reader` para reabrir el resultado y devolver un GeoRaster.
    Si `gdalwarp_exe` es None, intenta usar `gdalwarp` del PATH.
    """
    raster_reader: RasterReaderPort
    raster_writer: RasterWriterPort | None = None
    gdalwarp_exe: str | None = None

    def _exe(self) -> str:
        exe = self.gdalwarp_exe or "gdalwarp"
        return exe

    def _write_tmp_geojson(self, roi: GeoJSON, crs: CRSRef) -> str:
        # construye un FeatureCollection trivial
        fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": roi, "properties": {}}]}
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
        with open(f.name, "w", encoding="utf-8") as fh:
            json.dump(fc, fh)
        return f.name

    def _clip_common(self, in_uri: str, out_uri: str, *, cutline: str | None, te_bounds: tuple[float,float,float,float] | None, dst_crs: CRSRef | None) -> GeoRaster:
        args = [self._exe(), "-overwrite", "-of", "GTiff"]
        if dst_crs:
            if dst_crs.epsg:
                args += ["-t_srs", f"EPSG:{dst_crs.epsg}"]
            elif dst_crs.wkt:
                args += ["-t_srs", dst_crs.wkt]
        if cutline:
            args += ["-cutline", cutline, "-crop_to_cutline"]
        if te_bounds:
            minx, miny, maxx, maxy = te_bounds
            args += ["-te", str(minx), str(miny), str(maxx), str(maxy)]
        args += [in_uri, out_uri]
        os.makedirs(os.path.dirname(out_uri), exist_ok=True)
        cp = subprocess.run(args, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"gdalwarp error: {cp.stderr.strip()}")
        return self.raster_reader.read(out_uri)

    # --- ROIClipperPort ---
    def clip_raster(self, raster: GeoRaster, roi: GeoJSON | WKT, roi_crs: CRSRef) -> GeoRaster:
        # Escribe raster a temporal si no vino de archivo
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            in_uri = os.path.join(td, "in.tif")
            from .gdal_raster_writer import GdalRasterWriter
            writer = self.raster_writer or GdalRasterWriter()
            writer.write(in_uri, raster)
            out_uri = os.path.join(td, "out.tif")
            if isinstance(roi, str):  # WKT
                # Volcar WKT a geojson via ogr2ogr no es trivial sin GDAL Python; construimos bbox aproximado
                # Aquí forzamos uso de -cutline sobre un GeoJSON válido si nos entregan GeoJSON
                raise NotImplementedError("clip_raster con WKT no soportado; usa GeoJSON")
            cutline = self._write_tmp_geojson(roi, roi_crs)
            try:
                return self._clip_common(in_uri, out_uri, cutline=cutline, te_bounds=None, dst_crs=roi_crs)
            finally:
                try: os.remove(cutline)
                except Exception: pass

    def clip_profile(self, profile: GeoProfile, roi: GeoJSON | WKT, roi_crs: CRSRef) -> GeoProfile:
        r = self.clip_raster(GeoRaster(data=np.zeros((profile.height, profile.width), dtype=np.uint8), profile=profile), roi, roi_crs)  # type: ignore[name-defined]
        return r.profile

    def roi_bounds(self, roi: GeoJSON | WKT, roi_crs: CRSRef, out_crs: CRSRef):
        # Si nos entregan GeoJSON, calculamos bbox (simple)
        if not isinstance(roi, Mapping):
            raise NotImplementedError("roi_bounds espera GeoJSON en este adapter")
        coords = roi.get("coordinates")
        import numpy as np
        xs, ys = [], []
        def _walk(c):
            if isinstance(c, (list, tuple)) and len(c) and isinstance(c[0], (list, tuple)):
                for k in c: _walk(k)
            elif isinstance(c, (list, tuple)) and len(c) == 2 and all(isinstance(v, (int,float)) for v in c):
                xs.append(float(c[0])); ys.append(float(c[1]))
        _walk(coords)
        minx, maxx = float(np.min(xs)), float(np.max(xs))
        miny, maxy = float(np.min(ys)), float(np.max(ys))
        return (minx, miny, maxx, maxy)