"""Construye el DataFrame de entrenamiento por match fecha + ubicación.

Toma los puntos de ground truth del GeoJSON v7 (UTM_E, UTM_N, Ng, Fecha) y, para
cada fecha que tenga una escena Sentinel Hub disponible, **re-extrae los valores
de banda de esa escena** (ya con georef corregida) en las coordenadas UTM de los
puntos. Así el training sale de las mismas escenas que se van a clasificar —
consistencia radiométrica train↔inferencia.

Diseño multitemporal: las fechas sin escena se omiten (se reportan); entran
automáticamente cuando esas escenas se descarguen.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from ..ports.raster_read import RasterReaderPort, URI
from .mcal_georef_service import McalGeorefService
from .multiband_loader import (
    SENTINEL_HUB_BAND_ORDER,
    bandset_to_arrays,
    load_multiband_bandset,
)

# Fecha en el nombre Sentinel-2: ..._MSIL2A_YYYYMMDD T...
_DATE_RE = re.compile(r"MSIL2A_(\d{8})")

# Extensiones que SÍ son rasters; el resto (.aux.xml, .xml, .csv, .bak-*) se descarta.
# Las escenas Sentinel Hub se guardan sin extensión.
_RASTER_SUFFIXES = ("", ".tif", ".tiff")


def is_scene_file(uri: str) -> bool:
    """True si el uri parece un raster de escena (no un sidecar .aux.xml/.xml/.csv)."""
    return Path(uri).suffix.lower() in _RASTER_SUFFIXES


@dataclass(frozen=True)
class TrainingSetResult:
    df: pd.DataFrame                  # [Fecha, UTM_E, UTM_N, Ng, <bandas>]
    used_by_date: Dict[str, int]      # fecha → nº de muestras extraídas
    omitted_by_date: Dict[str, int]   # fecha → nº de puntos sin escena

    @property
    def n_used(self) -> int:
        return sum(self.used_by_date.values())

    @property
    def n_omitted(self) -> int:
        return sum(self.omitted_by_date.values())


def scene_index_from_uris(scene_uris: Sequence[URI]) -> Dict[str, URI]:
    """Mapea fecha 'YYYY-MM-DD' → uri parseando MSIL2A_YYYYMMDD del nombre.

    Si hay varias escenas por fecha, gana la última (orden de entrada).
    """
    index: Dict[str, URI] = {}
    for uri in scene_uris:
        if not is_scene_file(uri):
            continue
        m = _DATE_RE.search(Path(uri).name)
        if not m:
            continue
        ymd = m.group(1)
        iso = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        index[iso] = uri
    return index


@dataclass
class TrainingSetBuilder:
    reader: RasterReaderPort
    band_filter: Sequence[str] = SENTINEL_HUB_BAND_ORDER  # extraer todas las bandas

    def build(self, geojson_path: str | Path, scene_index: Mapping[str, URI]) -> TrainingSetResult:
        pts = self._load_geojson(geojson_path)
        svc = McalGeorefService()
        frames: list[pd.DataFrame] = []
        used: Dict[str, int] = {}
        omitted: Dict[str, int] = {}

        for fecha, group in pts.groupby("Fecha", sort=True):
            uri = scene_index.get(str(fecha))
            if uri is None:
                omitted[str(fecha)] = len(group)
                continue
            bandset = load_multiband_bandset(self.reader, uri)
            arrays = bandset_to_arrays(bandset)
            profile = next(iter(bandset.bands.values())).profile
            extracted = svc.extract_at_utm_points(
                group.reset_index(drop=True), profile, arrays, list(self.band_filter)
            )
            used[str(fecha)] = len(extracted)
            if len(extracted):
                frames.append(extracted)

        cols = ["Fecha", "UTM_E", "UTM_N", "Ng"] + list(self.band_filter)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        return TrainingSetResult(df=df, used_by_date=used, omitted_by_date=omitted)

    @staticmethod
    def _load_geojson(path: str | Path) -> pd.DataFrame:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = []
        for f in obj.get("features", []):
            x, y = f["geometry"]["coordinates"][:2]
            props = f.get("properties", {})
            rows.append({
                "Fecha": str(props.get("Fecha")),
                "UTM_E": float(x),
                "UTM_N": float(y),
                "Ng": int(props.get("Ng")),
            })
        return pd.DataFrame(rows)


__all__ = ["TrainingSetBuilder", "TrainingSetResult", "scene_index_from_uris", "is_scene_file"]
