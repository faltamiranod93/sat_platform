# =============================
# FILE: src/satplatform/ports/catalog.py
# =============================
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, field_validator


class ROIItem(BaseModel):
    """Unidad lógica de ROI (región de interés) del proyecto."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    roi_id: str
    name: Optional[str] = None
    geom_path: Optional[Path] = None  # .shp/.geojson/.gpkg
    epsg: Optional[int] = None
    extras: Mapping[str, object] = {}


class MosaicItem(BaseModel):
    """Mosaicos o colecciones asociadas a un ROI o a la escena base.
    Ej.: mosaicos mensuales, por nubes, o por campaña.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mosaic_id: str
    roi_id: Optional[str] = None
    acq_date: Optional[date] = None
    product_path: Optional[Path] = None  # carpeta o archivo representativo
    crs: Optional[str] = None
    res_m: Optional[int] = None
    sensor: Optional[str] = None  # e.g., 'S2'
    cloud_pct: Optional[float] = None
    extras: Mapping[str, object] = {}


class CatalogItem(BaseModel):
    """Entrada de catálogo combinada (opcional) para consumo por servicios.
    Representa una unidad procesable (activo/raster) vinculada a ROI/Mosaico.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    roi: Optional[ROIItem] = None
    mosaic: Optional[MosaicItem] = None
    asset_path: Optional[Path] = None  # p.ej. ruta a GeoTIFF/JP2
    band: Optional[str] = None
    date: Optional[date] = None
    crs: Optional[str] = None
    res_m: Optional[int] = None
    extras: Mapping[str, object] = {}


class CatalogPort(Protocol):
    """Puerto de alto nivel para obtener ROIs, mosaicos y activos desde un origen.
    Detrás del puerto, el origen puede ser CSV, DB, API, etc.
    """

    def list_rois(self) -> Sequence[ROIItem]:
        ...

    def list_mosaics(self, roi_id: Optional[str] = None) -> Sequence[MosaicItem]:
        ...

    def iter_assets(
        self,
        roi_id: Optional[str] = None,
        mosaic_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> Iterable[CatalogItem]:
        """Itera activos (raster/productos) filtrables por ROI, mosaico y fecha."""
        ...
