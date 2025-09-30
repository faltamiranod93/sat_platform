# src/satplatform/ports/catalog.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence, Protocol, runtime_checkable, Optional

from ..contracts.core import SceneId, S2BandName
from ..contracts.products import S2Asset
from ..contracts.geo import CRSRef, Bounds

URI = str

@dataclass(frozen=True)
class CatalogItem:
    """Item del catálogo con metadatos + URIs por banda."""
    asset: S2Asset
    band_uris: Mapping[S2BandName, URI]  # p.ej. GeoTIFFs por banda
    thumbnail_uri: Optional[URI] = None  # opcional (quicklook/thumbnail)

@dataclass(frozen=True)
class CatalogQuery:
    date_from: date
    date_to: date
    tile: Optional[str] = None          # MGRS (e.g., "19HFE")
    cloud_max: Optional[float] = None   # 0..100
    crs: Optional[CRSRef] = None
    bounds: Optional[Bounds] = None     # filtrar por AOI

@runtime_checkable
class S2CatalogPort(Protocol):
    """Descubrimiento/búsqueda de escenas y URIs de bandas."""
    def find(self, q: CatalogQuery) -> Sequence[CatalogItem]: ...
    def get(self, scene: SceneId) -> Optional[CatalogItem]: ...
    def ping(self) -> bool: ...  # salud del backend (STAC/API/FS)

__all__ = ["S2CatalogPort", "CatalogItem", "CatalogQuery", "URI"]
