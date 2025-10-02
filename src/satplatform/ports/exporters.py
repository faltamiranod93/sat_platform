# src/satplatform/ports/exporters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Mapping, Any, Sequence, Optional
from ..contracts.geo import GeoRaster
from ..contracts.core import ClassLabel

URI = str

@dataclass(frozen=True)
class QuicklookSpec:
    """Parámetros mínimos para quicklook/preview."""
    stretch: bool = True
    percent_clip: float = 2.0   # 2-98 por defecto
    max_size_px: int = 2048     # lado mayor

@runtime_checkable
class QuicklookExporterPort(Protocol):
    def export_rgb(self, r: GeoRaster, g: GeoRaster, b: GeoRaster, out_uri: URI, spec: Optional[QuicklookSpec] = None) -> URI: ...
    def export_gray(self, x: GeoRaster, out_uri: URI, spec: Optional[QuicklookSpec] = None) -> URI: ...
    def export_classmap(self, labels: GeoRaster, classes: Sequence[ClassLabel], out_uri: URI, spec: Optional[QuicklookSpec] = None) -> URI: ...


@runtime_checkable
class ReportExporterPort(Protocol):
    """
    Genera reportes (PDF/HTML/MD) en base a contexto.
    Adapter típico: Jinja2+WeasyPrint/LaTeX.
    """
    def render(self, template_id: str, context: Mapping[str, Any], out_uri: URI) -> URI: ...

__all__ = ["QuicklookExporterPort", "ReportExporterPort", "QuicklookSpec", "URI"]
