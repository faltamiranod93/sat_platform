"""Servicio de corrección de georreferencia de rasters mal etiquetados.

Algunos GeoTIFF de Sentinel Hub para Laguna Seca se exportaron con una
georreferencia inconsistente: declaran CRS geográfico (EPSG:4326, lon/lat)
pero su origen está en grados y el tamaño de píxel es métrico (10 m). El
raster es en realidad UTM 19 Sur (EPSG:32719) a 10 m/píxel.

Síntoma observable: al extraer en puntos UTM reales (p.ej. el GeoJSON Mcal v7,
EPSG:32719), `world_to_pixel` manda todas las coordenadas fuera de la grilla y
`extract_at_utm_points` devuelve un DataFrame vacío.

Este servicio es dominio puro: transforma un GeoProfile en otro corregido,
delegando la conversión del origen lon/lat→UTM al CrsTransformPort. No hace I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.geo import CRSRef, GeoProfile
from ..ports.crs_transform import CrsTransformPort

# EPSG de CRS geográficos donde un origen en grados es esperable.
_GEOGRAPHIC_EPSG = frozenset({4326})

# Un píxel geográfico real (en grados) es siempre << 1°. Si el raster declara
# CRS geográfico pero el píxel supera este umbral, está en metros mal etiquetados.
_DEGREE_PIXEL_MAX = 1.0


@dataclass
class GeorefFixService:
    """Corrige perfiles con CRS geográfico declarado pero grilla métrica."""

    crs_transform: CrsTransformPort

    def needs_fix(self, profile: GeoProfile) -> bool:
        """True si el perfil presenta la inconsistencia geográfico-vs-métrico."""
        epsg = profile.crs.epsg
        if epsg not in _GEOGRAPHIC_EPSG:
            return False
        px, _ = profile.pixel_size()
        return abs(px) >= _DEGREE_PIXEL_MAX

    def fix(
        self,
        profile: GeoProfile,
        target_epsg: int = 32719,
        *,
        source_epsg: int | None = None,
    ) -> GeoProfile:
        """Reescribe el origen a `target_epsg` y corrige el CRS.

        Convierte el origen (esquina superior izquierda) desde el CRS geográfico
        declarado al CRS UTM real. Conserva tamaño de píxel y rotación, y fija el
        CRS a `target_epsg`.

        Args:
            profile: GeoProfile con la georreferencia inconsistente.
            target_epsg: CRS proyectado real del raster (default 32719 = UTM 19S).
            source_epsg: CRS geográfico de origen; por defecto el declarado en el
                perfil.

        Returns:
            GeoProfile corregido (nuevo transform + CRS).

        Raises:
            ValueError: si el perfil no presenta la inconsistencia (evita
                corromper un perfil ya correcto) o si falta el EPSG de origen.
        """
        if not self.needs_fix(profile):
            raise ValueError(
                "El perfil no presenta la inconsistencia geográfico-vs-métrico; "
                "fix() abortado para no corromper una georreferencia válida."
            )

        src = source_epsg if source_epsg is not None else profile.crs.epsg
        if src is None:
            raise ValueError("No hay EPSG de origen para convertir el origen.")

        x0, px, rx, y0, ry, py = profile.transform
        east0, north0 = self.crs_transform.transform_xy(
            x0, y0, src_epsg=int(src), dst_epsg=int(target_epsg)
        )
        new_transform = (east0, px, rx, north0, ry, py)

        return profile.with_transform(new_transform).with_crs(
            CRSRef.from_epsg(int(target_epsg))
        )

    def fix_if_needed(self, profile: GeoProfile, target_epsg: int = 32719) -> GeoProfile:
        """Devuelve el perfil corregido si lo necesita; si no, lo deja igual."""
        return self.fix(profile, target_epsg) if self.needs_fix(profile) else profile


__all__ = ["GeorefFixService"]
