"""Servicio de georreferenciación de puntos de entrenamiento Mcal.

Permite convertir posiciones de píxel (i, j) a coordenadas UTM y viceversa,
haciendo los puntos de entrenamiento portátiles entre ROIs y computadores.

Flujo típico:
  Personal   → add_utm(mcal_df, profile_roi_personal) → CSV con UTM_E, UTM_N
  Universidad → extract_at_utm_points(utm_df, bandset_nuevo, profile_roi_univ)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..contracts.geo import GeoProfile, pixel_to_world, world_to_pixel


class McalGeorefService:
    """Convierte posiciones de píxel Mcal ↔ coordenadas UTM.

    Es un servicio de dominio puro — no hace I/O.
    """

    def add_utm(
        self,
        mcal_df: pd.DataFrame,
        geo_profile: GeoProfile,
        epsg: int = 32719,
    ) -> pd.DataFrame:
        """Añade columnas UTM_E, UTM_N, EPSG a un DataFrame Mcal.

        Args:
            mcal_df: DataFrame con columnas 'i' (fila) y 'j' (columna).
            geo_profile: GeoProfile del GeoTIFF usado como referencia.
            epsg: código EPSG del CRS de salida (default 32719 = UTM 19 Sur).

        Returns:
            Copia del DataFrame con columnas UTM_E, UTM_N, EPSG añadidas.
        """
        gt = geo_profile.transform
        utm_e = np.empty(len(mcal_df), dtype=np.float64)
        utm_n = np.empty(len(mcal_df), dtype=np.float64)

        for idx, (_, row) in enumerate(mcal_df.iterrows()):
            x, y = pixel_to_world(col=int(row["j"]), row=int(row["i"]), gt=gt)
            utm_e[idx] = x
            utm_n[idx] = y

        out = mcal_df.copy()
        out["UTM_E"] = utm_e
        out["UTM_N"] = utm_n
        out["EPSG"] = epsg
        return out

    def extract_at_utm_points(
        self,
        utm_df: pd.DataFrame,
        geo_profile: GeoProfile,
        band_arrays: dict[str, np.ndarray],
        band_filter: Sequence[str],
    ) -> pd.DataFrame:
        """Extrae valores espectrales en los puntos UTM dados usando un nuevo ROI.

        Args:
            utm_df: DataFrame con columnas 'Fecha', 'UTM_E', 'UTM_N', 'Ng'
                    (y opcionalmente otras que se conservan).
            geo_profile: GeoProfile del nuevo ROI donde extraer.
            band_arrays: dict {nombre_banda: ndarray 2D (H, W)} del nuevo ROI.
            band_filter: lista de bandas a extraer (ej. ['B02','B03','B04',...]).

        Returns:
            DataFrame con columnas: Fecha, UTM_E, UTM_N, Ng, <bandas>.
            Puntos fuera del bounds del nuevo ROI son descartados silenciosamente.
        """
        gt = geo_profile.transform
        H, W = geo_profile.height, geo_profile.width

        records = []
        for _, row in utm_df.iterrows():
            col_f, row_f = world_to_pixel(
                x=float(row["UTM_E"]),
                y=float(row["UTM_N"]),
                gt=gt,
            )
            col_i, row_i = int(round(col_f)), int(round(row_f))

            if not (0 <= row_i < H and 0 <= col_i < W):
                continue

            record: dict = {
                "Fecha": row["Fecha"],
                "UTM_E": float(row["UTM_E"]),
                "UTM_N": float(row["UTM_N"]),
                "Ng": int(row["Ng"]),
            }
            for band in band_filter:
                if band in band_arrays:
                    record[band] = float(band_arrays[band][row_i, col_i])
            records.append(record)

        if not records:
            cols = ["Fecha", "UTM_E", "UTM_N", "Ng"] + list(band_filter)
            return pd.DataFrame(columns=cols)

        return pd.DataFrame(records)


__all__ = ["McalGeorefService"]
