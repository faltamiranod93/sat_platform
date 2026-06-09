"""Añade coordenadas UTM a McalHSL_mod_v7_py.csv.

Lee el GeoTransform del GeoTIFF de referencia (B02 10m, fecha 2020-04-03)
y convierte cada par (i, j) en coordenadas UTM EPSG:32719.

Output: McalHSL_mod_v7_py_utm.csv  (mismas filas + columnas UTM_E, UTM_N, EPSG)

Uso:
    python s_Sat_McalAddUTM.py
    python s_Sat_McalAddUTM.py --mcal <ruta_csv> --ref_tif <ruta_tif> --out <ruta_salida>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite importar desde sat-platform/src sin instalación
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from satplatform.contracts.geo import pixel_to_world

import pandas as pd

try:
    from osgeo import gdal
    _HAS_GDAL = True
except ImportError:
    _HAS_GDAL = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# ---------------------------------------------------------------------------
# Defaults (rutas relativas al repo; se sobrescriben con --args)
# ---------------------------------------------------------------------------

_LEGACY_DIR = Path(__file__).resolve().parent.parent
_MSC_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Msc-UTFSM/

DEFAULT_MCAL = _LEGACY_DIR / "data" / "Laguna-Seca" / "McalHSL_mod_v7_py.csv"
DEFAULT_REF_TIF = (
    _MSC_ROOT
    / "Laguna-Seca" / "02-Space-Facilities" / "ROI"
    / "T19KDP_20200403T143721_B02_10m_roi.tif"
)
DEFAULT_OUT = _LEGACY_DIR / "data" / "Laguna-Seca" / "McalHSL_mod_v7_py_utm.csv"


def _read_geotransform(tif_path: Path) -> tuple:
    """Lee el GeoTransform de un GeoTIFF. Intenta GDAL, luego tifffile."""
    if _HAS_GDAL:
        ds = gdal.Open(str(tif_path))
        if ds is None:
            raise FileNotFoundError(f"GDAL no pudo abrir: {tif_path}")
        gt = ds.GetGeoTransform()
        ds = None
        return gt

    if _HAS_TIFFFILE:
        import tifffile as tf
        with tf.TiffFile(str(tif_path)) as t:
            page = t.pages[0]
            tags = page.tags
            # ModelTiepointTag + ModelPixelScaleTag → construir GeoTransform
            tiepoint = tags.get("ModelTiepointTag")
            pixscale = tags.get("ModelPixelScaleTag")
            if tiepoint and pixscale:
                tp = tiepoint.value
                ps = pixscale.value
                x0 = tp[3] - tp[0] * ps[0]
                y0 = tp[4] + tp[1] * ps[1]
                return (x0, ps[0], 0.0, y0, 0.0, -ps[1])

    # Fallback: GeoTransform conocido del ROI de Laguna-Seca
    print("ADVERTENCIA: No se pudo leer el GeoTIFF. Usando GeoTransform fijo de Laguna-Seca.")
    return (482000.0, 10.0, 0.0, 7305000.0, 0.0, -10.0)


def add_utm(mcal_path: Path, ref_tif: Path, out_path: Path, epsg: int = 32719) -> None:
    print(f"Leyendo Mcal: {mcal_path}")
    df = pd.read_csv(mcal_path)
    n = len(df)
    print(f"  {n} puntos de entrenamiento, clases Ng: {sorted(df['Ng'].unique())}")

    print(f"Leyendo GeoTransform de: {ref_tif}")
    gt = _read_geotransform(ref_tif)
    print(f"  GeoTransform: {gt}")

    utm_e = []
    utm_n = []
    for _, row in df.iterrows():
        x, y = pixel_to_world(col=int(row["j"]) + 0.5, row=int(row["i"]) + 0.5, gt=gt)
        utm_e.append(x)
        utm_n.append(y)

    df["UTM_E"] = utm_e
    df["UTM_N"] = utm_n
    df["EPSG"] = epsg

    # Validación rápida
    e_min, e_max = df["UTM_E"].min(), df["UTM_E"].max()
    n_min, n_max = df["UTM_N"].min(), df["UTM_N"].max()
    print(f"  UTM_E rango: {e_min:.0f} – {e_max:.0f} m")
    print(f"  UTM_N rango: {n_min:.0f} – {n_max:.0f} m")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}  ({n} filas)")
    return df


def export_geojson(utm_df: pd.DataFrame, out_csv_path: Path, epsg: int = 32719) -> Path:
    from satplatform.services.mcal_georef_service import McalGeorefService
    svc = McalGeorefService()
    geojson_path = out_csv_path.with_suffix(".geojson")
    svc.to_geojson(utm_df, path=geojson_path, epsg=epsg)
    print(f"GeoJSON guardado: {geojson_path}")
    return geojson_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Añade UTM_E, UTM_N a un CSV Mcal.")
    parser.add_argument("--mcal",    type=Path, default=DEFAULT_MCAL)
    parser.add_argument("--ref_tif", type=Path, default=DEFAULT_REF_TIF)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epsg",    type=int,  default=32719)
    parser.add_argument("--geojson", action="store_true",
                        help="También exporta un GeoJSON con geometría + clase (sin bandas)")
    args = parser.parse_args()

    utm_df = add_utm(args.mcal, args.ref_tif, args.out, args.epsg)
    if args.geojson:
        export_geojson(utm_df, args.out, args.epsg)
