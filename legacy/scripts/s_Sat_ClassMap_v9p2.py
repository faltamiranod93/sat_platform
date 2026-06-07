# -*- coding: utf-8 -*-
"""
s_Sat_ClassMap_v9.py
------------------------------------------------------------
Variante "v9" (derivada de ClassMap v5.x) que implementa clasificación GLOBAL
por distancia de Mahalanobis usando features = Nband_filter + H,S,L (escala física).

Objetivo:
- Reemplazar la lógica 2-etapas (coseno + euclídea 3-5) por una sola etapa global:
  clase = argmin_g d^2_Mahalanobis(x, μ_g, Σ_g^{-1})

Notas:
- Usa covarianza regularizada por shrinkage (Ledoit-Wolf) y opción de regularización diagonal.
- Mantiene outputs compatibles con v5.x: class tif + confidence tif + max_similarity tif + PNG coloreado.
- Asume imágenes ROI-MOD histogram-normalized (consistencia con tus McalHSL_mod).
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from PIL import Image
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2

# ---------------------------------------------------------
# INPUT (AJUSTA ESTO)
# ---------------------------------------------------------
name = 'Laguna-Seca'
ver = 'v9'
base_path = Path(r'C:/Users/felip/Desktop/Msc-UTFSM') / name  # <-- cambia si corresponde

archivo_mcal_mod = base_path / 'McalHSL_mod_v7_py.csv'        # <-- tu McalHSL_mod (con H,S,L)
archivo_roi_list = base_path / '02-Space-Facilities' / '04-ROI-MOD.csv'
archivo_roi_class = base_path / '02-Space-Facilities' / '05-ROI-MOD-CLASS.csv'
config_json = Path(r'C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python') / 'config_bandas_v3.json'  # <-- ajusta

# name = 'Laguna-Seca'
# ver = 'v9_modv3'
# base_path = Path(r'C:/Users/felip/Desktop/Msc-UTFSM') / name  # <-- cambia si corresponde
# 
# archivo_mcal_mod = base_path / '02-Space-Facilities/ROI-MOD-v2' / 'McalHSL_mod_temporal.csv'        # <-- tu McalHSL_mod (con H,S,L)
# archivo_roi_list = base_path / '02-Space-Facilities' / '04-ROI-MOD-v2.csv'
# archivo_roi_class = base_path / '02-Space-Facilities' / '05-ROI-MOD-CLASS.csv'
# config_json = Path(r'C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python') / 'config_bandas_v3.json'  # <-- ajusta

# Directorio de salida
class_dir = base_path / '03-Report' / '04_CLASSMAP' / ver
conf_dir = class_dir / 'confidence'
sim_dir = class_dir / 'max_similarity'
d2_dir = class_dir / 'd2_best'

# Parámetros Mahalanobis
use_rejection = False          # True para permitir "no class" (0) por umbral chi-cuadrado
reject_chi2_q = 0.995          # cuantil chi2 (si use_rejection=True)
diag_reg = 1e-6                # regularización diagonal adicional (Σ <- Σ + diag_reg * I)

# Rendimiento / robustez
dtype_work = np.float32

# ---------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------
def _require_exists(path: Path, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(f'{label} no existe: {path}')

def load_config(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    required = ['nameg', 'color', 'Nband_sort', 'Nband_filter']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f'Config incompleto. Falta(n): {missing}')
    return cfg

def load_roi_list(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Fecha' not in df.columns or 'Ruta' not in df.columns:
        raise ValueError("ROI list debe tener columnas: 'Fecha' y 'Ruta'")
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['Fecha', 'Ruta'])
    return df

def open_multiband_geotiff(path: str) -> tuple[np.ndarray, tuple, str]:
    """
    Returns:
        cube: (rows, cols, bands) float32
        geotransform
        projection_wkt
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'No se pudo abrir GeoTIFF: {path}')
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    arr = ds.ReadAsArray()  # (bands, rows, cols) o (rows, cols)
    ds = None
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # (rows, cols, bands)
    cube = np.transpose(arr, (1, 2, 0)).astype(dtype_work, copy=False)
    return cube, geotransform, projection

def rgb_to_hsl_vectorized(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rgb01: (P,3) en rango [0,1]
    Retorna h,s,l en [0,1]
    """
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    b = rgb01[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Lightness
    l = (cmax + cmin) / 2.0

    # Saturation
    s = np.zeros_like(l)
    nonzero = delta > 0
    # fórmula estándar HSL
    s[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0) + 1e-12)

    # Hue
    h = np.zeros_like(l)
    # evitar div/0
    delta_safe = delta + 1e-12
    mask_r = (cmax == r) & nonzero
    mask_g = (cmax == g) & nonzero
    mask_b = (cmax == b) & nonzero

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta_safe[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta_safe[mask_b]) + 4.0
    h = h / 6.0  # [0,1)

    return h, s, l

def build_features_from_cube(cube: np.ndarray, band_to_idx: dict, bands_filter: list[str]) -> np.ndarray:
    """
    cube: (rows, cols, bands) en escala 0..10000 aprox
    features: (P, len(bands_filter)+3) -> bandas seleccionadas + H,S,L
    """
    rows, cols, nb = cube.shape
    P = rows * cols

    # Bandas seleccionadas
    feats = []
    for bname in bands_filter:
        if bname not in band_to_idx:
            raise KeyError(f'Band {bname} no está en Nband_sort del config.')
        feats.append(cube[:, :, band_to_idx[bname]].reshape(P))

    Xb = np.stack(feats, axis=1).astype(dtype_work, copy=False)

    # HSL desde RGB (B04,B03,B02)
    for req in ('B04', 'B03', 'B02'):
        if req not in band_to_idx:
            raise KeyError(f'Band requerida para RGB/HSL no está en config: {req}')

    R = cube[:, :, band_to_idx['B04']].reshape(P)
    G = cube[:, :, band_to_idx['B03']].reshape(P)
    B = cube[:, :, band_to_idx['B02']].reshape(P)

    rgb01 = np.stack([R, G, B], axis=1) / 10000.0
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    h, s, l = rgb_to_hsl_vectorized(rgb01)
    H = (h * 360.0)
    S = (s * 100.0)
    L = (l * 100.0)
    HSL = (np.stack([H, S, L], axis=1)).astype(dtype_work, copy=False)

    X = np.concatenate([Xb, HSL], axis=1)
    return X

def fit_class_models(df_mcal: pd.DataFrame, class_ids: list[int], feature_cols: list[str]):
    """
    Para cada clase:
        - mean vector μ
        - precision matrix Σ^{-1} (LedoitWolf + diag_reg)
    """
    models = {}
    n_features = len(feature_cols)

    for cid in class_ids:
        X = df_mcal.loc[df_mcal['Ng'] == cid, feature_cols].values.astype(np.float64, copy=False)
        if X.shape[0] < (n_features + 2):
            raise ValueError(
                f'Clase {cid} tiene muy pocas muestras ({X.shape[0]}) para estimar covarianza con {n_features} features.'
            )
        lw = LedoitWolf().fit(X)
        mu = lw.location_.astype(np.float64)
        cov = lw.covariance_.astype(np.float64)

        # regularización diagonal adicional
        cov = cov + diag_reg * np.eye(n_features, dtype=np.float64)
        # invertir de forma robusta
        prec = np.linalg.pinv(cov)

        models[cid] = {'mu': mu, 'prec': prec}

    return models

def mahalanobis_d2(X: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """
    X: (P,F) float32/float64
    mu: (F,)
    prec: (F,F)
    """
    diff = X.astype(np.float64, copy=False) - mu[np.newaxis, :]
    # d^2 = diff * prec * diff^T (por fila)
    d2 = np.einsum('ij,jk,ik->i', diff, prec, diff, optimize=True)
    return d2.astype(dtype_work, copy=False)

def write_geotiff(path_out: Path, array_2d: np.ndarray, geotransform, projection, gdal_dtype, nodata=None):
    rows, cols = array_2d.shape
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(str(path_out), cols, rows, 1, gdal_dtype)
    ds_out.SetGeoTransform(geotransform)
    ds_out.SetProjection(projection)
    band = ds_out.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array_2d)
    band.FlushCache()
    ds_out.FlushCache()
    ds_out = None


def ensure_hsl_physical(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte H,S,L a escala física (H:0-360; S,L:0-100) si detecta escala 0-10000.
    No altera si ya está en escala física.
    """
    for c in ("H","S","L"):
        if c not in df.columns:
            return df
    Hmax = float(df["H"].max())
    Smax = float(df["S"].max())
    Lmax = float(df["L"].max())
    # Heurística: si H supera 360 o S/L superan 100 -> asumimos escala 0-10000 (o similar)
    if (Hmax > 360.5) or (Smax > 100.5) or (Lmax > 100.5):
        df = df.copy()
        df["H"] = df["H"] * (360.0 / 10000.0)
        df["S"] = df["S"] / 100.0
        df["L"] = df["L"] / 100.0
    return df

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    # Validaciones de inputs
    _require_exists(archivo_mcal_mod, 'McalHSL_mod')
    _require_exists(archivo_roi_list, 'ROI list')
    _require_exists(config_json, 'config JSON')

    cfg = load_config(config_json)
    # Validación de consistencia config vs IDs
    n_classes = len(cfg.get('nameg', []))
    if n_classes == 0:
        raise ValueError('config_bandas_v2.json debe incluir key "nameg" con al menos 1 clase')
    if 'color' not in cfg or len(cfg['color']) != n_classes:
        raise ValueError(f'config: "color" debe tener la misma longitud que "nameg" (esperado {n_classes})')
    color = (np.array(cfg['color'], dtype=np.float32) / 255.0)
    Nband_sort = cfg['Nband_sort']
    Nband_filter = cfg['Nband_filter']
    band_to_idx = {b: i for i, b in enumerate(Nband_sort)}

    # Features = Nband_filter + HSL
    feature_cols = list(Nband_filter) + ['H', 'S', 'L']

    # Leer Mcal
    df_mcal = pd.read_csv(archivo_mcal_mod)
    df_mcal = ensure_hsl_physical(df_mcal)
    if 'Ng' not in df_mcal.columns:
        raise ValueError("McalHSL_mod debe contener columna 'Ng'")
    # asegurar columnas
    missing_cols = [c for c in feature_cols if c not in df_mcal.columns]
    if missing_cols:
        raise ValueError(f'Faltan columnas en McalHSL_mod: {missing_cols}')

    class_ids = sorted(df_mcal['Ng'].dropna().unique().astype(int).tolist())
    # Consistencia: IDs deben coincidir con índices del config (0..n_classes-1)
    if any((cid < 0) or (cid >= n_classes) for cid in class_ids):
        raise ValueError(f'Hay Ng fuera del rango del config: {class_ids} vs 0..{n_classes-1}')
    if 0 not in class_ids:
        print('Nota: Ng=0 (no class) no está en entrenamiento; se usará solo como etiqueta de rechazo si corresponde.')
    # (opcional) reservar clase 0 como "no class" aunque no esté en el entrenamiento
    # class_ids no incluye 0 (se usa solo como rechazo si se activa)

    # Ajustar modelos por clase
    print(f'Clases entrenadas (Ng) = {class_ids}')
    models = fit_class_models(df_mcal, class_ids, feature_cols)

    # Umbral de rechazo (chi2)
    n_features = len(feature_cols)
    reject_thr = chi2.ppf(reject_chi2_q, df=n_features) if use_rejection else None
    if use_rejection:
        print(f'Rechazo activado: d2 > chi2(q={reject_chi2_q}, df={n_features}) = {reject_thr:.3f} => clase 0')

    # Leer ROI list
    df_roi_list = load_roi_list(archivo_roi_list)

    # Crear carpetas output
    class_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    d2_dir.mkdir(parents=True, exist_ok=True)

    # Georef: usar primer archivo válido
    first_file = str(df_roi_list.iloc[0]['Ruta'])
    cube0, geotransform, projection = open_multiband_geotiff(first_file)
    rows, cols, _ = cube0.shape

    start = time.time()
    data_catalog = []

    for fecha in df_roi_list['Fecha'].unique():
        row = df_roi_list[df_roi_list['Fecha'] == fecha].iloc[0]
        path_img = str(row['Ruta'])
        print(f'Procesando fecha: {fecha} | {path_img}')

        cube, geotransform, projection = open_multiband_geotiff(path_img)
        if cube.shape[0] != rows or cube.shape[1] != cols:
            # No asumo que todas las fechas tengan igual shape (evita reshape silencioso)
            rows, cols, _ = cube.shape

        X = build_features_from_cube(cube, band_to_idx, Nband_filter)  # (P,F)
        P = X.shape[0]

        # Calcular d2 por clase (loop por clases para limitar RAM)
        d2_all = np.empty((P, len(class_ids)), dtype=dtype_work)
        for k, cid in enumerate(class_ids):
            d2_all[:, k] = mahalanobis_d2(X, models[cid]['mu'], models[cid]['prec'])

        # Best/2nd best
        best_idx = np.argmin(d2_all, axis=1)
        best_d2 = d2_all[np.arange(P), best_idx]

        # second best (sin ordenar completo)
        # método: copiar best como +inf y volver a argmin
        d2_tmp = d2_all.copy()
        d2_tmp[np.arange(P), best_idx] = np.inf
        second_idx = np.argmin(d2_tmp, axis=1)
        second_d2 = d2_tmp[np.arange(P), second_idx]

        confidence = (second_d2 - best_d2).astype(dtype_work)  # mayor = más confiable
        max_similarity = np.exp(-0.5 * best_d2).astype(dtype_work)  # proxy de similitud

        # Labels: mapear índice -> Ng real
        labels = np.array([class_ids[i] for i in best_idx], dtype=np.int16)

        # Rechazo opcional
        if reject_thr is not None:
            labels[best_d2 > reject_thr] = 0

        # Reshape
        label_img = labels.reshape((rows, cols))
        conf_img = confidence.reshape((rows, cols))
        sim_img = max_similarity.reshape((rows, cols))
        d2_img = best_d2.reshape((rows, cols))

        # Guardar GeoTIFFs
        out_class_tif = class_dir / f'{fecha}_{name}_class_{ver}.tif'
        out_conf_tif = conf_dir / f'{fecha}_{name}_confidence_{ver}.tif'
        out_sim_tif = sim_dir / f'{fecha}_{name}_max_similarity_{ver}.tif'
        out_d2_tif = d2_dir / f'{fecha}_{name}_d2_best_{ver}.tif'

        write_geotiff(out_class_tif, label_img, geotransform, projection, gdal.GDT_Int16, nodata=-9999)
        write_geotiff(out_conf_tif, conf_img, geotransform, projection, gdal.GDT_Float32, nodata=None)
        write_geotiff(out_sim_tif, sim_img, geotransform, projection, gdal.GDT_Float32, nodata=None)
        write_geotiff(out_d2_tif, d2_img, geotransform, projection, gdal.GDT_Float32, nodata=None)

        # PNG coloreado
        colored = np.zeros((rows, cols, 3), dtype=np.float32)
        for cid in np.unique(label_img):
            cid = int(cid)
            if 0 <= cid < len(color):
                m = (label_img == cid)
                colored[m, 0] = color[cid, 0]
                colored[m, 1] = color[cid, 1]
                colored[m, 2] = color[cid, 2]

        out_png = class_dir / f'{fecha}_{name}_class_map_{ver}.png'
        img_save = (np.clip(colored, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_save).save(out_png)

        print(f'  -> class: {out_class_tif.name}')
        print(f'  -> conf : {out_conf_tif.name}')
        print(f'  -> sim  : {out_sim_tif.name}')
        print(f'  -> png  : {out_png.name}')

        data_catalog.append({'Fecha': fecha, 'Ver Class': ver, 'Ruta Class': str(out_class_tif)})

    # Actualizar catálogo 05-ROI-MOD-CLASS.csv (append si no existe fila)
    if Path(archivo_roi_class).exists():
        df_cat = pd.read_csv(archivo_roi_class)
    else:
        df_cat = pd.DataFrame(columns=['Fecha', 'Ver Class', 'Ruta Class'])

    df_new = pd.DataFrame(data_catalog)
    # normalizar Fecha
    df_cat['Fecha'] = pd.to_datetime(df_cat['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_new['Fecha'] = pd.to_datetime(df_new['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')

    # evitar duplicados exactos (Fecha, Ver Class)
    merged = pd.concat([df_cat, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=['Fecha', 'Ver Class'], keep='first')
    merged.to_csv(archivo_roi_class, index=False)

    elapsed = time.time() - start
    print(f'\nListo. Tiempo total: {elapsed/60:.2f} min | Output: {class_dir}')

if __name__ == '__main__':
    main()
