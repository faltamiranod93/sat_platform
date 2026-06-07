"""
Genera, por cada fecha del ROI, un conteo de bandas dentro de μ±kσ + una vista PNG;
Mantiene trazabilidad en un CSV catálogo (05-ROI-MOD-CLASS.csv) versionado por ver.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat 
import os
import sys
import importlib
import pathlib
from pathlib import Path
from osgeo import gdal, osr
import seaborn as sns

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name = 'Laguna-Seca'
terrain = "02-BAND-SAT"
ver = 'v8'
path = os.path.join('C:/Users/felip/Desktop/Msc-UTFSM', name)

archivo_mcal = f'{path}/Mcal_py.csv'
archivo_mcal_mod = f'{path}/McalHSL_mod_v5_py.csv'
archivo_roi = f'{path}/02-Space-Facilities/00-ROI-utm.csv'
archivo_roi_list = f'{path}/02-Space-Facilities/04-ROI-MOD.csv'
archivo_roi_class = f'{path}/02-Space-Facilities/05-ROI-MOD-CLASS.csv'

# Ruta de la carpeta con las funciones
ruta_funciones = 'C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python'

#---------------------------------------------------------
# Leer funciones
#---------------------------------------------------------
# Verificar que el directorio existe
if not os.path.exists(path):
    raise FileNotFoundError(f"El directorio especificado no existe: {path}")

if not os.path.exists(ruta_funciones):
    raise FileNotFoundError(f"La carpeta de funciones no existe: {ruta_funciones}")

sys.path.append(ruta_funciones)
funciones_importadas = []

for filename in os.listdir(ruta_funciones):
    if filename.startswith('f_') and filename.endswith('.py'):
        try:
            module_name = filename[:-3]  # Eliminar ".py"
            module = importlib.import_module(module_name)

            # Iterar sobre las funciones del módulo
            for attr in dir(module):
                if not attr.startswith('__') and callable(getattr(module, attr)):
                    globals()[attr] = getattr(module, attr)
                    funciones_importadas.append(attr)
        except Exception as e:
            print(f"Error al importar funciones desde {filename}: {e}")
print(f"Funciones importadas correctamente: {funciones_importadas}")

#---------------------------------------------------------
# Ruta Inicial
#---------------------------------------------------------

# Read Data
df_roi = read_csv_file(archivo_roi,'#')
df_roi_list = read_csv_file(archivo_roi_list,'#')
df_roi_class = read_csv_file(archivo_roi_class,'#')
Mcal = read_csv_file(archivo_mcal,'#')
McalHSL_mod = read_csv_file(archivo_mcal_mod,'#')

# Extract ROI points
if df_roi is not None:
    ip1 = df_roi[df_roi['pl'] == 'p1'].index[0]
    ip2 = df_roi[df_roi['pl'] == 'p2'].index[0]
    p1 = df_roi.loc[ip1, ['xUTM', 'yUTM']].values
    p2 = df_roi.loc[ip2, ['xUTM', 'yUTM']].values

#---------------------------------------------------------
# Diccionario que almacena imágenes
#---------------------------------------------------------

# Lista para almacenar los archivos existentes
archivos_validos = []

# Iterar sobre los archivos en el directorio y verificar su existencia
for ifile in range(len(df_roi_list)):
    file_name = df_roi_list.loc[ifile, 'Ruta']
    date = df_roi_list.loc[ifile, 'Fecha']

    # Verificar si el archivo existe
    if os.path.exists(file_name):
        archivos_validos.append((date, file_name))  # Guardar fecha y ruta del archivo
    else:
        print(f"Archivo no encontrado: {file_name}")

# Verificar si hay archivos válidos
if not archivos_validos:
    raise ValueError("No se encontró ningún archivo válido en el listado.")

# Cargar solo la primera imagen válida
date, first_file = archivos_validos[0]
print(f"Cargando la primera imagen válida: {first_file} para la fecha {date}")


#---------------------------------------------------------
# Definición de Bandas y Grupos
#---------------------------------------------------------
# Fechas, Ni, Nj, Nlam ....................................................
fechas = df_roi_list['Fecha'].unique()

# Cargar la imagen
try:
    image = load_image(first_file)
    [Ni, Nj, Nl] = image.shape  # Obtener las dimensiones de la imagen
    print(f"Imagen cargada con dimensiones: {Ni}x{Nj} y {Nl} bandas.")
except Exception as e:
    print(f"Error al cargar la imagen {first_file}: {e}")
    raise

import json
os.path.join(ruta_funciones,"config_bandas.json")
with open(os.path.join(ruta_funciones,"config_bandas_v2.json"), "r") as f:
    config = json.load(f)

nameg = config["nameg"]
color = np.array(config["color"]) / 255.0  # Normalizar colores
lam = np.array(config["lam"])
lam_sorted = np.sort(lam)
Nband = config["Nband"]
Nband_sort = config["Nband_sort"]
NbandHSL = config["NbandHSL"]
NbandHSL_sort = config["NbandHSL_sort"]
Nlam = len(Nband)

Nband_filter = config.get("Nband_filter", NbandHSL_sort)
# mapa banda -> λ
band2lam = dict(zip(Nband, lam))

# lam sólo para bandas de clasificación
lam_filter = np.array([band2lam[b] for b in Nband_filter])

#---------------------------------------------------------
# Diccionario Vectorización Refle
#---------------------------------------------------------

# Crear una lista para almacenar DataFrames temporales
df_refle_list = []

# Iterar sobre cada archivo en df_roi_list
for ifile in range(len(df_roi_list)):
    file_name = df_roi_list.loc[ifile, 'Ruta']
    fecha = df_roi_list.loc[ifile, 'Fecha']  # Usar fecha directamente del dataframe

    # Verificar si el archivo existe
    if not os.path.exists(file_name):
        print(f"Archivo no encontrado: {file_name}")
        continue

    try:
        # Cargar la imagen
        image = load_image(file_name)

        # Extraer datos de reflectancia
        Nlam = image.shape[2]  # Número de bandas
        refle = image.reshape(-1, Nlam)  # Aplanar las bandas en columnas

        # Limitar valores de reflectancia
        refle = np.clip(refle, None, 10000.)

        # Normalizar RGB para HSL (Bandas B04, B03, B02)
        rgb_refle = refle[:, [3, 2, 1]] / 10000.
        hsl_refle = rgb2hsl(rgb_refle)
        hsl_refle[:, 0] = hsl_refle[:, 0] * 10000 / 360  # Escalar Hue
        hsl_refle[:, 1] = hsl_refle[:, 1] * 10000 / 100  # Escalar Saturation
        hsl_refle[:, 2] = hsl_refle[:, 2] * 10000 / 100  # Escalar Lightness

        # Combinar todas las bandas y HSL
        refle = np.hstack((refle, hsl_refle))

        # Crear DataFrame para esta fecha
        df_refle_fecha = pd.DataFrame(refle, columns=NbandHSL_sort)
        df_refle_fecha['Fecha'] = fecha

        # Agregar el DataFrame temporal a la lista
        df_refle_list.append(df_refle_fecha)

        print(f"Procesada imagen para la fecha: {fecha}")

    except Exception as e:
        print(f"Error al procesar la imagen {file_name}: {e}")

# Concatenar todos los DataFrames
df_refle = pd.concat(df_refle_list, ignore_index=True)

print("Creación de df_refle completada.")

#---------------------------------------------------------
# Matriz Correlacion por Grupos
#---------------------------------------------------------

# Convertir la columna Fecha al formato adecuado
McalHSL_mod['Fecha'] = pd.to_datetime(McalHSL_mod['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')

#---------------------------------------------------------
# Calcular MR_ref_mod (promedio espectral + HSL por grupo)

# ==== REFERENCIA ESPECTRAL: PROMEDIO Y STD POR GRUPO ====

group_ids = np.unique(McalHSL_mod["Ng"])
Ng = len(group_ids)

bands_cls = Nband_filter  # <<< bandas usadas en la clasificación

# Promedio por grupo (μ) solo en bandas de clasificación
mean_ref = (
    McalHSL_mod
    .groupby("Ng")[bands_cls]
    .mean()
    .loc[group_ids]
    .values
)  # shape (Ng, B_cls)

# Desviación estándar por grupo (σ) solo en bandas de clasificación
std_ref = (
    McalHSL_mod
    .groupby("Ng")[bands_cls]
    .std(ddof=1)
    .loc[group_ids]
    .values
)  # shape (Ng, B_cls)

# Evitar σ = 0
std_ref_safe = std_ref.copy()
std_ref_safe[std_ref_safe == 0] = 1e-6

#---------------------------------------------------------
# Iteración Clasificación
#---------------------------------------------------------    
 
 # Inicializar resultados
import time
from scipy.spatial import cKDTree

# Directorio de salida para las matrices de clasificación
class_dir = os.path.join(path, '03-Report', '04_CLASSMAP', ver)  # Ruta para guardar los mapas de clasificación
os.makedirs(class_dir, exist_ok=True)

start_time = time.time()

# Obtener geotransformación y proyección de la imagen original **fuera del bucle**
dataset_original = gdal.Open(first_file)
geotransform = dataset_original.GetGeoTransform()
projection = dataset_original.GetProjection()
dataset_original = None  # Liberar el dataset original

data = [] # Iterar por fecha

# ==== PARÁMETRO: TOLERANCIA EN SIGMAS ====
k_sigma = 0.8  # puedes exponerlo como parámetro de la función/script

# ==== LOOP POR FECHA (reemplazo del loop de distancia euclidiana) ====

for fecha in df_roi_list["Fecha"].unique():
    print(f"Procesando fecha: {fecha}...")

    # Matriz P×B con todas las bandas (mismas columnas que NbandHSL_sort)
    refle_fecha = df_refle[df_refle["Fecha"] == fecha].drop(columns=["Fecha"])
    refle_data = refle_fecha[bands_cls].values  # shape (P, B)
    P, B = refle_data.shape

    if P != Ni * Nj:
        print(f"Advertencia: P={P} != Ni*Nj={Ni*Nj}, reshape puede no coincidir")

    # Expandir dimensiones para broadcast:
    # refle_data_exp  : (P, 1, B)
    # mean_ref        : (1, Ng, B)
    # std_ref_safe    : (1, Ng, B)
    refle_data_exp = refle_data[:, None, :]       # (P, 1, B)
    mean_exp       = mean_ref[None, :, :]        # (1, Ng, B)
    std_exp        = std_ref_safe[None, :, :]    # (1, Ng, B)

    # Distancia absoluta por banda
    diff = np.abs(refle_data_exp - mean_exp)     # (P, Ng, B)

    # Condición de pertenencia banda a banda:
    # 1 si cae dentro de [μ - kσ, μ + kσ], 0 si no
    within = diff <= (k_sigma * std_exp)         # bool (P, Ng, B)

    # Vector de puntaje por grupo = número de bandas "dentro"
    scores = within.sum(axis=2)                  # (P, Ng), enteros en [0, B]

    # Grupo con mayor puntaje
    best_group_idx = np.argmax(scores, axis=1)   # (P,), índices 0..Ng-1
    max_score = scores.max(axis=1)               # (P,)

    # Inicializamos etiquetas con "grupo ganador" usando índice Ng (1..Ng)
    labels = group_ids[best_group_idx]           # mapea índice 0..Ng-1 -> Ng real

    # Pixels sin ningún hit: max_score == 0 -> clase especial 0 (no clasificado)
    labels[max_score == 0] = 0                   # 0 = "verde" luego en QGIS

    # Reshape a raster
    ico_image = labels.reshape((Ni, Nj))

    # ==== Escritura de raster ====
    output_filename_class = f"{fecha}_{name}_class_{ver}.tif"
    output_file_class = os.path.join(class_dir, output_filename_class)

    driver = gdal.GetDriverByName("GTiff")
    if not os.path.exists(output_file_class):
        dataset_class = driver.Create(output_file_class, Nj, Ni, 1, gdal.GDT_Int16)
        if geotransform:
            dataset_class.SetGeoTransform(geotransform)
        if projection:
            dataset_class.SetProjection(projection)
        dataset_class.GetRasterBand(1).WriteArray(ico_image)
        dataset_class.FlushCache()
        dataset_class = None
        print(f"Clasificación guardada en {output_file_class}")
    else:
        print(f"El archivo ya existe, se omite el guardado: {output_file_class}")

    # ==== Actualización CSV (igual lógica que antes) ====
    if not df_roi_class[
        (df_roi_class["Fecha"] == fecha) & (df_roi_class["Ver Class"] == ver)
    ].any().any():
        df_roi_class = pd.concat(
            [
                df_roi_class,
                pd.DataFrame(
                    [
                        {
                            "Fecha": fecha,
                            "Ver Class": ver,
                            "Ruta": output_file_class,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        df_roi_class.to_csv(archivo_roi_class, index=False)
        print(f"CSV actualizado: {archivo_roi_class}")

# Tiempo total
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")


#---------------------------------------------------------
# Función de debug píxel a píxel
#---------------------------------------------------------
from typing import List, Dict, Tuple, Optional
def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    if num_colors <= 0:
        return [(31, 119, 180)]
    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)
        saturation = 0.8
        value = 0.9

        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)

        if h_i == 0:
            r, g, b = value, t, p
        elif h_i == 1:
            r, g, b = q, value, p
        elif h_i == 2:
            r, g, b = p, value, t
        elif h_i == 3:
            r, g, b = p, q, value
        elif h_i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q

        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _rgba_color(i: int, num_groups: int, alpha: int = 160) -> Tuple[float, float, float, float]:
    if not hasattr(_rgba_color, 'palette_cache'):
        _rgba_color.palette_cache = {}

    if num_groups not in _rgba_color.palette_cache:
        _rgba_color.palette_cache[num_groups] = generate_color_palette(num_groups)

    palette = _rgba_color.palette_cache[num_groups]
    r, g, b = palette[i % len(palette)]
    return r / 255.0, g / 255.0, b / 255.0, alpha / 255.0

def debug_pixel(fecha, row, col, k_sigma=1.0, plot=True):
    """
    Inspecciona la clasificación de UN píxel (row, col) para una fecha dada,
    usando la lógica espectral μ ± k·σ y colores predefinidos.
    """

    # 1) Extraer todos los píxeles de esa fecha
    refle_fecha = df_refle[df_refle["Fecha"] == fecha].drop(columns=["Fecha"])
    if refle_fecha.empty:
        print(f"[debug_pixel] No hay datos para fecha={fecha}")
        return

    P, B = refle_fecha.shape
    expected_P = Ni * Nj
    if P != expected_P:
        print(f"[debug_pixel] Advertencia: P={P} != Ni*Nj={expected_P}. "
              f"El mapeo row,col -> índice lineal puede no coincidir.")
    
    # 2) Índice lineal
    if not (0 <= row < Ni and 0 <= col < Nj):
        print(f"[debug_pixel] (row, col)=({row}, {col}) fuera de rango.")
        return
    idx = row * Nj + col
    if idx >= P:
        print(f"[debug_pixel] idx={idx} >= P={P}, algo no cuadra.")
        return

    # 3) Vector espectral B-dimensional
    pix_spec = refle_fecha.iloc[idx][bands_cls].values

    # 4) μ y σ
    mean_exp = mean_ref        # (Ng, B)
    std_exp  = std_ref_safe    # (Ng, B)

    # 5) Distancias |pixel - μ|
    diff = np.abs(pix_spec[None, :] - mean_exp)

    # 6) within[g,b] ∈ {0,1}
    within = diff <= (k_sigma * std_exp)

    # 7) Puntaje por grupo
    scores = within.sum(axis=1)

    # 8) Grupo ganador
    best_group_idx = np.argmax(scores)
    max_score = scores[best_group_idx]

    if max_score == 0:
        label = 0     # verde
    else:
        label = group_ids[best_group_idx]

    # 9) Print
    print("\n================ DEBUG PIXEL ================")
    print(f"Fecha          : {fecha}")
    print(f"(row, col)     : ({row}, {col})")
    print(f"idx lineal     : {idx}")
    print(f"k_sigma        : {k_sigma}")
    print("--------------------------------------------")
    print("Vector del píxel:")
    for bname, val in zip(bands_cls, pix_spec):
        print(f"  {bname:>6} : {val:8.1f}")
    print("--------------------------------------------")
    print("Scores por grupo:")
    for i_g, g_id in enumerate(group_ids):
        print(f"  Ng={g_id:2d}  score={scores[i_g]:2d}")

    print("--------------------------------------------")
    if label == 0:
        print("Etiqueta final : 0 (PIXEL NO CLASIFICADO -> 'VERDE')")
    else:
        print(f"Etiqueta final : {label}")

    # 10) Detalle grupo ganador
    print("--------------------------------------------")
    if label != 0:
        g = best_group_idx
        print(f"Detalle grupo ganador Ng={group_ids[g]}:")
        for b_idx, bname in enumerate(bands_cls):
            mu  = mean_exp[g, b_idx]
            sig = std_exp[g, b_idx]
            lo  = mu - k_sigma * sig
            hi  = mu + k_sigma * sig
            val = pix_spec[b_idx]
            hit = within[g, b_idx]
            print(f"  {bname:>6} : val={val:8.1f} | μ={mu:8.1f} σ={sig:8.1f} "
                  f"[μ±kσ]=[{lo:8.1f}, {hi:8.1f}] hit={int(hit)}")
    else:
        print("Pixel queda como no-clasificado (no hay grupo ganador).")

    # ------------------------
    # 11) Plot con colores predefinidos
    # ------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(bands_cls))

        # Firma del píxel (negro)
        ax.plot(x, pix_spec, "-o", color="black", linewidth=2, label="Pixel")

        num_groups = Ng

        # Para cada grupo Ng, ploteamos μ y μ±kσ usando _rgba_color
        for i_g, g_id in enumerate(group_ids):
            rgba = _rgba_color(i_g, num_groups, alpha=200)  # color base del grupo
            mu = mean_exp[i_g, :]
            sig = std_exp[i_g, :]
            lo = mu - k_sigma * sig
            hi = mu + k_sigma * sig

            # Línea μ con color del grupo
            ax.plot(
                x, mu, "--", linewidth=1.8,
                color=rgba[:3], alpha=0.9,
                label=f"μ Ng={g_id}"
            )

            # Área μ±kσ con el mismo color
            ax.fill_between(
                x, lo, hi,
                color=rgba[:3], alpha=0.15
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bands_cls, rotation=45)
        ax.set_ylabel("Reflectancia / índice")
        ax.set_title(f"Pixel debug (row={row}, col={col}) - {fecha}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.show()
