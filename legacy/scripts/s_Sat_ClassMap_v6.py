import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat 
#import tkinter as tk
import os
import sys
import importlib
import pathlib
from pathlib import Path
#from tkinter import filedialog
from osgeo import gdal, osr
import seaborn as sns

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name = 'Laguna-Seca'
terrain = "02-BAND-SAT"
ver = 'v6'
path = os.path.join('C:/Users/felip/Desktop/Msc-UTFSM', name)

archivo_mcal = f'{path}/Mcal_py.csv'
archivo_mcal_mod = f'{path}/McalHSL_mod_v3_py.csv'
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
with open(os.path.join(ruta_funciones,"config_bandas.json"), "r", encoding="utf-8") as f:
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

# Agregar HSL a Mcal
Rg = Mcal.reindex(columns=Nband_sort)  # Reordenar columnas de bandas
hsl_rg = pd.DataFrame(rgb2hsl(Rg.iloc[:, [3, 2, 1]] / 10000.), columns=['H', 'S', 'L'])
# Escalar los valores de H, S y L
hsl_rg['H'] = hsl_rg['H'] * (10000 / 360)
hsl_rg['S'] = hsl_rg['S'] * (10000 / 100)
hsl_rg['L'] = hsl_rg['L'] * (10000 / 100)

Mcal_HSL = Mcal.join(hsl_rg, how='outer')

# Convertir la columna Fecha al formato adecuado
Mcal_HSL['Fecha'] = pd.to_datetime(Mcal_HSL['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')

group = np.unique(Mcal["Ng"])
Ng = len(group)
MR_ref  = np.zeros((Ng,Nlam+3))

# Calcular MR_ref (promedio espectral + HSL por grupo)
for jg in range(Ng):
    aux = Mcal_HSL[Mcal_HSL['Ng'] == group[jg]][NbandHSL_sort].mean()
    MR_ref[jg, :] = np.hstack((aux.values))

# matriz para almacenar las correlaciones por grupo
Mco_lam = np.zeros((Ni * Nj, Ng))

#---------------------------------------------------------
# Mcal de datos modificados

# Calcular MR_ref_mod (promedio espectral + HSL por grupo)
MR_ref_mod = np.zeros((Ng, Nlam + 3))

for jg in range(Ng):
    aux = McalHSL_mod[McalHSL_mod['Ng'] == group[jg]][NbandHSL_sort].mean()
    MR_ref_mod[jg, :] = np.hstack((aux.values))

print("Procesamiento completado y MR_ref_mod calculado.")

#---------------------------------------------------------
# Mcal de datos modificados

#---------------------------------------------------------
# Iteración Clasificación
#---------------------------------------------------------    

 # Inicializar resultados
import time
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

# Directorio de salida para las matrices de clasificación
class_dir = os.path.join(path, '03-Report', '04_CLASSMAP', ver)  # Ruta para guardar los mapas de clasificación
os.makedirs(class_dir, exist_ok=True)
conf_dir = os.path.join(class_dir, "confidence")
sim_dir = os.path.join(class_dir, "max_similarity")


# Crear carpetas si no existen
os.makedirs(class_dir, exist_ok=True)
os.makedirs(conf_dir, exist_ok=True)
os.makedirs(sim_dir, exist_ok=True)

start_time = time.time()

# Crear un árbol KD para MR_ref
tree_ref = cKDTree(MR_ref_mod)

# Obtener geotransformación y proyección de la imagen original **fuera del bucle**
dataset_original = gdal.Open(first_file)
geotransform = dataset_original.GetGeoTransform()
projection = dataset_original.GetProjection()
dataset_original = None  # Liberar el dataset original


# # Definir los grupos principales

macroclass_map = {
    'agua superficial': 'Agua',
    'agua profunda': 'Agua',
    'relave seco': 'Relave',
    'relave húmedo': 'Relave',
    'relave consolidado': 'Relave',
    'terreno natural': 'Terreno',
    'otros': 'Terreno'
}

data = []# Iterar por fecha
intento = '0.5 - (cos_sim)..2'
# Normalizar las firmas espectrales de referencia (MR_ref_mod)

MR_ref_mod_norm = (MR_ref_mod/ 10000) / np.linalg.norm(MR_ref_mod/10000, axis=1, keepdims=True)

# Iterar por cada fecha
for fecha in df_roi_list['Fecha'].unique():
    print(f"Procesando fecha: {fecha}...")

    refle_data = df_refle[df_refle['Fecha'] == fecha].drop(columns=['Fecha']).values / 10000
    refle_data_norm = refle_data / np.linalg.norm(refle_data, axis=1, keepdims=True)

    # Calcular similitud del coseno
    cos_sim = cosine_similarity(refle_data_norm, MR_ref_mod_norm) # Npixeles , Ng
    cos_dissim = 0.5 - (cos_sim)**2 # disimilaridad angular
    
    # Distancia euclidiana
    dists = np.array([np.linalg.norm(refle_data - ref_vec, axis=1) for ref_vec in MR_ref_mod / 10000])
    dists = dists.T  # Npixeles , Ng
    
    # Radio combinado
    R = np.sqrt(cos_dissim**2 + dists**2)
    
    # Clasificación por mínimo radio
    ico = np.argmin(R, axis=1)
    # Recontruir como imagen
    ico_image = ico.reshape((Ni, Nj))
    
    #_________________________
    

#     # Crear máscaras binarias para cada grupo principal
#     mask_grupos = {
#     1: (ico_image == 1).astype(np.uint8),  # Grupo 1: Agua
#     2: (ico_image == 2).astype(np.uint8),  # Grupo 2: Relave
#     3: (ico_image == 3).astype(np.uint8)   # Grupo 3: Otros
#     }
#     # Expandir las máscaras para que coincidan con la forma de refle_data
#     mask_expanded = {grupo: np.repeat(mask[:, :, np.newaxis], refle_data.shape[1], axis=2).reshape(Ni * Nj, refle_data.shape[1])
#                  for grupo, mask in mask_grupos.items()}
#     #_________________________
#     
#     new_ico_image = np.copy(ico_image)  # Copia de la clasificación principal
#     
#     for grupo_id in [1,2,3]:
#         print(f"Procesando subclasificación del grupo {grupo_id}...")
#         refle_data_masked = refle_data * mask_expanded[grupo_id]
#         Mco_lam_2 = np.full(refle_data.shape[:1] + (Ng,), np.inf)
#         # Definir qué clases pertenecen al grupo actual
#         clases_en_grupo = [i for i, cat in enumerate(nameg) if grupo_principal[cat] == grupo_id]
#         # Calcular distancia espectral solo dentro del grupo
#         for jg in clases_en_grupo:
#             distancias_ref = np.sum((refle_data_masked - MR_ref_mod[jg])**2, axis=1)
#             Mco_lam_2[:, jg] = distancias_ref
#         
#         # Reclasificar píxeles dentro del grupo
#         ico_2 = np.argmin(Mco_lam_2, axis=1)
#         # Mapear a etiquetas de subgrupos
#         ico_2_labels = np.array([subgrupos[nameg[i]] for i in ico_2])
#         # Asignar la nueva clasificación en la imagen final
#         new_ico_image = (new_ico_image * (1 - mask_grupos[grupo_id])) + (ico_2_labels * mask_grupos[grupo_id])
    
    #_________________________

    # Definir rutas de salida
    output_filename_class = f'{fecha}_{name}_class_{ver}_try_{intento}.tif'
    output_filename_conf = f'{fecha}_{name}_confidence_{ver}.tif'
    output_filename_sim = f'{fecha}_{name}_max_similarity_{ver}.tif'

    output_file_class = os.path.join(class_dir,output_filename_class)
    output_file_class = output_file_class.replace("\\", "/")
    output_file_conf = os.path.join(class_dir,'confidence',output_filename_conf)
    output_file_sim = os.path.join(class_dir,'max_similarity',output_filename_sim)

    data.append([fecha, output_file_class, output_file_conf, output_file_sim])
    
    driver = gdal.GetDriverByName("GTiff")
    
    # Guardar imágenes rasterizadas solo si no existen
    if not os.path.exists(output_file_class):
        dataset_class = driver.Create(output_file_class, Nj, Ni, 1, gdal.GDT_Int16)
        if geotransform:
            dataset_class.SetGeoTransform(geotransform)   
        if projection:
            dataset_class.SetProjection(projection)
        dataset_class.GetRasterBand(1).WriteArray(ico_image)
        dataset_class.FlushCache()
        dataset_class = None  # Liberar memoria
        print(f"Clasificación guardada en {output_file_class}")
    else:
        print(f'El archivo ya existe, se omite el guardado: {output_file_class}')

#     # Guardar imagen de confianza
#     if not os.path.exists(output_file_conf):
#         dataset_conf = driver.Create(output_file_conf, Nj, Ni, 1, gdal.GDT_Float32)
#         if geotransform:
#             dataset_conf.SetGeoTransform(geotransform)
#         if projection:
#             dataset_conf.SetProjection(projection)
#         dataset_conf.GetRasterBand(1).WriteArray(confidence_image)
#         dataset_conf.FlushCache()
#         dataset_conf = None  # Liberar memoria
#         print(f"Mapa de confianza guardado en {output_file_conf}")
#     else:
#         print(f'El archivo ya existe, se omite el guardado: {output_file_conf}')
# 
#     # Guardar imagen de similitud máxima
#     if not os.path.exists(output_file_sim):
#         dataset_sim = driver.Create(output_file_sim, Nj, Ni, 1, gdal.GDT_Float32)
#         if geotransform:
#             dataset_sim.SetGeoTransform(geotransform)
#         if projection:
#             dataset_sim.SetProjection(projection)
#         dataset_sim.GetRasterBand(1).WriteArray(max_similarity_image)
#         dataset_sim.FlushCache()
#         dataset_sim = None  # Liberar memoria
#         print(f"Mapa de similitud máxima guardado en {output_file_sim}")
#     else:
#         print(f'El archivo ya existe, se omite el guardado: {output_file_sim}')

# # Ahora, sin importar si se crearon o no los archivos, se actualiza el CSV
#     if not df_roi_class[
#         (df_roi_class["Fecha"] == fecha) & (df_roi_class["Ver Class"] == ver)].any().any():
#         df_roi_class = pd.concat([
#             df_roi_class,
#             pd.DataFrame([
#             {"Fecha": fecha, "Ver Class": ver, "Ruta": output_file_class},])
#     ], ignore_index=True)
#         df_roi_class.to_csv(archivo_roi_class, index=False)
#         print(f"CSV actualizado: {archivo_roi_class}")


# Verifica si ya existe esa combinación de fecha y versión
mask = (df_roi_class["Fecha"] == fecha) & (df_roi_class["Ver Class"] == ver)

if df_roi_class[mask].empty:
    nueva_fila = pd.DataFrame([{
        "Fecha": fecha,
        "Ver Class": ver,
        "Ruta": output_file_class.replace("\\", "/")  # Evita backslashes en Windows
    }])
    df_roi_class = pd.concat([df_roi_class, nueva_fila], ignore_index=True)
    df_roi_class.to_csv(archivo_roi_class, index=False)
    print(f"CSV actualizado: {archivo_roi_class}")
else:
    print(f"La fila ya existe en el CSV para {fecha}, versión {ver}")

# Tiempo total
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")

# Guardar la actualización del archivo CSV
df_roi_class.to_csv(archivo_roi_class, index=False)
print(f"CSV actualizado: {archivo_roi_class}")
