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
ver = 'v7.12'
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
df_mcal = read_csv_file(archivo_mcal_mod,'#')

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
    aux = df_mcal[df_mcal['Ng'] == group[jg]][NbandHSL_sort].mean()
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
from scipy.special import softmax

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

# Agrupar por macroclase y calcular promedio
macroclass_labels = [macroclass_map.get(nombre, 'Desconocido') for nombre in nameg]
ng_to_macroclass = {i + 1: macroclass_labels[i] for i in range(len(macroclass_labels))}
df_mcal['macroclass'] = df_mcal['Ng'].map(ng_to_macroclass)
macroclass_ref = df_mcal.groupby('macroclass')[NbandHSL_sort].mean()

data = []# Iterar por fecha

# Normalizar las firmas espectrales de referencia (MR_ref_mod)

ref_matrix = macroclass_ref.loc[['Agua', 'Relave', 'Terreno']].values
ref_matrix_norm = (ref_matrix/ 10000) / np.linalg.norm(ref_matrix/10000, axis=1, keepdims=True)
#MR_ref_mod_norm = (MR_ref_mod/ 10000) / np.linalg.norm(MR_ref_mod/10000, axis=1, keepdims=True)

T = 0.1
# Iterar por cada fecha
for fecha in df_roi_list['Fecha'].unique():
    print(f"Procesando fecha: {fecha}...")

    refle_data = df_refle[df_refle['Fecha'] == fecha].drop(columns=['Fecha']).values / 10000
    refle_data_norm = refle_data / np.linalg.norm(refle_data, axis=1, keepdims=True)

    # Calcular similitud del coseno
    cos_sim = cosine_similarity(refle_data_norm, ref_matrix_norm) # Npixeles , Ng
    # Normalizar por fila
    proportions = cos_sim / cos_sim.sum(axis=1, keepdims=True)
    proportions = softmax(cos_sim /T, axis=1)
    mapas = {
    'Agua': proportions[:, 0].reshape((Ni, Nj)),
    'Relave': proportions[:, 1].reshape((Ni, Nj)),
    'Terreno': proportions[:, 2].reshape((Ni, Nj))}
    
    #_________________________

    # Definir rutas de salida

    driver = gdal.GetDriverByName("GTiff")
    
    for clase, img in mapas.items():
        output_filename_class = f'{fecha}_{name}_soft_{clase}_{ver}.tif'
        output_file_class = os.path.join(class_dir,output_filename_class).replace("\\", "/")
        
        # Guardar imágenes rasterizadas solo si no existen
        if not os.path.exists(output_file_class):
            dataset_class = driver.Create(output_file_class, Nj, Ni, 1, gdal.GDT_Float32)
            if geotransform:
                dataset_class.SetGeoTransform(geotransform)
            if projection:
                dataset_class.SetProjection(projection)
            dataset_class.GetRasterBand(1).WriteArray(img)
            dataset_class.FlushCache()
            dataset_class = None  # Liberar memoria
            print(f"Clasificación guardada en {output_file_class}")
        else:
            print(f'El archivo ya existe, se omite el guardado: {output_file_class}')
        
        # Registrar en CSV por cada clase (opcional)
        df_roi_class = pd.concat([df_roi_class,
                                  pd.DataFrame([{"Fecha": fecha,"Ver Class": f"{ver}_{clase}","Ruta": output_file_class}])
                                  ], ignore_index=True)
        # Guardar CSV actualizado (una sola vez)
        df_roi_class.to_csv(archivo_roi_class, index=False)
        print(f"CSV actualizado: {archivo_roi_class}")


# Tiempo total
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")

# Guardar la actualización del archivo CSV
df_roi_class.to_csv(archivo_roi_class, index=False)
print(f"CSV actualizado: {archivo_roi_class}")
