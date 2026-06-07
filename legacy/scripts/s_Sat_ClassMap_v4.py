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
ver = 'v4'
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
with open(os.path.join(ruta_funciones,"config_bandas.json"), "r") as f:
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

# # Crear una copia de Mcal para agregar las columnas modificadas
# McalHSL_mod = pd.DataFrame()
# 
# # Iterar sobre las fechas únicas en Mcal
# for fecha in Mcal['Fecha'].unique():
#     print(f"Procesando datos para la fecha: {fecha}...")
# 
#     # Filtrar las filas de Mcal para la fecha actual
#     Mcal_fecha = Mcal[Mcal['Fecha'] == fecha].copy()
# 
#     # Buscar el archivo correspondiente a la fecha
#     file_name = df_roi_list[df_roi_list['Fecha'] == fecha]['Ruta'].values
#     if len(file_name) == 0:
#         print(f"No se encontró un archivo para la fecha {fecha}. Saltando...")
#         continue
#     file_name = file_name[0]  # Tomar el primer archivo (debe ser único)
# 
#     # Verificar si el archivo existe
#     if not os.path.exists(file_name):
#         print(f"Archivo no encontrado para la fecha {fecha}: {file_name}. Saltando...")
#         continue
# 
#     # Cargar la imagen
#     try:
#         imagen_modificada = load_image(file_name)
#     except Exception as e:
#         print(f"Error al cargar la imagen {file_name}: {e}")
#         continue
# 
#     # Procesar cada fila en Mcal para extraer valores de la imagen modificada
#     for idx, row in Mcal_fecha.iterrows():
#         i, j = int(row['i']), int(row['j'])  # Índices espaciales
# 
#         # Extraer los valores de las bandas espectrales de la imagen modificada
#         bandas_modificadas = imagen_modificada[i, j, :Nlam]
#         bandas_modificadas = np.round(bandas_modificadas).astype(int)
# 
#         # Asignar las bandas modificadas a las columnas correspondientes
#         for b, banda in enumerate(Nband_sort):  # Recorre B01, B02, ..., B12
#             Mcal_fecha.at[idx, banda] = bandas_modificadas[b]
# 
#         # Calcular H, S, L a partir de B04 (Red), B03 (Green), B02 (Blue)
#         rgb = bandas_modificadas[[3, 2, 1]] / 10000.  # Normalizar RGB
#         h, s, l = rgb2hsl(rgb)
# 
#         # Escalar H, S y L al rango [0, 10000]
#         h = h * (10000 / 360)
#         s = s * (10000 / 100)
#         l = l * (10000 / 100)
# 
#         # Asignar los valores escalados a las columnas correspondientes
#         Mcal_fecha.at[idx, 'H'] = h
#         Mcal_fecha.at[idx, 'S'] = s
#         Mcal_fecha.at[idx, 'L'] = l
# 
#     # Agregar los datos procesados para esta fecha a McalHSL_mod
#     McalHSL_mod = pd.concat([McalHSL_mod, Mcal_fecha], ignore_index=True)

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
 # SIMILITUD COSENO ---------------------------

 # Inicializar resultados
import time
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

# Directorio de salida para las matrices de clasificación
class_dir = os.path.join(path, '03-Report', '04_CLASSMAP', ver)  # Ruta para guardar los mapas de clasificación
os.makedirs(class_dir, exist_ok=True)

start_time = time.time()

# Crear un árbol KD para MR_ref
tree_ref = cKDTree(MR_ref_mod)

# Obtener geotransformación y proyección de la imagen original **fuera del bucle**
dataset_original = gdal.Open(first_file)
geotransform = dataset_original.GetGeoTransform()
projection = dataset_original.GetProjection()
dataset_original = None  # Liberar el dataset original

data = []# Iterar por fecha

# Normalizar las firmas espectrales de referencia (MR_ref_mod)
MR_ref_mod_norm = MR_ref_mod / np.linalg.norm(MR_ref_mod, axis=1, keepdims=True)

# Iterar por cada fecha
for fecha in df_roi_list['Fecha'].unique():
    print(f"Procesando fecha: {fecha}...")

    # Extraer datos espectrales
    refle_data = df_refle[df_refle['Fecha'] == fecha].drop(columns=['Fecha']).values
    n_pixeles = refle_data.shape[0]

    # Normalizar los vectores de los píxeles
    refle_data_norm = refle_data / np.linalg.norm(refle_data, axis=1, keepdims=True)

    # Calcular similitud del coseno
    Mco_lam = cosine_similarity(refle_data_norm, MR_ref_mod_norm)

    # Asignar cada píxel al grupo con mayor similitud
    ico = np.argmax(Mco_lam, axis=1)
    max_similarity = np.max(Mco_lam, axis=1)  # Similitud más alta

    # Calcular la diferencia con el segundo mejor grupo
    sorted_similarities = np.sort(Mco_lam, axis=1)
    second_best_similarity = sorted_similarities[:, -2]  # Segunda mayor similitud
    confidence = max_similarity - second_best_similarity  # Diferencia entre primer y segundo grupo

    # Convertir en imágenes 2D
    ico_image = ico.reshape((Ni, Nj))
    confidence_image = confidence.reshape((Ni, Nj))
    max_similarity_image = max_similarity.reshape((Ni, Nj))

    # Definir rutas de salida
    output_filename_class = f'{fecha}_{name}_class_{ver}.tif'
    output_filename_conf = f'{fecha}_{name}_confidence_{ver}.tif'
    output_filename_sim = f'{fecha}_{name}_max_similarity_{ver}.tif'

    output_file_class = os.path.join(class_dir,output_filename_class)
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

    # Guardar imagen de confianza
    if not os.path.exists(output_file_conf):
        dataset_conf = driver.Create(output_file_conf, Nj, Ni, 1, gdal.GDT_Float32)
        if geotransform:
            dataset_conf.SetGeoTransform(geotransform)
        if projection:
            dataset_conf.SetProjection(projection)
        dataset_conf.GetRasterBand(1).WriteArray(confidence_image)
        dataset_conf.FlushCache()
        dataset_conf = None  # Liberar memoria
        print(f"Mapa de confianza guardado en {output_file_conf}")
    else:
        print(f'El archivo ya existe, se omite el guardado: {output_file_conf}')

    # Guardar imagen de similitud máxima
    if not os.path.exists(output_file_sim):
        dataset_sim = driver.Create(output_file_sim, Nj, Ni, 1, gdal.GDT_Float32)
        if geotransform:
            dataset_sim.SetGeoTransform(geotransform)
        if projection:
            dataset_sim.SetProjection(projection)
        dataset_sim.GetRasterBand(1).WriteArray(max_similarity_image)
        dataset_sim.FlushCache()
        dataset_sim = None  # Liberar memoria
        print(f"Mapa de similitud máxima guardado en {output_file_sim}")
    else:
        print(f'El archivo ya existe, se omite el guardado: {output_file_sim}')

# Ahora, sin importar si se crearon o no los archivos, se actualiza el CSV
    if not df_roi_class[
        (df_roi_class["Fecha"] == fecha) & (df_roi_class["Ver Class"] == ver)].any().any():
        df_roi_class = pd.concat([
            df_roi_class,
            pd.DataFrame([
            {"Fecha": fecha, "Ver Class": ver, "Ruta": output_file_class},])
    ], ignore_index=True)
        df_roi_class.to_csv(archivo_roi_class, index=False)
        print(f"CSV actualizado: {archivo_roi_class}")

    
#     # Evitar sobrescribir si ya existe
#     if os.path.exists(output_file_class):
#         print(f'El archivo ya existe, se omite el guardado: {output_file_class}')
#         continue
# 
#     # Guardar imágenes rasterizadas (clasificación, confianza y similitud máxima)
#     driver = gdal.GetDriverByName("GTiff")
# 
#     # Guardar clasificación
#     dataset_class = driver.Create(output_file_class, Nj, Ni, 1, gdal.GDT_Int16)
#     if geotransform:
#         dataset_class.SetGeoTransform(geotransform)
#     if projection:
#         dataset_class.SetProjection(projection)
#     dataset_class.GetRasterBand(1).WriteArray(ico_image)
#     dataset_class.FlushCache()
#     dataset_class = None  # Liberar memoria
# 
#     # Guardar imagen de confianza
#     dataset_conf = driver.Create(output_file_conf, Nj, Ni, 1, gdal.GDT_Float32)
#     if geotransform:
#         dataset_conf.SetGeoTransform(geotransform)
#     if projection:
#         dataset_conf.SetProjection(projection)
#     dataset_conf.GetRasterBand(1).WriteArray(confidence_image)
#     dataset_conf.FlushCache()
#     dataset_conf = None  # Liberar memoria
# 
#     # Guardar imagen de similitud máxima
#     dataset_sim = driver.Create(output_file_sim, Nj, Ni, 1, gdal.GDT_Float32)
#     if geotransform:
#         dataset_sim.SetGeoTransform(geotransform)
#     if projection:
#         dataset_sim.SetProjection(projection)
#     dataset_sim.GetRasterBand(1).WriteArray(max_similarity_image)
#     dataset_sim.FlushCache()
#     dataset_sim = None  # Liberar memoria
# 
#     print(f"Clasificación por similitud del coseno de {fecha} guardada en {output_file_class}")
#     print(f"Mapa de confianza guardado en {output_file_conf}")
#     print(f"Mapa de similitud máxima guardado en {output_file_sim}")



# Tiempo total
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")

# Guardar la actualización del archivo CSV
df_roi_class.to_csv(archivo_roi_class, index=False)
print(f"CSV actualizado: {archivo_roi_class}")

# # Crear un DataFrame para el resultado
# df_result = pd.DataFrame(data, columns=['Fecha', 'Ver Class', 'Ruta'])
# # Exportar el DataFrame a CSV
# output_csv_path = os.path.join(path,'02-Space-Facilities', '05-ROI-MOD-CLASS.csv')
# df_result.to_csv(output_csv_path, index=False)
# print(f'DataFrame de clasificación guardado en: {output_csv_path}')

# # Definir una lista de colores para los grupos
# from matplotlib.cm import get_cmap
# 
# colormap = get_cmap("tab10")  # Usar un colormap predefinido
# colores_grupos = [colormap(i) for i in range(Ng)]  # Asignar un color único a cada grupo
# 
# plt.figure(figsize=(12, 6))
# 
# for jg in range(Ng):
#     plt.plot(lam_sorted, MR_ref_mod_norm[jg, :Nlam], label=f'{nameg[jg]} (Modificado)', linestyle='-', color=colores_grupos[jg])
# 
# # Configuración de la leyenda
# plt.legend(
#     title="Grupos Espectrales",        # Título de la leyenda
#     loc='upper center',                # Ubicación
#     bbox_to_anchor=(0.5, -0.1),        # Coloca la leyenda debajo del gráfico
#     ncol=2,                            # Número de columnas
#     fontsize='small',                  # Tamaño del texto
#     frameon=True,                      # Mostrar marco
#     framealpha=0.9,                    # Transparencia del marco
#     edgecolor='gray',                  # Color del borde del marco
#     title_fontsize='medium'            # Tamaño del título
# )
# 
# # Configuración adicional del gráfico
# plt.title('Comparación de Reflectancia Promedio por Grupo')
# plt.xlabel('Longitud de Onda (nm)')
# plt.ylabel('Reflectancia Promedio')
# plt.grid(True)
# plt.tight_layout()  # Ajustar para evitar superposiciones
# plt.show()
