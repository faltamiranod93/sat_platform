# s_Sat_fil2ROI
# Necesita generarse el archivo en '00-ROI-utm.csv' en la carpeta 02-Space-Facilities
# Script para unir un ROI con diferentes imagenes satelitales

#---------------------------------------------------------
# Librerías
#---------------------------------------------------------
from osgeo import gdal, osr
from scipy.interpolate import griddata

import numpy as np
import pandas as pd
from datetime import datetime

import subprocess
import os
import sys
import importlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name = 'Salar-Pedernales'
terrain = "02-BAND-SAT"
path = os.path.join('C:/Users/felip/Desktop/Msc-UTFSM', name)

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
# Read Data
#---------------------------------------------------------
# Read ROI of site ..............................
roi_path = os.path.join(path, '02-Space-Facilities')
os.chdir(roi_path)
df_roi = pd.read_csv('00-ROI-utm.csv', comment='#')

# Puntos extremos ROI
p1 = df_roi[df_roi['pl'] == 'p1'][['xUTM', 'yUTM']].values[0]
p2 = df_roi[df_roi['pl'] == 'p2'][['xUTM', 'yUTM']].values[0]

### Read DEM Raster ..............................
#os.chdir(path + '/01-Terrain/')
#df_dem = pd.read_csv('01-DEM-SAT.csv',comment='#')
#N_dem = len(df_dem['DAT'])

# Read SAT Raster ..............................
terrain_path = os.path.join(path, '01-Terrain')
os.chdir(terrain_path)
df_ban = pd.read_csv('02-BAND-SAT-RAS.csv', comment='#')
#Limpia datos y reinicia indices
df_ras = df_ban.copy()
df_ras = df_ras.dropna(subset=['Banda'])
df_ras = df_ras.reset_index(drop=True)

N_ban = len(df_ras)

# Prioridad de resoluciones
resolucion_prioridad = {'R10m': 1, 'R20m': 2, 'R60m': 3}

# Añadir columna de prioridad al DataFrame original
df_ras['Prioridad'] = df_ras['Resolución'].map(resolucion_prioridad)

# Seleccionar la mejor resolución por banda, fecha y cuadrante (Tile ID)
df_bandas_unicas = (
    df_ras
    .sort_values(by=['Fecha', 'Tile ID', 'Banda', 'Prioridad'])  # Ordenar por Fecha, Tile ID, Banda y Prioridad
    .groupby(['Fecha', 'Tile ID', 'Banda'], as_index=False)      # Agrupar por Fecha, Tile ID y Banda
    .first()                                                     # Tomar la primera fila de cada grupo (mejor resolución)
)

#---------------------------------------------------------
# Procesamiento de Imágenes
#---------------------------------------------------------

# Crear directorios de salida
roi_dir = os.path.join(path, '02-Space-Facilities', 'ROI') #Ruta para dejar ROI y MOSAICOS
roi_list_dir = os.path.join(path, '02-Space-Facilities', 'ROI-LIST') #Ruta para dejar MultiBandas
os.makedirs(roi_dir, exist_ok=True)
os.makedirs(roi_list_dir, exist_ok=True)

# Orden deseado de bandas
Nband_sort = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
data = []

#---------------------------------------------------------
# Recorte del Mosaico

for index, row in df_bandas_unicas.iterrows():
    # Definir parámetros
    s_srs = f'EPSG:{int(row["PRJ"])}'
    t_srs = 'EPSG:32719'
    input_file = row['Ruta']
    output_filename = row['Archivo'].replace('.jp2', '_roi.tif')
    output_file = os.path.join(roi_dir, output_filename)

    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Evitar recorte innecesario si el archivo ya existe
    if os.path.exists(output_file):
        print(f'El archivo ya existe, se omite el recorte: {output_file}')
        #mantiene la ruta original del archivo
        df_bandas_unicas.loc[index, 'Ruta'] = output_file
        df_bandas_unicas.loc[index, 'Archivo'] = output_filename
        continue

    # Construir el comando gdalwarp
    cmd = [
        'gdalwarp',
        '-overwrite',
        '-s_srs', s_srs,
        '-t_srs', t_srs,
        '-te', str(p1[0]), str(p2[1]), str(p2[0]), str(p1[1]),  # xmin, ymin, xmax, ymax
        '-tr', str(10.0), str(10.0),  # Resolución de salida (ajustar si necesario)
        '-r', 'bilinear',  # Algoritmo de remuestreo
        '-dstnodata','0', #Los valores de NoData se rellenan con 0
        input_file, output_file
    ]

    # Ejecutar el comando
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f'Imagen recortada y guardada en: {output_file}')
        #Actualiza la ruta solo si el recorte fue exitoso
        df_bandas_unicas.loc[index, 'Ruta'] = output_file
        df_bandas_unicas.loc[index, 'Archivo'] = output_filename
        df_bandas_unicas.loc[index, 'PRJ'] = int(t_srs.split(':')[1])
    except subprocess.CalledProcessError as e:
        print(f'Error al ejecutar gdalwarp en {input_file}: {e.stderr}')
        continue

# Guardar el DataFrame actualizado si es necesario
output_csv_path = os.path.join(path, '02-Space-Facilities', '02-BAND-SAT-RAS-ROI.csv')
df_bandas_unicas.to_csv(output_csv_path, index=False)
print(f'DataFrame actualizado guardado en: {output_csv_path}')

#---------------------------------------------------------
# Union de Tiles por Banda

df_bandas_unicas = pd.read_csv(output_csv_path, comment='#')
# Orden deseado
Nband_sort = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
data = []# Iterar por fecha
for fecha in df_bandas_unicas['Fecha'].unique():
    df_fecha = df_bandas_unicas[df_bandas_unicas['Fecha'] == fecha]
    
    for banda in Nband_sort:
        # Filtrar por banda
        df_banda = df_fecha[df_fecha['Banda'] == banda]
        if df_banda.empty:
            print(f"No se encontraron imágenes para la banda {banda} en la fecha {fecha}")
            continue

        # Archivos por cuadrante
        cuadrante_files = df_banda['Ruta'].tolist()

        # Validar existencia de todos los archivos
        if not all(os.path.exists(f) for f in cuadrante_files):
            print(f"Archivos faltantes o no válidos para banda {banda} en fecha {fecha}: {cuadrante_files}")
            continue
        
        # Nombre del archivo combinado de cuadrantes para esta banda
        output_banda_file = os.path.join(roi_dir, f'{fecha}_{banda}_mosaic.tif')

        # Crear comando para mosaico
        cmd_mosaic = [
            'gdalwarp',
            '-overwrite',
            '-r', 'bilinear',  # Algoritmo de remuestreo
            '-t_srs', 'EPSG:32719',  # Sistema de coordenadas de salida
            '-of', 'GTiff',  # Formato de salida
            '-co', 'COMPRESS=LZW',  # Compresión
            '-co', 'BIGTIFF=YES',  # Para archivos grandes
            '-srcnodata', '0',  # Ignorar valores 0 en las entradas
            '-dstnodata', '0',  # Establecer 0 como NoData en la salida
            *cuadrante_files,  # Archivos de entrada
            output_banda_file  # Archivo de salida
        ]

        # Ejecutar el mosaico
        try:
            result = subprocess.run(cmd_mosaic, check=True, capture_output=True, text=True)
            print(f'Cuadrantes unidos para banda {banda} en: {output_banda_file}')
            data.append([fecha, banda, output_banda_file])
        except subprocess.CalledProcessError as e:
            print(f'Error al unir cuadrantes para banda {banda} en fecha {fecha}: {e.stderr}')
            continue

# Crear un DataFrame para el resultado
df_result = pd.DataFrame(data, columns=['Fecha', 'Banda', 'Ruta'])
# Exportar el DataFrame a CSV
output_csv_path = os.path.join(path,'02-Space-Facilities', '03-ROI-LIST-MOSAICS.csv')
df_result.to_csv(output_csv_path, index=False)
print(f'DataFrame de mosaicos exitosos guardado en: {output_csv_path}')

