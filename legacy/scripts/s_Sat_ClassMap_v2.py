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

import tracemalloc

# Iniciar el rastreo de memoria
tracemalloc.start()

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name = 'Laguna-Seca'
terrain = "02-BAND-SAT"
ver = 'v2'
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

# Crear una copia de Mcal para agregar las columnas modificadas
McalHSL_mod = pd.DataFrame()

# Iterar sobre las fechas únicas en Mcal
for fecha in Mcal['Fecha'].unique():
    print(f"Procesando datos para la fecha: {fecha}...")

    # Filtrar las filas de Mcal para la fecha actual
    Mcal_fecha = Mcal[Mcal['Fecha'] == fecha].copy()

    # Buscar el archivo correspondiente a la fecha
    file_name = df_roi_list[df_roi_list['Fecha'] == fecha]['Ruta'].values
    if len(file_name) == 0:
        print(f"No se encontró un archivo para la fecha {fecha}. Saltando...")
        continue
    file_name = file_name[0]  # Tomar el primer archivo (debe ser único)

    # Verificar si el archivo existe
    if not os.path.exists(file_name):
        print(f"Archivo no encontrado para la fecha {fecha}: {file_name}. Saltando...")
        continue

    # Cargar la imagen
    try:
        imagen_modificada = load_image(file_name)
    except Exception as e:
        print(f"Error al cargar la imagen {file_name}: {e}")
        continue

    # Procesar cada fila en Mcal para extraer valores de la imagen modificada
    for idx, row in Mcal_fecha.iterrows():
        i, j = int(row['i']), int(row['j'])  # Índices espaciales

        # Extraer los valores de las bandas espectrales de la imagen modificada
        bandas_modificadas = imagen_modificada[i, j, :Nlam]
        bandas_modificadas = np.round(bandas_modificadas).astype(int)

        # Asignar las bandas modificadas a las columnas correspondientes
        for b, banda in enumerate(Nband_sort):  # Recorre B01, B02, ..., B12
            Mcal_fecha.at[idx, banda] = bandas_modificadas[b]

        # Calcular H, S, L a partir de B04 (Red), B03 (Green), B02 (Blue)
        rgb = bandas_modificadas[[3, 2, 1]] / 10000.  # Normalizar RGB
        h, s, l = rgb2hsl(rgb)

        # Asignar los valores escalados a las columnas correspondientes
        Mcal_fecha.at[idx, 'H'] = h
        Mcal_fecha.at[idx, 'S'] = s
        Mcal_fecha.at[idx, 'L'] = l

    # Agregar los datos procesados para esta fecha a McalHSL_mod
    McalHSL_mod = pd.concat([McalHSL_mod, Mcal_fecha], ignore_index=True)

# Calcular MR_ref_mod (promedio espectral + HSL por grupo)
MR_ref_mod = np.zeros((Ng, Nlam + 3))

for jg in range(Ng):
    aux = McalHSL_mod[McalHSL_mod['Ng'] == group[jg]][NbandHSL_sort].mean()
    MR_ref_mod[jg, :] = np.hstack((aux.values))

print("Procesamiento completado y MR_ref_mod calculado.")
 
#---------------------------------------------------------
# Mcal de datos modificados

#---------------------------------------------------------
# Comparación de datos modificados

# Definir una lista de colores para los grupos
from matplotlib.cm import get_cmap

colormap = get_cmap("tab10")  # Usar un colormap predefinido
colores_grupos = [colormap(i) for i in range(Ng)]  # Asignar un color único a cada grupo

plt.figure(figsize=(12, 6))

for jg in range(Ng):
    plt.plot(lam_sorted, MR_ref[jg, :Nlam], label=f'{nameg[jg]} (Original)', linestyle='--', color=colores_grupos[jg])
    plt.plot(lam_sorted, MR_ref_mod[jg, :Nlam], label=f'{nameg[jg]} (Modificado)', linestyle='-', color=colores_grupos[jg])

# Configuración de la leyenda
plt.legend(
    title="Grupos Espectrales",        # Título de la leyenda
    loc='upper center',                # Ubicación
    bbox_to_anchor=(0.5, -0.1),        # Coloca la leyenda debajo del gráfico
    ncol=2,                            # Número de columnas
    fontsize='small',                  # Tamaño del texto
    frameon=True,                      # Mostrar marco
    framealpha=0.9,                    # Transparencia del marco
    edgecolor='gray',                  # Color del borde del marco
    title_fontsize='medium'            # Tamaño del título
)

# Configuración adicional del gráfico
plt.title('Comparación de Reflectancia Promedio por Grupo')
plt.xlabel('Longitud de Onda (nm)')
plt.ylabel('Reflectancia Promedio')
plt.grid(True)
plt.tight_layout()  # Ajustar para evitar superposiciones
plt.show()

#---------------------------------------------------------
# Comparación de datos modificados

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

# Crear un árbol KD para MR_ref
tree_ref = cKDTree(MR_ref_mod)

# Obtener geotransformación y proyección de la imagen original **fuera del bucle**
dataset_original = gdal.Open(first_file)
geotransform = dataset_original.GetGeoTransform()
projection = dataset_original.GetProjection()
dataset_original = None  # Liberar el dataset original

# Procesar cada fecha
for fecha in df_roi_list['Fecha'].unique():
    print(f"Procesando fecha: {fecha}...")

    # Preparar datos espectrales para la fecha
    refle_data = df_refle[df_refle['Fecha'] == fecha].drop(columns=['Fecha']).values
    n_pixeles = refle_data.shape[0]

    # Inicializar matriz de distancias
    Mco_lam = np.full((n_pixeles, Ng), np.inf)  # Rellenar con infinito

    # Iterar sobre cada grupo
    for jg in range(Ng):
        # Calcular distancias a MR_ref
        distancias_ref = np.sum((refle_data - MR_ref_mod[jg])**2, axis=1)
        Mco_lam[:, jg] = np.minimum(Mco_lam[:, jg], distancias_ref)

    # Clasificar píxeles al grupo más cercano
    ico = np.argmin(Mco_lam, axis=1)  # Índices de los grupos más cercanos
    ico_image = ico.reshape((Ni, Nj))  # Convertir 1D a 2D

    # Definir la ruta del archivo de salida    
    output_filename_class = f'{fecha}_{name}_class_{ver}.tif'
    output_file_class = os.path.join(class_dir,output_filename_class)

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

# Tiempo total
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total transcurrido: {elapsed_time:.2f} segundos")

# Tomar una instantánea del uso de memoria
snapshot = tracemalloc.take_snapshot()

# Mostrar las estadísticas de memoria
stats = snapshot.statistics('lineno')
for stat in stats[:10]:  # Las 10 líneas que más consumen memoria
    print(stat)

#---------------------------------------------------------
# Graficos
#---------------------------------------------------------    
# 
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# 
# Ng = len(group)  # Número de grupos
# 
# # Inicializar figura con subplots
# fig = make_subplots(
#     rows=1, cols=Ng,
#     shared_yaxes=True,  # Compartir eje y entre subplots
#     subplot_titles=nameg
# )
# 
# # Encontrar los límites del eje y para ajustar todos los gráficos
# y_min = float('inf')
# y_max = float('-inf')
# 
# # Agregar trazas a cada subplot
# for jg in range(1, Ng+1):  # Recorre cada grupo
#     for fecha in Mcal['Fecha'].unique():
#         aux = Mcal[Mcal['Fecha'] == fecha]
#         y_data = (aux[aux['Ng'] == group[jg-1]][Nband_sort]).mean()
#         y_min = min(y_min, y_data.min())
#         y_max = max(y_max, y_data.max())
#         
#         fig.add_trace(
#             go.Scatter(x=lam[i_slam], y=y_data, mode='lines', showlegend=False),
#             row=1, col=jg
#         )
# 
# # Ajustar el rango del eje y para todos los subplots
# for i in range(1, Ng+1):
#     fig.update_yaxes(range=[y_min, y_max], row=1, col=i)
# 
# # Actualizar el diseño
# fig.update_layout(
#     title='Your Plot Title',
#     height=500, width=1500,
#     xaxis_title='λ [nm]',
#     yaxis_title='R [-]',
#     showlegend=False,
#     plot_bgcolor='white'
# )
# 
# # Mostrar la figura
# fig.show()
# 
# # Imagen clasificada por grupo y por fecha//promedios............................
# 
# plt.figure(2, figsize=(15, 5))
# for jg in range(1, Ng+1): #Recorre cada grupo
#     plt.subplot(1, Ng, jg)
#     for fecha in Mcal['Fecha'].unique():
#         aux=Mcal[Mcal['Fecha']==fecha]
#         plt.plot(lam[i_slam],(aux[aux['Ng']==group[jg-1]][Nband_sort]).mean())
#         
#         #plt.plot(lam[i_slam], Mcal.loc[index,Nband_sort], marker='o')
#     #plt.plot(lam[i_slam],MR_ref[jg-1][0:12],color='black',linewidth=2)
#     plt.ylim([0, 6000])
#     plt.grid(True)
#     plt.xlabel('λ [nm]')
#     plt.ylabel(' R [-]')
#     plt.title(nameg[jg-1])
#     plt.legend(Mcal['Fecha'].unique())
#     
# plt.tight_layout()  # Ajusta el espaciado entre subplots para que no se solapen
# plt.savefig('subplots_fechas.png', dpi=300, bbox_inches='tight')
# plt.show()
# 
# # Imagen clasificada por grupo y por fecha//scatters............................
# 
# plt.figure(2, figsize=(15, 5))
# for jg in range(1, Ng+1): #Recorre cada grupo
#     plt.subplot(1, Ng, jg)
#     for index in np.where(Mcal['Ng'] == group[jg-1])[0]: ###AQUI HAY CAMBIO
#         plt.plot(lam[i_slam], Mcal.loc[index,Nband_sort], marker='o')
#     #plt.plot(lam[i_slam],MR_ref[jg-1][0:12],color='black',linewidth=2)
#     plt.ylim([0, 6000])
#     plt.grid(True)
#     plt.xlabel('λ [nm]')
#     plt.ylabel(' R [-]')
#     plt.title(nameg[jg-1])
#     
# plt.tight_layout()  # Ajusta el espaciado entre subplots para que no se solapen
# plt.show()
# 
# # Imagen clasificada por grupo ............................
# # Crear una figura con subplots de 4 filas y 2 columnas
# fig, axes = plt.subplots(len(df_roi_list['Fecha'].unique()), 2, figsize=(10, 20))
# 
# fechas=df_roi_list['Fecha'].unique()
# 
# # Iterar sobre cada fecha
# for i in range(len(fechas)):
#     # Mostrar la imagen visible a la izquierda
#     ax_visible = axes[i, 0]
#     ax_visible.imshow(visible_fecha[fechas[i]])
#     ax_visible.set_title(f'{fechas[i]} - Visible')
#     ax_visible.axis('off')  # Ocultar ejes
# 
#     # Mostrar la imagen clasificada a la derecha
#     ax_classified = axes[i, 1]
#     ax_classified.imshow(class_fecha[fechas[i]])
#     #ax_classified.imshow(classified_images[i], cmap='gray')
#     ax_classified.set_title(f'{fechas[i]} - Clasificada')
#     ax_classified.axis('off')  # Ocultar ejes
# 
# # Ajustar el layout para que no haya superposición
# plt.tight_layout()
# plt.savefig('subplots_imagenes.png', dpi=300, bbox_inches='tight')
# plt.show()
# 
# 
# # Boxplot por grupo
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Ng', y='H', data=Mcal_HSL)
# plt.title('Distribución de H por Grupo')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Ng', y='S', data=Mcal_HSL)
# plt.title('Distribución de S por Grupo')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Ng', y='L', data=Mcal_HSL)
# plt.title('Distribución de L por Grupo')
# plt.show()
# 
# # Boxplots Grupo1  ............................
# 
# df_ng1=Mcal_HSL[Mcal_HSL['Ng']==1]
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='H', data=df_ng1)
# plt.title('Distribución de H por Grupo Ng=1 Agua Profunda')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='L', data=df_ng1)
# plt.title('Distribución de L por Grupo Ng=1 Agua Profunda')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='S', data=df_ng1)
# plt.title('Distribución de S por Grupo Ng=1 Agua Profunda')
# plt.show()
# 
# # Boxplots Grupo2  ............................
# 
# df_ng2=Mcal_HSL[Mcal_HSL['Ng']==2]
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='H', data=df_ng2)
# plt.title('Distribución de H por Grupo Ng=2 Agua Superficial')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='L', data=df_ng2)
# plt.title('Distribución de L por Grupo Ng=2 Agua Superficial')
# plt.show()
# 
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Fecha', y='S', data=df_ng2)
# plt.title('Distribución de S por Grupo Ng=2 Agua Superficial')
# plt.show()
# 
# # 3D Scatters Grupo 1 y 2  ............................
# 
# import plotly.express as px
# fig1 = px.scatter_3d(Mcal_HSL, x='H', y='S', z='L',
#               color='Ng')
# fig1.show()
# 
# fig2 = px.scatter_3d(df_ng1, x='H', y='S', z='L',
#               color='Fecha')
# fig2.show()

