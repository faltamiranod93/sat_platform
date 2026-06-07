import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io import loadmat 
import tkinter as tk
import os
import pathlib
from pathlib import Path
import sys
from tkinter import filedialog
from osgeo import gdal, osr
from matplotlib.widgets import Button as MplButton

import importlib
import seaborn as sns

##--CARGA DE DATOS--##
#---------------------------------------------------------
# Input
#---------------------------------------------------------
#directorio_actual = os.getcwd()
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


##--CLASIFICACIÓN PIXEL--##............................
fecha=fechas[0]

# Composición RGB para visualización (B04, B03, B02 → [3,2,1])
ima_rgb = image[:, :, [3, 2, 1]].astype(np.float32)

# Normalizar cada banda individualmente al rango 0–255
for k in range(3):
    max_val = np.max(ima_rgb[:, :, k])
    if max_val > 0:
        ima_rgb[:, :, k] = 255. * ima_rgb[:, :, k] / max_val

# Convertir a uint8
ima = ima_rgb.astype(np.uint8)

# logica de seleccion de area de interes ..................     
roi = np.ones(len(nameg), dtype=int)
# logica seleccion de puntos de cal .......................
select = np.ones(len(nameg), dtype=int)
# numero de grupos .......................................
Ng = len(nameg)

#Selección General de puntos
MR_ref  = np.zeros((Ng,Nlam)); 
MRe_ref =[None]*Ng #lista
Mijp=[None]*Ng #lista 

# Crear la figura
fig, ax = plt.subplots()
ax.imshow(ima)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title(f'Seleccione los puntos para {nameg[0]}')
plt.gca().set_aspect('equal', adjustable='box')

# Instrucciones
print("Instrucciones:")
print("1. Haz clic en los puntos de la imagen para seleccionarlos.")
print("2. Usa la rueda del mouse para hacer zoom en la imagen.")
print("3. Haz clic en 'Next Group' cuando hayas terminado con el grupo actual.")

# === VARIABLES INICIALES ===

serie=np.zeros((1,15));
current_group_index = [0]  # Usar lista para modificar en la función

def select_points():
    # Llamar a la función para iniciar selección
    ax.set_title(f'Selecciona puntos para: {nameg[current_group_index[0]]}', loc='left')
    plt.draw()

# Función para continuar al siguiente grupo
def next_group(event):
    if current_group_index[0] < Ng - 1:
        current_group_index[0] += 1
        ax.set_title(f'Selecciona puntos para: {nameg[current_group_index[0]]}', loc='left')
        plt.draw()
        print(f"Cambiado a grupo {nameg[current_group_index[0]]}")
    else:
        print("Todos los grupos han sido completados.")


# Función para deshacer el último punto seleccionado
def undo_last_point(event):
    global serie
    if len(serie) > 0:
        last_point = serie[-1, :]
        x_point, y_point = last_point[:2].astype(int)
        serie = serie[:-1, :]
        print("Último punto eliminado.")
        ax.plot(x_point, y_point, 'ko')  # Marcar en negro el punto deshecho
        plt.draw()

def reset_zoom(event):
    ax.set_xlim(0, Nj)
    ax.set_ylim(Ni, 0)
    plt.draw()        


# === CREACIÓN DE BOTONES ===
axnext = plt.axes([0.8, 0.01, 0.1, 0.075])
axundo = plt.axes([0.6, 0.01, 0.1, 0.075])
axreset = plt.axes([0.4, 0.01, 0.1, 0.075])

bnext = MplButton(axnext, 'Next Group')
bnext.on_clicked(next_group)

bundo = MplButton(axundo, 'Undo Last')
bundo.on_clicked(undo_last_point)

breset = MplButton(axreset, 'Reset Zoom')
breset.on_clicked(reset_zoom)

# Estado del pan y zoom
pan_active = False
last_mouse_pos = None

# Función de zoom con la rueda del mouse
def on_scroll(event):
    xdata = event.xdata
    ydata = event.ydata
    if xdata is None or ydata is None:
        return
    scale_factor = 1.2
    if event.button == 'up':
        scale_factor = 1 / scale_factor
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
    ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
    plt.draw()

# Función para el pan
def on_mouse_move(event):
    global pan_active, last_mouse_pos
    if pan_active:
        if last_mouse_pos is None:
            last_mouse_pos = (event.xdata, event.ydata)
            return
        dx = last_mouse_pos[0] - event.xdata
        dy = last_mouse_pos[1] - event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        ax.set_xlim([cur_xlim[0] + dx, cur_xlim[1] + dx])
        ax.set_ylim([cur_ylim[0] + dy, cur_ylim[1] + dy])
        plt.draw()
        last_mouse_pos = (event.xdata, event.ydata)

# Funciones para activar/desactivar el pan
def activate_pan(event):
    global pan_active, last_mouse_pos
    pan_active = True
    last_mouse_pos = (event.xdata, event.ydata)

def deactivate_pan(event):
    global pan_active, last_mouse_pos
    pan_active = False
    last_mouse_pos = None

# Restablecer el zoom
def reset_zoom(event):
    ax.set_xlim(0, Nj)
    ax.set_ylim(Ni, 0)
    plt.draw()

breset.on_clicked(reset_zoom)

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', activate_pan)
fig.canvas.mpl_connect('button_release_event', deactivate_pan)

# Capturar clics
def onclick(event):
    x_point, y_point = int(event.xdata), int(event.ydata)
    if event.inaxes == ax and 0 <= x_point < Nj and 0 <= y_point < Ni:
        # Almacena x, y, índice de grupo, y valores Bandas
        spectral_values = image[y_point, x_point, :].tolist()
        serie_aux = np.array([y_point, x_point, current_group_index[0] + 1] + spectral_values)
        
        global serie
        serie = np.vstack((serie, serie_aux))
        
        print(f"Punto registrado en ({x_point}, {y_point}) para {nameg[current_group_index[0]]}")
        # Dibujar el punto en la imagen
        ax.plot(x_point, y_point, 'ro')
        plt.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

# Iniciar la selección
select_points()
plt.show()

# Crear DataFrame y guardar los resultados
df1 = pd.DataFrame(serie).astype(int)
df1.insert(0,'Fecha',fecha)
df1.columns=Mcal.columns
df1=df1.drop(index=0)
Mcal = pd.concat([Mcal, df1], ignore_index=True)
os.chdir(path)

Mcal['Fecha'] = pd.to_datetime(Mcal['Fecha']) #Convertir la columna 'Fecha' a tipo datetime
Mcal = Mcal.sort_values(by=['Fecha', 'Ng','i','j']) #Ordenar el dataframe por las columnas 'Fecha', 'Ng' 'i' y 'j'
#duplicados = Mcal[Mcal.duplicated(subset=['Fecha', 'i', 'j'], keep=False)]
Mcal = Mcal.drop_duplicates(subset=['Fecha', 'i', 'j']) # Eliminar duplicados y mantener solo la primera ocurrencia
Mcal=Mcal.reset_index(drop=True)

#Mcal.to_csv('Mcal_py.csv',index=False)

print(df1)
