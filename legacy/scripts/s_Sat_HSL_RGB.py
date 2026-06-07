import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat 
#import tkinter as tk
import os
import sys
import pathlib
from pathlib import Path
#from tkinter import filedialog
from f_functions import *
from osgeo import gdal, osr
import seaborn as sns

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name='Laguna-Seca'
#site="05-TSF"
terrain="02-BAND-SAT"
path = 'C:/Users/felip/Desktop/Msc-UTFSM/' + name

sys.path.append('C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos')

# Read Data
df_roi = read_csv_file(f'{path}/02-Space-Facilities/00-ROI-utm.csv','#')
df_roi_list = read_csv_file(f'{path}/02-Space-Facilities/03-ROI-LIST.csv','#')
Mcal = read_csv_file(f'{path}/Mcal_py.csv','#')

# Extract ROI points
if df_roi is not None:
    ip1 = df_roi[df_roi['pl'] == 'p1'].index[0]
    ip2 = df_roi[df_roi['pl'] == 'p2'].index[0]
    p1 = df_roi.loc[ip1, ['xUTM', 'yUTM']].values
    p2 = df_roi.loc[ip2, ['xUTM', 'yUTM']].values


#---------------------------------------------------------
# Diccionario que almacena imágenes
#---------------------------------------------------------

# Diccionario para almacenar las imágenes
images_by_date = {}

# Directorio con los archivos .tif
directory = path+'/02-Space-Facilities/'+'ROI-LIST/'

# Iterar sobre los archivos en el directorio
for ifile in range(len(df_roi_list)):
    file_name=df_roi_list.loc[ifile,'Ruta']
    if file_name.endswith('.tif'):
        # Extraer la fecha del nombre del archivo (suponiendo que el nombre del archivo contiene la fecha)
        date = df_roi_list.loc[ifile,'Fecha']  # Ajusta esto según el formato de tus nombres de archivo
        # Cargar la imagen y almacenarla en el diccionario
        images_by_date[date] = load_image(file_name)

#---------------------------------------------------------
# Definición de Bandas y Grupos
#---------------------------------------------------------

nameg = ['agua profunda',
         'agua superficial',
         'terreno natural',
         'relave seco',
         'relave consolidado',
         'relave húmedo',
         'otros']
# Definir los colores para cada grupo
color = np.array([
    [0, 0, 100],       # azul oscuro
    [0, 0, 200],       # azul claro
    [100, 50, 50],     # café oscuro
    [200, 200, 200],   # gris claro
    [100, 100, 100],   # gris oscuro
    [50, 50, 50],      # gris-negro
    [255, 0, 0]])/255.0 # Rojo

# Lista Bandas ............................................................
lam = np.array([492.4, 559.8, 664.6, 832.8, 704.1, 740.5, 782.8, 1613.7, 2202.4, 864.7, 442.7, 945.1])
Nband=["B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A","B01","B09"]
NbandHSL=["B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A","B01","B09","H","S","L"]
Nlam = len(lam)
i_slam = np.argsort(lam)
s_lam = np.arange(1, 10)

# Orden deseado
Nband_sort= ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
#Diccionario Temporal
temp_dict = dict(zip(Nband, lam))
# Crear la lista ordenada de tuplas
lista_ordenada = [(banda, temp_dict[banda]) for banda in Nband_sort]
# Ordenar el diccionario según el orden deseado
diccionario_ordenado = {banda: temp_dict[banda] for banda in sorted(temp_dict, key=lambda x: Nband_sort.index(x))}

#---------------------------------------------------------
# Imagen Visible
#---------------------------------------------------------

fechas=[]
for fecha in df_roi_list['Fecha'].unique():
    fechas.append(fecha)

[Ni, Nj, Nban]= (images_by_date[fecha][:,:,[3,2,1]]).shape
ima = np.zeros((Ni, Nj, 3), dtype=np.uint8)
ima=images_by_date[fecha][:,:,[3,2,1]]

def plot_image(ima):
    ima[:,:,0]=np.floor(255. * ima[:,:,0] / np.max(ima[:,:,0])) #B04 Red
    ima[:,:,1]=np.floor(255. * ima[:,:,1] / np.max(ima[:,:,1])) #B03 Green
    ima[:,:,2]=np.floor(255. * ima[:,:,2] / np.max(ima[:,:,2])) #B02 Blue
    ima=ima.astype(np.uint8)
    fig1=plt.imshow(ima)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

plot_image(ima)

#---------------------------------------------------------
# Imagen Visible
#---------------------------------------------------------
import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact

# Función para ajustar la luminosidad y generar la imagen modificada
def adjust_lightness_and_plot(Light):
    ima_orig = images_by_date[date][:,:,[3,2,1]]/10000.0  # Normaliza la imagen
    Ni, Nj, Nl = ima_orig.shape
    
    # Convertir imagen a formato aplanado para manipulación en HSL
    rgb = ima_orig.reshape(-1, Nl)
    
    # Convertir RGB a HSL
    hsl = f_rgb2hsl(rgb)
    
    # Modificar la luminosidad
    hsl[:, 2] = Light / 100.0  # Ajustar la luminosidad a un valor entre 0 y 1
    
    # Convertir HSL modificado de vuelta a RGB
    rgb_mod = f_hsl2rgb(hsl)
    
    # Reconstruir la imagen en su forma original
    ima_mod = rgb_mod.reshape(Ni, Nj, Nl)
    
    # Normalizar para visualización
    ima_mod = np.clip(ima_mod, 0, 1) * 255
    ima_mod = ima_mod.astype(np.uint8)
    
    # Plotly plotting
    fig = go.Figure(go.Image(z=ima_mod))
    fig.update_layout(
        title=f"Imagen con Luminosidad = {Light}%",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Mantiene la relación de aspecto
    )
    fig.show()

# Usar un slider interactivo para ajustar la luminosidad
interact(adjust_lightness_and_plot, Light=(0, 100, 1))


