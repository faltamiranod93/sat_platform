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
import ipywidgets as widgets
from IPython.display import display
import networkx as nx
from osgeo import gdal, osr



##--CARGA DE DATOS--##
#---------------------------------------------------------
# Input
#---------------------------------------------------------
#directorio_actual = os.getcwd()
sys.path.append('C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python')

name='Laguna-Seca'
#site="05-TSF"
terrain="02-BAND-SAT"
path = 'C:/Users/felip/Desktop/Msc-UTFSM/' + name

#---------------------------------------------------------
# Read Data
#---------------------------------------------------------
# Read ROI of site ..............................
os.chdir(path + '/02-Space-Facilities/')
df_roi = pd.read_csv('00-ROI-utm.csv',comment='#')
ip1 = df_roi[df_roi['pl'] == 'p1'].index[0]
ip2 = df_roi[df_roi['pl'] == 'p2'].index[0]

# Puntos extremos ROI ............................
p1 = df_roi.loc[ip1,['xUTM','yUTM']].values
p2 = df_roi.loc[ip2,['xUTM','yUTM']].values

# Read SAT Raster ..............................
os.chdir(path + '/02-Space-Facilities/')
df_roi_list = pd.read_csv('03-ROI-LIST.csv',comment='#')
#N_ban = len(df_ban['DAT'])

Mcal=pd.read_csv(path+'/'+'Mcal_py.csv',sep=',') #Usando CSV

# Configuración de parámetros
pixel_width = 3
pixel_height = 9

# Asumiendo que Mcal es tu DataFrame original
fechas = Mcal['Fecha'].unique()
groups = Mcal['Ng'].unique()

# Normalizar las bandas y convertirlas a uint8
blue_band = np.floor(Mcal['B02'] * 255. / 10000).astype(np.uint8)
green_band = np.floor(Mcal['B03'] * 255. / 10000).astype(np.uint8)
red_band = np.floor(Mcal['B04'] * 255. / 10000).astype(np.uint8)

# Crear el directorio para guardar las imágenes
output_dir = path + '/03-Report/'
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre cada grupo y fecha
for group in groups:
    df_group = Mcal[Mcal['Ng'] == group]
    for fecha in fechas:
        df_fecha = df_group[df_group['Fecha'] == fecha]
        if df_fecha.empty:
            continue

        image_rows = []

        for idx, row in df_fecha.iterrows():
            # Crear un bloque de 3x3 píxeles para cada punto
            blue_block = np.full((pixel_height, pixel_width), blue_band[idx])
            green_block = np.full((pixel_height, pixel_width), green_band[idx])
            red_block = np.full((pixel_height, pixel_width), red_band[idx])

            # Combinar las bandas en un solo bloque RGB
            block = np.stack((red_block, green_block, blue_block), axis=2)

            # Colocar el bloque en la lista de filas
            image_rows.append(block)

        # Combinar los bloques en una sola imagen
        group_image = np.hstack(image_rows)

        # Guardar la imagen del grupo para la fecha
        plt.figure(figsize=(15, 10))
        plt.imshow(group_image)
        plt.axis('off')
        output_file = os.path.join(output_dir, f'group_{group}_fecha_{fecha}.jpg')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Imagen guardada: {output_file}')
