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
#from f_functions import *

from osgeo import gdal, osr

##--CARGA DE DATOS--##
#---------------------------------------------------------
# Input
#---------------------------------------------------------
#directorio_actual = os.getcwd()
sys.path.append('C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python')

name='Color-example'
#site="05-TSF"
#terrain="02-BAND-SAT"
path = 'C:/Users/felip/Desktop/Msc-UTFSM/' + name

#---------------------------------------------------------
# Read Data
#---------------------------------------------------------

Mcal=pd.read_csv(path+'/'+'Mcal_py.csv',sep=',') #Usando CSV

Mcal=pd.read_csv(path+'/'+'Mcal_py.csv',sep=',') #Usando CSV
#imagen=load_image(path+'/'+'smpte-color-bars-5791787_1280.png')
imagen=plt.imread(path+'/'+'smpte-color-bars-5791787_1280.png')
if imagen.dtype != np.uint8:  # Verificar si ya es un tipo entero
    imagen = (imagen * 255).astype(np.uint8)  # Convertir a enteros de 8 bits (0-255)

# Nombre grupos
nameg = ['Gris',
         'Amarillo',
         'Cian',
         'Verde',
         'Morado',
         'Rojo',
         'Azul']
# Orden deseado
Nband_sort= ["B01", "B02", "B03"]
fecha='24-07-2024'
[Ni, Nj, Nban]= (imagen[:,:,[3,2,1]]).shape

# logica seleccion de puntos de cal .......................
select = np.ones(len(nameg), dtype=int)
# numero de grupos .......................................
Ng = len(nameg)
#Selección General de puntos
MR_ref  = np.zeros((Ng,Nban)); 
MRe_ref =[None]*Ng #lista
Mijp=[None]*Ng #lista

fig, ax = plt.subplots()
ax.imshow(imagen)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title(f'{name}')
plt.gca().set_aspect('equal', adjustable='box')

serie = np.zeros((0, len(['i', 'j', 'Ng', 'B01', 'B02', 'B03'])))  # Empieza vacío

for i_Ng in range(1, Ng + 1):
    print('======================================')
    print(f'{i_Ng} de {Ng}: {nameg[i_Ng - 1]}')
    points_loc = []
    while True:
        point = plt.ginput(1, show_clicks=True)
        if not point:
            break
        x_point, y_point = np.floor(point[0]).astype(int)  # Convertir a enteros
        # Asegúrate de que los puntos estén dentro de los límites de la imagen
        if 0 <= x_point < Nj and 0 <= y_point < Ni:
            # Almacena x, y, índice de grupo, y valores RGB
            serie_aux = np.array([x_point, y_point, i_Ng-1] + imagen[y_point, x_point, 0:3].tolist())
            serie = np.vstack((serie, serie_aux))
        else:
            print("Punto fuera de los límites de la imagen")
        ax.plot(x_point, y_point, 'ro')
        plt.draw()

print(serie)

df1 = pd.DataFrame(serie).astype(int)
df1.insert(0,'Fecha',fecha)
df1.columns=Mcal.columns
#df1=df1.drop(index=0)
Mcal = pd.concat([Mcal, df1], ignore_index=True)
os.chdir(path)
Mcal.to_csv('Mcal_py.csv',index=False)

print(df1)
