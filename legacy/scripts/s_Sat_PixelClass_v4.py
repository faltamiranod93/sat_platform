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
from matplotlib.widgets import Button

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

#---------------------------------------------------------
#Función para cargar una imagen y convertirla en un np.array
def load_image(file_path):
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
    if not dataset:
        raise FileNotFoundError(f'No se pudo abrir el archivo: {file_path}')
    
    num_bands = dataset.RasterCount
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    image_array = np.zeros((y_size, x_size, num_bands), dtype=np.float32)
    
    for i in range(num_bands):
        band = dataset.GetRasterBand(i + 1)
        image_array[:, :, i] = band.ReadAsArray()
    
    dataset = None
    return image_array

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

nameg = ['agua profunda',
         'agua superficial',
         'terreno natural',
         'relave seco',
         'relave consolidado',
         'relave húmedo',
         'otros']

# Lista Bandas ............................................................
lam = np.array([492.4, 559.8, 664.6, 832.8, 704.1, 740.5, 782.8, 1613.7, 2202.4, 864.7, 442.7, 945.1])
Nband=["B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A","B01","B09"]
NbandHSL=["B02","B03","B04","B08","B05","B06","B07","B11","B12","B8A","B01","B09","H","S","L"]
Nlam = len(lam)
i_slam = np.argsort(lam)
s_lam = np.arange(0, 12)


# Orden deseado
Nband_sort= ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
#Diccionario Temporal
temp_dict = dict(zip(Nband, lam))
# Crear la lista ordenada de tuplas
lista_ordenada = [(banda, temp_dict[banda]) for banda in Nband_sort]
# Ordenar el diccionario según el orden deseado
diccionario_ordenado = {banda: temp_dict[banda] for banda in sorted(temp_dict, key=lambda x: Nband_sort.index(x))}

##--CLASIFICACIÓN PIXEL--##............................
fechas=df_roi_list['Fecha'].unique()
fecha=fechas[0]

[Ni, Nj, Nban]= (images_by_date[fecha][:,:,[3,2,1]]).shape
ima = np.zeros((Ni, Nj, 3), dtype=np.uint8)
ima=images_by_date[fecha][:,:,[3,2,1]]
ima[:,:,0]=np.floor(255. * ima[:,:,0] / np.max(ima[:,:,0])) #B04 Red
ima[:,:,1]=np.floor(255. * ima[:,:,1] / np.max(ima[:,:,1])) #B03 Green
ima[:,:,2]=np.floor(255. * ima[:,:,2] / np.max(ima[:,:,2])) #B02 Blue
ima=ima.astype(np.uint8)


# logica de seleccion de area de interes ..................     
roi = np.ones(len(nameg), dtype=int)
# logica seleccion de puntos de cal .......................
select = np.ones(len(nameg), dtype=int)
# numero de grupos .......................................
Ng = len(nameg)
# Lista Bandas ............................................................
lam = [492.4, 559.8, 664.6, 832.8, 704.1, 740.5, 782.8, 1613.7, 2202.4, 864.7, 442.7, 945.1]
Nlam = len(lam)
i_slam = np.argsort(lam)
s_lam = np.arange(0, 12)


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

# Variables para almacenar puntos seleccionados
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

# Botón para continuar al siguiente grupo
axnext = plt.axes([0.8, 0.01, 0.1, 0.075])
bnext = Button(axnext, 'Next Group')
bnext.on_clicked(next_group)

# Botón para deshacer
axundo = plt.axes([0.6, 0.01, 0.1, 0.075])
bundo = Button(axundo, 'Undo Last')
bundo.on_clicked(undo_last_point)

# Botón para restablecer el zoom
axreset = plt.axes([0.4, 0.01, 0.1, 0.075])
breset = Button(axreset, 'Reset Zoom')

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
        rgb_values = images_by_date[fecha][y_point, x_point, :].tolist()
        serie_aux = np.array([y_point, x_point, current_group_index[0] + 1] + rgb_values)
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
Mcal.to_csv('Mcal_py.csv',index=False)

print(df1)
