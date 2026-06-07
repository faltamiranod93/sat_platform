#---------------------------------------------------------
# Librerías
#---------------------------------------------------------
from osgeo import gdal
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys
import importlib
from datetime import datetime

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
# Ruta Inicial
#---------------------------------------------------------

ruta_inicial = path

# Obtener las rutas de los archivos .jp2
rutas_archivos_jp2 = explorar_carpetas(ruta_inicial,'.jp2')
df_resultado = procesar_archivos(rutas_archivos_jp2)
print(df_resultado)
# Guardado de las rutas
os.chdir(path + '/01-Terrain/')
df_resultado.to_csv('02-BAND-SAT-RAS.csv', index=False)

