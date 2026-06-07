import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat 

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Tk, Canvas
from tkinter import Toplevel, Button
from PIL import Image, ImageTk

import os
import sys
import importlib
import pathlib
from pathlib import Path
from osgeo import gdal, osr
import seaborn as sns

#---------------------------------------------------------
# Input
#---------------------------------------------------------
name = 'Laguna-Seca'
terrain = "02-BAND-SAT"
path = os.path.join('C:/Users/felip/Desktop/Msc-UTFSM', name)
ver = 'v5'

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
df_roi = read_csv_file(f'{path}/02-Space-Facilities/00-ROI-utm.csv','#')
df_roi_list = read_csv_file(f'{path}/02-Space-Facilities/04-ROI-MOD.csv','#')
Mcal = read_csv_file(f'{path}/Mcal_py.csv','#')
McalHSL_mod = read_csv_file(f'{path}/McalHSL_mod_{ver}_py.csv','#')

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

#-------------------------------------------------------------------------------
# FUNCIONES 
#-------------------------------------------------------------------------------

# Cargar el DataFrame con las rutas de las imágenes
def load_dataframe():
    global df_roi_list
    try:
        # Seleccionar archivo CSV
        csv_file = filedialog.askopenfilename(
            title="Seleccionar Archivo CSV",
            filetypes=[("Archivos CSV", "*.csv")],
        )
        
        if not csv_file:  # Si el usuario cancela la selección
            print("No se seleccionó ningún archivo.")
            return

        # Cargar el archivo CSV
        df_roi_list = pd.read_csv(csv_file)

        # Mostrar información del DataFrame
        print("Columnas del DataFrame:", df_roi_list.columns)
        print(df_roi_list.head())

        # Verificar si la columna 'Ruta' existe
        if 'Ruta' not in df_roi_list.columns:
            messagebox.showerror("Error", "La columna 'Ruta' no está en el archivo CSV.")
            return

        # Mostrar un cuadro de diálogo con el nombre del archivo cargado
        messagebox.showinfo("Archivo Cargado", f"Archivo '{csv_file.split('/')[-1]}' cargado correctamente.")

        # Actualizar el Listbox
        update_listbox()

    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el archivo CSV: {e}")

# --- Función para cargar Mcal ---
def load_mcal_dataframe():
    global df_mcal
    file_path = filedialog.askopenfilename(title="Seleccionar Archivo CSV de Mcal",
                                           filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df_mcal = pd.read_csv(file_path)
            print("Mcal cargado con éxito:", df_mcal.head())
            update_listbox_2()
            update_mcal_status(os.path.basename(file_path))
        except Exception as e:
            update_mcal_status("Error")

# Actualizar el Listbox con las rutas de las imágenes
def update_listbox():
    listbox.delete(0, tk.END)
    for idx, row in df_roi_list.iterrows():
        file_name = os.path.basename(row['Ruta'])
        listbox.insert(tk.END, file_name)

# --- Función para actualizar listbox de Mcal ---
def update_listbox_2():
    listbox_2.delete(0, tk.END)
    if df_mcal is not None:
        unique_dates = df_mcal['Fecha'].unique()
        for date in sorted(unique_dates):
            listbox_2.insert(tk.END, date)
    listbox_2.bind("<<ListboxSelect>>", display_selected_mcal)

# --- Función para alternar la visualización de puntos de Mcal ---
def toggle_mcal_points():
    global mcal_points  # Asegurar que usamos la lista global de puntos

    # Verificar si la opción de mostrar puntos está activada
    if show_mcal.get():
        selected_idx = listbox_2.curselection()

        # Si no hay selección en el listbox, no podemos mostrar puntos
        if not selected_idx:
            messagebox.showwarning("Advertencia", "Seleccione una fecha en Mcal para visualizar los puntos.")
            show_mcal.set(False)  # Desmarcar el checkbutton si no hay selección
            return
        
        display_selected_mcal()  # Llamar a la función que dibuja los puntos
        update_mcal_status("Visualización activa")
    else:
        clear_mcal_points()  # Limpiar los puntos si se desactiva la opción
        update_mcal_status("Visualización desactivada")

# --- Función para limpiar los puntos de Mcal en el Canvas ---
def clear_mcal_points():
    global mcal_points

    # Verificar si la ventana sigue activa antes de modificar widgets
    if not root.winfo_exists():
        return  

    if not mcal_points:
        return  # Si la lista está vacía, no hay nada que limpiar

    for point in mcal_points:
        canvas.delete(point)  # Eliminar cada punto dibujado en el Canvas

    mcal_points.clear()  # Vaciar la lista de puntos

    if legend_frame.winfo_exists():  # Verificar si legend_frame sigue existiendo
        legend_frame.pack_forget()  # Ocultar la leyenda

    print("Todos los puntos de Mcal han sido eliminados.")


from osgeo import gdal
import numpy as np
from PIL import Image, ImageTk

def display_selected_image(event):
    global img_tk, image_array, rgb_image_visual  # Asegurarse de declarar rgb_image_visual como global
    selected_idx = listbox.curselection()
    if selected_idx:
        selected_path = df_roi_list.iloc[selected_idx[0]]['Ruta']
        try:
            # Abrir el archivo TIFF con GDAL
            dataset = gdal.Open(selected_path, gdal.GA_ReadOnly)
            if not dataset:
                raise FileNotFoundError(f"No se pudo abrir el archivo: {selected_path}")
            # 🔍 CLASIFICACIÓN
            if 'CLASSMAP' in selected_path:
                print('Cargando imagen de clasificación')
                # Leer la imagen de clases (una banda)
                class_array = dataset.GetRasterBand(1).ReadAsArray().astype(np.int16)
                # Crear imagen RGB a partir de las clases
                img_rgb = np.zeros((class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)
                for cls, rgb in enumerate(color):
                    img_rgb[class_array == cls] = (rgb * 255).astype(np.uint8)
                rgb_image_visual = img_rgb.copy()
                img = Image.fromarray(rgb_image_visual)
                img_tk = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas.image = img_tk
                canvas.config(scrollregion=(0, 0, img_rgb.shape[1], img_rgb.shape[0]))
                #update_image_status(image_name)
                return
            # 🌈 IMAGEN SATELITAL MULTIBANDA
            # Cargar todas las bandas en un arreglo 3D
            num_bands = dataset.RasterCount
            image_array = np.zeros((dataset.RasterYSize, dataset.RasterXSize, num_bands), dtype=np.float32)
            for i in range(num_bands):
                band = dataset.GetRasterBand(i + 1)
                image_array[:, :, i] = band.ReadAsArray()

            # Seleccionar las bandas para la composición RGB (B04, B03, B02)
            band_red = image_array[:, :, 3] / 10000.0  # Normalizar al rango 0-1
            band_green = image_array[:, :, 2] / 10000.0
            band_blue = image_array[:, :, 1] / 10000.0

            # Asegurarse de que los valores estén dentro del rango [0, 1]
            band_red = np.clip(band_red, 0, 1)
            band_green = np.clip(band_green, 0, 1)
            band_blue = np.clip(band_blue, 0, 1)

            # Crear la imagen RGB con valores entre 0 y 1
            rgb_image = np.dstack((band_red, band_green, band_blue))

            # Escalar a 0-255 para visualizar en el canvas
            rgb_image_visual = (rgb_image * 255).astype(np.uint8)  # Actualiza la variable global

            # Ajustar el tamaño del canvas
            img_height, img_width, _ = rgb_image_visual.shape
            canvas.config(scrollregion=(0, 0, img_width, img_height))  # Ajustar la región visible

            # Convertir a formato compatible con Tkinter
            img = Image.fromarray(rgb_image_visual)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk
            
            update_image_status(os.path.basename(selected_path))

        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            update_image_status("Error")

# --- Función para mostrar datos de Mcal ---
import random

def display_selected_mcal(event=None):  # Permitir llamada sin evento
    global mcal_points, color_dict

    selected_idx = listbox_2.curselection()
    if not selected_idx:
        print("No se ha seleccionado una fecha en Mcal.")
        return

    selected_date = listbox_2.get(selected_idx[0])

    if df_mcal is None or df_mcal.empty:
        print("Error: No hay datos en df_mcal.")
        return

    if "Fecha" not in df_mcal.columns or "i" not in df_mcal.columns or "j" not in df_mcal.columns or "Ng" not in df_mcal.columns:
        print("Error: Falta alguna de las columnas requeridas en df_mcal.")
        return

    df_mcal["Fecha"] = df_mcal["Fecha"].astype(str)
    selected_date = str(selected_date)

    filtered_df = df_mcal[df_mcal["Fecha"] == selected_date]

    if filtered_df.empty:
        print(f"No hay datos en Mcal para la fecha {selected_date}")
        return

    unique_groups = filtered_df["Ng"].unique()
    for group in unique_groups:
        if group not in color_dict:
            color_dict[group] = f"#{np.random.randint(0, 0xFFFFFF):06x}"  

    clear_mcal_points()

    for _, row in filtered_df.iterrows():
        x, y, group = row["j"], row["i"], row["Ng"]
        if not (0 <= x <= canvas.winfo_width() and 0 <= y <= canvas.winfo_height()):
            #print(f"Punto ({x}, {y}) fuera del área visible del canvas.")
            continue

        color = color_dict.get(group, "#FF0000")
        point = canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color)
        mcal_points.append(point)

    print(f"{len(mcal_points)} puntos dibujados en el Canvas para la fecha {selected_date}.")
    update_legend()

# --- Función para actualizar la leyenda ---
def update_legend():
    # Verificar si la ventana sigue activa antes de modificar widgets
    if not root.winfo_exists():
        return  

    for widget in legend_frame.winfo_children():
        widget.destroy()  # Limpiar widgets anteriores

    if not color_dict:
        legend_frame.pack_forget()  # Ocultar si no hay datos
        return

    tk.Label(legend_frame, text="Leyenda:", font=("Arial", 10, "bold")).pack()

    for group, color in color_dict.items():
        label = tk.Label(legend_frame, text=f"{group}-{nameg[group-1]}", bg=color, width=15)
        label.pack()

    legend_frame.pack(side=tk.BOTTOM, padx=10, pady=10)


# Graficar la firma espectral al hacer clic en la imagen
def on_canvas_click(event):
    x, y = event.x, event.y
    if image_array is not None:
        print(f"Clic detectado en coordenadas: ({x}, {y})")
        plot_spectral_signature(x, y)

# # Generar el gráfico de la firma espectral
def plot_spectral_signature(x, y):
    if 0 <= x < image_array.shape[1] and 0 <= y < image_array.shape[0]:
        spectral_values = image_array[y, x, :]

        plt.figure(figsize=(8, 5))
        plt.plot(lam_sorted, spectral_values, marker='o', label=f"Píxel ({x}, {y})")
        plt.title("Firma Espectral")
        plt.xlabel("Bandas")
        plt.ylabel("Reflectancia Simulada")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Coordenadas fuera de los límites de la imagen.")


import matplotlib.pyplot as plt
from tkinter import Toplevel, Button
import numpy as np

# Lista para almacenar los píxeles seleccionados
selected_pixels = []

# Crear la ventana secundaria
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

def open_spectral_window():
    global selected_pixels
    selected_pixels = []  # Lista para almacenar píxeles seleccionados

    # Crear la ventana secundaria
    spectral_window = Toplevel(root)
    spectral_window.title("Firmas Espectrales")
    spectral_window.geometry("900x700")

    # Crear un marco para el gráfico
    frame_graph = tk.Frame(spectral_window)
    frame_graph.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Crear el gráfico con Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Firmas Espectrales")
    ax.set_xlabel("Longitud de Onda (nm)")
    ax.set_ylabel("Reflectancia")
    ax.grid(True)

    # Integrar el gráfico en Tkinter
    canvas_graph = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Función para añadir firmas espectrales al gráfico
    def add_spectral_signature(x, y):
        if image_array is not None:
            if 0 <= x < image_array.shape[1] and 0 <= y < image_array.shape[0]:
                spectral_values = image_array[y, x, :]
                ax.plot(lam_sorted, spectral_values, marker='o', label=f"Píxel ({x}, {y})")
                ax.legend(loc="upper right")
                canvas_graph.draw()
                selected_pixels.append((x, y))
                print(f"Píxel añadido: ({x}, {y})")
            else:
                print("Píxel fuera de los límites.")

    # Función para mostrar firmas en todas las fechas
    def show_spectral_all_dates():
        if not selected_pixels:
            print("No hay píxeles seleccionados.")
            return

        for x, y in selected_pixels:
            for idx, row in df_roi_list.iterrows():
                dataset = gdal.Open(row['Ruta'], gdal.GA_ReadOnly)
                if not dataset:
                    print(f"No se pudo abrir el archivo: {row['Ruta']}")
                    continue
                spectral_values = []
                for i in range(dataset.RasterCount):
                    band = dataset.GetRasterBand(i + 1)
                    spectral_values.append(band.ReadAsArray()[y, x])
                ax.plot(lam_sorted, spectral_values, linestyle='--', label=f"Píxel ({x}, {y}) - {row['Fecha']}")
        canvas_graph.draw()

    # Función para limpiar el gráfico
    def clear_selected_pixels():
        global selected_pixels
        selected_pixels = []
        ax.clear()
        ax.set_title("Firmas Espectrales")
        ax.set_xlabel("Longitud de Onda (nm)")
        ax.set_ylabel("Reflectancia")
        ax.grid(True)
        canvas_graph.draw()
        print("Gráfico limpio y píxeles borrados.")

    # Función para exportar las firmas a CSV
    def export_to_csv():
        if not selected_pixels:
            print("No hay firmas espectrales para exportar.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV Files", "*.csv")])
        if save_path:
            data = {"Longitud de Onda (nm)": lam_sorted}
            for x, y in selected_pixels:
                spectral_values = image_array[y, x, :]
                data[f"Píxel ({x}, {y})"] = spectral_values
            df = pd.DataFrame(data)
            df.to_csv(save_path, index=False)
            print(f"Firmas espectrales exportadas a {save_path}")

    # Crear botones en la ventana secundaria
    btn_all_dates = Button(spectral_window, text="Mostrar en Todas las Fechas", command=show_spectral_all_dates)
    btn_all_dates.pack(pady=5)

    btn_clear = Button(spectral_window, text="Borrar Píxeles", command=clear_selected_pixels)
    btn_clear.pack(pady=5)

    btn_export_csv = Button(spectral_window, text="Exportar a CSV", command=export_to_csv)
    btn_export_csv.pack(pady=5)

    # Conectar el canvas principal al gráfico
    def on_pixel_click(event):
        x, y = event.x, event.y
        add_spectral_signature(x, y)

    canvas.bind("<Button-1>", on_pixel_click)  # Conectar clics al canvas principal

# Función para mostrar la firma espectral en todas las fechas
def show_spectral_all_dates():
    if not selected_pixels:
        print("No hay píxeles seleccionados.")
        return

    # Supongamos que `df_roi_list` tiene todas las rutas a las imágenes en fechas distintas
    for x, y in selected_pixels:
        for idx, row in df_roi_list.iterrows():
            dataset = gdal.Open(row['Ruta'], gdal.GA_ReadOnly)
            if not dataset:
                print(f"No se pudo abrir el archivo: {row['Ruta']}")
                continue
            # Extraer las bandas del píxel para esta imagen
            num_bands = dataset.RasterCount
            pixel_spectral_values = []
            for i in range(num_bands):
                band = dataset.GetRasterBand(i + 1)
                pixel_spectral_values.append(band.ReadAsArray()[y, x])
            # Graficar las firmas para todas las fechas
            plt.plot(lam_sorted, pixel_spectral_values, marker='x', label=f"Fecha {row['Fecha']}")
    
    plt.legend(loc="upper right")
    plt.title("Firma Espectral en Todas las Fechas")
    plt.xlabel("Longitud de Onda (nm)")
    plt.ylabel("Reflectancia")
    plt.grid(True)
    plt.show()

# Función para limpiar los píxeles seleccionados
def clear_selected_pixels():
    global selected_pixels
    selected_pixels = []
    plt.clf()  # Limpiar el gráfico
    print("Píxeles seleccionados borrados.")

def save_normalized_image_as_png(rgb_image_visual):
    """
    Guarda la imagen visual (escala 0-255) como PNG o JPEG.

    Parámetros:
    - rgb_image_visual: Arreglo NumPy con valores escalados entre 0 y 255.
    """
    # Pedir al usuario la ubicación y el nombre del archivo
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG Files", "*.png"), 
                                                        ("JPEG Files", "*.jpg")])
    if save_path:
        try:
            # Convertir el arreglo NumPy en una imagen de Pillow y guardar
            img = Image.fromarray(rgb_image_visual)
            img.save(save_path)
            print(f"Imagen guardada como {save_path}")
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")

from tkinter import filedialog
import os
from PIL import Image

def save_all_images():
    """
    Guarda todas las imágenes del listado en una carpeta seleccionada por el usuario.
    """
    global df_roi_list
    if df_roi_list is None or df_roi_list.empty:
        print("No hay imágenes en el listado.")
        return

    # Seleccionar la carpeta destino
    save_folder = filedialog.askdirectory(title="Seleccionar Carpeta de Destino")
    if not save_folder:
        print("No se seleccionó una carpeta.")
        return

    try:
        for idx, row in df_roi_list.iterrows():
            file_path = row['Ruta']
            file_name = os.path.basename(file_path).replace('.tif', '_processed.png')  # Nombre del archivo de salida

            # Abrir la imagen con GDAL
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if not dataset:
                print(f"No se pudo abrir el archivo: {file_path}")
                continue
            
            # Procesar la imagen
            num_bands = dataset.RasterCount
            image_array = np.zeros((dataset.RasterYSize, dataset.RasterXSize, num_bands), dtype=np.float32)
            for i in range(num_bands):
                band = dataset.GetRasterBand(i + 1)
                image_array[:, :, i] = band.ReadAsArray()

            # Crear la imagen RGB
            band_red = np.clip(image_array[:, :, 3] / 10000.0, 0, 1)
            band_green = np.clip(image_array[:, :, 2] / 10000.0, 0, 1)
            band_blue = np.clip(image_array[:, :, 1] / 10000.0, 0, 1)
            rgb_image = np.dstack((band_red, band_green, band_blue))

            # Escalar a 0-255 y convertir a PNG
            rgb_image_visual = (rgb_image * 255).astype(np.uint8)
            img = Image.fromarray(rgb_image_visual)

            # Guardar la imagen
            save_path = os.path.join(save_folder, file_name)
            img.save(save_path)
            print(f"Imagen guardada: {save_path}")

        print("Todas las imágenes se han guardado correctamente.")
    except Exception as e:
        print(f"Error al guardar las imágenes: {e}")

def update_image_status(image_name):
    """Actualiza la barra de estado con el nombre de la imagen cargada."""
    image_status_bar.config(text=f"Imagen: {image_name}")
    root.update_idletasks()  # Refresca la interfaz para que el cambio sea inmediato

def update_mcal_status(message):
    """Actualiza la barra de estado con el estado de Mcal."""
    mcal_status_bar.config(text=f"Mcal: {message}")
    root.update_idletasks()  # Refresca la interfaz para que el cambio sea inmediato


#---------------------------------------------------------
# Ventana Principal
#---------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox

# Crear la ventana principal
root = tk.Tk()
root.title("Visor de Imágenes Satelitales")
root.geometry("1200x700")  # Tamaño inicial de la ventana

# Variables globales
df_roi_list = None
df_mcal = None
image_array = None
img_tk = None
mcal_points = []  # Lista para almacenar los identificadores de los puntos dibujados
color_dict = {}   # Diccionario para almacenar los colores de cada grupo Ng
show_mcal = tk.BooleanVar(value=False)  # Estado del toggle

# --------------------- MENÚ SUPERIOR ---------------------
menubar = tk.Menu(root)

# Menú Archivo
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Cargar CSV", command=load_dataframe)
file_menu.add_command(label="Cargar Mcal CSV", command=load_mcal_dataframe)
file_menu.add_separator()
file_menu.add_command(label="Salir", command=root.quit)
menubar.add_cascade(label="Archivo", menu=file_menu)

# Configurar el menú en la ventana
root.config(menu=menubar)

# --------------------- PANEL IZQUIERDO ---------------------
panel_izquierdo = tk.Frame(root, padx=10, pady=10)
panel_izquierdo.grid(row=0, column=0, sticky="ns")  # Se ancla verticalmente

# --- Listbox de Imágenes ---
lbl_listbox = tk.Label(panel_izquierdo, text="Lista de Imágenes", font=("Arial", 10, "bold"))
lbl_listbox.pack(pady=5)

# Listbox con Scrollbar
frame_listbox = tk.Frame(panel_izquierdo)
frame_listbox.pack()

scroll_listbox_y = tk.Scrollbar(frame_listbox, orient=tk.VERTICAL)
listbox = tk.Listbox(frame_listbox, width=50, height=15, yscrollcommand=scroll_listbox_y.set)
scroll_listbox_y.config(command=listbox.yview)

listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll_listbox_y.pack(side=tk.RIGHT, fill=tk.Y)

listbox.bind("<<ListboxSelect>>", display_selected_image)

# --- Listbox de Mcal ---
lbl_listbox_2 = tk.Label(panel_izquierdo, text="Lista de Fechas Mcal", font=("Arial", 10, "bold"))
lbl_listbox_2.pack(pady=5)

frame_listbox_2 = tk.Frame(panel_izquierdo)
frame_listbox_2.pack()

scroll_listbox_2 = tk.Scrollbar(frame_listbox_2, orient=tk.VERTICAL)
listbox_2 = tk.Listbox(frame_listbox_2, width=50, height=7, yscrollcommand=scroll_listbox_2.set)
scroll_listbox_2.config(command=listbox_2.yview)
listbox_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll_listbox_2.pack(side=tk.RIGHT, fill=tk.Y)
listbox_2.bind("<<ListboxSelect>>", display_selected_mcal)

# --- Botón para mostrar puntos Mcal ---
btn_toggle_mcal = tk.Checkbutton(panel_izquierdo, text="Mostrar Puntos de Mcal", variable=show_mcal, command=toggle_mcal_points)
btn_toggle_mcal.pack(pady=5)

# --- Botones de control ---
frame_buttons = tk.Frame(panel_izquierdo)
frame_buttons.pack(pady=10, fill=tk.X)

btn_open_spectral_window = tk.Button(frame_buttons, text="Abrir Ventana de Firmas", command=open_spectral_window)
btn_open_spectral_window.pack(side=tk.LEFT, padx=2, expand=True)

btn_save_png = tk.Button(frame_buttons, text="Guardar PNG", 
                         command=lambda: save_normalized_image_as_png(rgb_image_visual))
btn_save_png.pack(side=tk.LEFT, padx=2, expand=True)

btn_save_all = tk.Button(frame_buttons, text="Guardar Todo", command=save_all_images)
btn_save_all.pack(side=tk.LEFT, padx=2, expand=True)

# --------------------- CANVAS PARA IMAGEN ---------------------

frame_canvas = tk.Frame(root, bg="white")
frame_canvas.grid(row=0, column=1, sticky="nsew")  # Ocupa todo el espacio restante

scroll_x = tk.Scrollbar(frame_canvas, orient=tk.HORIZONTAL)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

scroll_y = tk.Scrollbar(frame_canvas, orient=tk.VERTICAL)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

canvas = tk.Canvas(frame_canvas, bg="white", 
                   xscrollcommand=scroll_x.set, 
                   yscrollcommand=scroll_y.set)
canvas.pack(fill=tk.BOTH, expand=True)
canvas.bind("<Button-1>", on_canvas_click)

scroll_x.config(command=canvas.xview)
scroll_y.config(command=canvas.yview)

# # --- Marco para la Leyenda ---
legend_frame = tk.Frame(panel_izquierdo, relief=tk.SUNKEN, borderwidth=2)
legend_frame.pack(pady=10, fill=tk.X)
tk.Label(legend_frame, text="Leyenda:", font=("Arial", 10, "bold"), bg="white").pack()

# --------------------- BARRAS DE ESTADO ---------------------

frame_status = tk.Frame(root, relief=tk.SUNKEN, borderwidth=1)
frame_status.grid(row=1, column=0, columnspan=2, sticky="ew")

status_bar = tk.Label(frame_status, text="Listo", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

image_status_bar = tk.Label(frame_status, text="Imagen: Ninguna", bd=1, relief=tk.SUNKEN, anchor=tk.W)
image_status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Se coloca justo encima de status_bar

mcal_status_bar = tk.Label(frame_status, text="Mcal: Ninguno", bd=1, relief=tk.SUNKEN, anchor=tk.W)
mcal_status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)  # Se coloca justo encima de status_bar

# Ajustar tamaño de filas y columnas para expandir canvas
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Ejecutar la aplicación
root.mainloop()
