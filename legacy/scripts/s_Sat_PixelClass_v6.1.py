# -*- coding: utf-8 -*-
"""
Roi Viewer + Picker (PyQt5 + Matplotlib + GDAL) - VERSIÓN CORREGIDA
"""

import os
import sys
import json
import math
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QWidget, QListWidget, QPushButton, QLabel, QHBoxLayout,
    QVBoxLayout, QSplitter, QSizePolicy, QCheckBox, QListWidgetItem, QStatusBar
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

from f_functions import f_rgb2hsl


# ----------------------------- Utilidades GDAL ----------------------------- #

def gdal_open_readonly(path: str):
    gdal.UseExceptions()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL no pudo abrir: {path}")
    return ds


def read_rgb_thumbnail(ds, rgb_indices: Tuple[int, int, int], max_dim: int = 2048) -> Tuple[np.ndarray, float, float]:
    """Lee un thumbnail RGB uint8 optimizado."""
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize

    if max(ncols, nrows) > max_dim:
        if ncols >= nrows:
            dst_w = max_dim
            dst_h = int(round(nrows * (max_dim / ncols)))
        else:
            dst_h = max_dim
            dst_w = int(round(ncols * (max_dim / nrows)))
    else:
        dst_w, dst_h = ncols, nrows

    scale_x = ncols / dst_w if dst_w else 1.0
    scale_y = nrows / dst_h if dst_h else 1.0

    rgb = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    for k, bidx in enumerate(rgb_indices):
        band = ds.GetRasterBand(bidx + 1)
        arr = band.ReadAsArray(buf_xsize=dst_w, buf_ysize=dst_h)
        if arr is None:
            raise RuntimeError("Error leyendo banda GDAL")
        arr = arr.astype(np.float32)
        m = arr.max()
        if m > 0:
            arr = 255.0 * arr / m
        rgb[:, :, k] = arr

    return rgb.clip(0, 255).astype(np.uint8), scale_y, scale_x


def read_pixel_spectrum(ds, i_full: int, j_full: int) -> List[float]:
    """Lee todas las bandas en un único píxel."""
    nb = ds.RasterCount
    vals = []
    for b in range(1, nb + 1):
        arr = ds.GetRasterBand(b).ReadAsArray(j_full, i_full, 1, 1)
        if arr is None:
            vals.append(np.nan)
        else:
            vals.append(float(arr[0, 0]))
    return vals


# ----------------------------- Canvas Matplotlib ----------------------------- #

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.base_img_artist = None
        self.mcal_scatter = None
        self.picks_scatter = None

    def show_image(self, img: np.ndarray):
        self.ax.clear()
        self.base_img_artist = self.ax.imshow(img, zorder=0)
        self.ax.set_xlabel("X [px]")
        self.ax.set_ylabel("Y [px]")
        self.ax.set_aspect("equal", adjustable="box")
        
        # Inicializar scatter plots para overlays
        self.mcal_scatter = self.ax.scatter([], [], s=20, alpha=0.7, zorder=2)
        self.picks_scatter = self.ax.scatter([], [], s=30, alpha=0.9, zorder=3, 
                                           color='red', marker='x', linewidth=2)
        self.draw_idle()

    def update_overlay_mcal(self, points: List[Tuple[float, float]], colors: List[Tuple[float, float, float, float]]):
        """Actualiza puntos de Mcal usando scatter plot"""
        if self.mcal_scatter is None or not points:
            self.mcal_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.mcal_scatter.set_offsets(points)
            self.mcal_scatter.set_color(colors)
        self.draw_idle()

    def update_overlay_picks(self, points: List[Tuple[float, float]]):
        """Actualiza picks de sesión usando scatter plot"""
        if self.picks_scatter is None or not points:
            self.picks_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.picks_scatter.set_offsets(points)
        self.draw_idle()


# ----------------------------- Ventana Principal ----------------------------- #

class RoiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Viewer/Picker (S2) - CORREGIDO")

        # Estado
        self.config: Dict = {}
        self.group_names: List[str] = []
        self.rgb_indices: Tuple[int, int, int] = (3, 2, 1)
        self.df_roi: Optional[pd.DataFrame] = None
        self.df_mcal: Optional[pd.DataFrame] = None
        self.current_ds = None
        self.current_thumb: Optional[np.ndarray] = None
        self.scale_y: float = 1.0
        self.scale_x: float = 1.0
        self.overlay_toggle: bool = True
        self.capture_enabled: bool = True

        # picks de la sesión
        self.session_picks: List[Dict] = []

        # Widgets
        self.btn_load_cfg = QPushButton("Cargar Config")
        self.btn_load_roi = QPushButton("Cargar ROI list")
        self.btn_load_mcal = QPushButton("Cargar Mcal")
        self.btn_toggle_overlay = QPushButton("Puntos Mcal: ON"); 
        self.btn_toggle_overlay.setCheckable(True); 
        self.btn_toggle_overlay.setChecked(True)
        self.btn_capture = QPushButton("Capturar: ON"); 
        self.btn_capture.setCheckable(True); 
        self.btn_capture.setChecked(True)
        self.btn_save_csv = QPushButton("Guardar picks → CSV")
        self.btn_clear_picks = QPushButton("Limpiar Picks")  # Nuevo botón

        self.lbl_cfg = QLabel("Config: —")
        self.lbl_roi = QLabel("ROI list: —")
        self.lbl_mcal = QLabel("Mcal: —")

        self.list_images = QListWidget()
        self.list_groups = QListWidget()
        self.list_groups.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_picks = QListWidget()

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Layouts
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_load_cfg)
        top_bar.addWidget(self.btn_load_roi)
        top_bar.addWidget(self.btn_load_mcal)
        top_bar.addWidget(self.btn_toggle_overlay)
        top_bar.addWidget(self.btn_capture)
        top_bar.addWidget(self.btn_save_csv)
        top_bar.addWidget(self.btn_clear_picks)
        top_bar.addStretch(1)

        info_bar = QHBoxLayout()
        info_bar.addWidget(self.lbl_cfg)
        info_bar.addWidget(self.lbl_roi)
        info_bar.addWidget(self.lbl_mcal)
        info_bar.addStretch(1)
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Panel izquierdo (listas)
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Imágenes (por fecha):"))
        left_panel.addWidget(self.list_images)
        left_panel.addWidget(QLabel("Grupos (selecciona para pick):"))
        left_panel.addWidget(self.list_groups)
        left_panel.addWidget(QLabel("Picks (sesión):"))
        left_panel.addWidget(self.list_picks)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)

        # Panel derecho (canvas + toolbar)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.canvas)
        right_panel.addWidget(self.toolbar)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)

        splitter = QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addLayout(top_bar)
        outer.addLayout(info_bar)
        outer.addWidget(splitter)
        self.setCentralWidget(central)

        # Signals
        self.btn_load_cfg.clicked.connect(self.on_load_config)
        self.btn_load_roi.clicked.connect(self.on_load_roi)
        self.btn_load_mcal.clicked.connect(self.on_load_mcal)
        self.btn_toggle_overlay.toggled.connect(self.on_toggle_overlay)
        self.btn_capture.toggled.connect(self.on_toggle_capture)
        self.btn_save_csv.clicked.connect(self.on_save_csv)
        self.btn_clear_picks.clicked.connect(self.on_clear_picks)
        self.list_images.currentRowChanged.connect(self.on_select_image)
        self.list_picks.itemChanged.connect(self.on_pick_item_changed)

        # Picking
        self.canvas.mpl_connect('button_release_event', self.on_canvas_click)

    def closeEvent(self, event):
        """Cerrar recursos GDAL al salir"""
        if self.current_ds is not None:
            self.current_ds = None
        event.accept()

    # ------------------- Actions ------------------- #
    def on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona config_bandas.json", "", "JSON (*.json);;Todos (*.*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.group_names = list(self.config.get("nameg", []))
            if not self.group_names:
                raise ValueError("Config no contiene 'nameg'")
            if "rgb_indices" in self.config and len(self.config["rgb_indices"]) == 3:
                idx = self.config["rgb_indices"]
                self.rgb_indices = (idx[0]-1, idx[1]-1, idx[2]-1) if min(idx) == 1 else tuple(idx)
            else:
                self.rgb_indices = (3, 2, 1)
            self.list_groups.clear()
            self.list_groups.addItems(self.group_names)
            self.lbl_cfg.setText(f"Config: {os.path.basename(path)}")
            self._refresh_all_overlays()
        except Exception as e:
            self._error(f"Error al cargar config: {e}")

    def on_load_roi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona ROI list (CSV)", "", "CSV (*.csv);;Todos (*.*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, comment="#")
            needed = {"Fecha", "Ruta"}
            if not needed.issubset(df.columns):
                raise ValueError(f"ROI list debe contener columnas {needed}")
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["Fecha"]).copy()
            df = df.sort_values("Fecha").reset_index(drop=True)
            self.df_roi = df
            self.list_images.clear()
            for _, row in df.iterrows():
                fecha_str = row["Fecha"].strftime("%Y-%m-%d")
                self.list_images.addItem(f"{fecha_str} | {os.path.basename(str(row['Ruta']))}")
            self.lbl_roi.setText(f"ROI list: {os.path.basename(path)}")
        except Exception as e:
            self._error(f"Error al cargar ROI list: {e}")

    def on_load_mcal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona Mcal_py.csv", "", "CSV (*.csv);;Todos (*.*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, comment="#", na_values=['', ' ', 'NaN', 'N/A'])
            
            needed = {"Fecha", "i", "j", "Ng"}
            if not needed.issubset(df.columns):
                raise ValueError(f"Mcal debe contener columnas {needed}")
            
            # Limpiar datos
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["Fecha"]).copy()
            
            for c in ["i", "j", "Ng"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")
            
            # Asegurar que tenemos todas las bandas y HSL
            bandas = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            for banda in bandas:
                if banda in df.columns:
                    df[banda] = pd.to_numeric(df[banda], errors='coerce').fillna(0).astype('Int64')
                else:
                    df[banda] = 0
            
            for col in ['H', 'S', 'L']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = 0.0
            
            self.df_mcal = df
            self.lbl_mcal.setText(f"Mcal: {os.path.basename(path)}")
            self._refresh_all_overlays()
            
        except Exception as e:
            self._error(f"Error al cargar Mcal: {e}")
    
    
    def _clean_mcal_data(self, df):
        """Limpia y formatea los datos del Mcal"""
        df_clean = df.copy()
        
        # 1. Formatear fecha consistentemente
        df_clean["Fecha"] = pd.to_datetime(df_clean["Fecha"], errors="coerce").dt.normalize()
        df_clean = df_clean.dropna(subset=["Fecha"]).copy()
        
        # 2. Convertir columnas numéricas, manejando valores vacíos
        for c in ["i", "j", "Ng"]:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0).astype("Int64")
        
        # 3. Manejar bandas espectrales (B01-B12, B8A)
        bandas = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        for banda in bandas:
            if banda in df_clean.columns:
                # Convertir a numérico, llenar NaN con 0 y convertir a entero
                df_clean[banda] = pd.to_numeric(df_clean[banda], errors='coerce').fillna(0).round().astype('Int64')
            else:
                # Si falta la banda, crear columna con 0
                df_clean[banda] = 0
        
        # 4. Manejar columnas HSL
        for col in ['H', 'S', 'L']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
            else:
                df_clean[col] = 0.0
        
        # 5. Filtrar filas con coordenadas válidas
        df_clean = df_clean[(df_clean['i'] > 0) & (df_clean['j'] > 0)].copy()
        
        # 6. Log para debugging
        grupos = df_clean['Ng'].unique()
        print(f"Grupos cargados: {sorted(grupos)}")
        print(f"Total puntos: {len(df_clean)}")
        for grupo in sorted(grupos):
            count = len(df_clean[df_clean['Ng'] == grupo])
            print(f"Grupo {grupo}: {count} puntos")
        
        return df_clean

    def on_toggle_overlay(self, checked: bool):
        self.overlay_toggle = checked
        self.btn_toggle_overlay.setText("Puntos Mcal: ON" if checked else "Puntos Mcal: OFF")
        self._refresh_all_overlays()

    def on_toggle_capture(self, checked: bool):
        self.capture_enabled = checked
        self.btn_capture.setText("Capturar: ON" if checked else "Capturar: OFF")
        self.status.showMessage("Captura habilitada" if checked else "Captura deshabilitada", 3000)

    def on_save_csv(self):
        if not self.session_picks:
            QMessageBox.information(self, "Guardar", "No hay picks para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "Mcal_py_new.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            # construir DataFrame SOLO desde session_picks
            cols = ['Fecha','i','j','Ng','B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12','H','S','L']
            rows = []
            for rec in self.session_picks:
                row = {c: rec.get(c, 0) for c in cols}
                rows.append(row)
            df_new = pd.DataFrame(rows)
            # merge opcional con Mcal cargado
            if self.df_mcal is not None:
                df_all = pd.concat([self.df_mcal, df_new], ignore_index=True)
                df_all['Fecha'] = pd.to_datetime(df_all['Fecha'], errors='coerce').dt.normalize()
                df_all = df_all.drop_duplicates(subset=['Fecha','i','j','Ng'], keep='last')
            else:
                df_all = df_new
            tmp = path + ".tmp"
            df_all.to_csv(tmp, index=False)
            os.replace(tmp, path)
            QMessageBox.information(self, "Guardar", f"Guardado: {path}\nPicks: {len(self.session_picks)}")
        except Exception as e:
            self._error(f"Error al guardar: {e}\n{traceback.format_exc()}")

    def on_clear_picks(self):
        """Limpiar todos los picks de la sesión"""
        if self.session_picks:
            reply = QMessageBox.question(self, "Limpiar Picks", 
                                       f"¿Eliminar todos los {len(self.session_picks)} picks?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.session_picks.clear()
                self.list_picks.clear()
                self._refresh_all_overlays()

    def on_select_image(self, row: int):
        if row < 0 or self.df_roi is None or row >= len(self.df_roi):
            return
            
        # Cerrar dataset anterior
        if self.current_ds is not None:
            self.current_ds = None
            
        rec = self.df_roi.iloc[row]
        tif_path = str(rec["Ruta"])
        
        if not os.path.isfile(tif_path):
            self._error(f"Archivo no existe: {tif_path}")
            return
            
        try:
            ds = gdal_open_readonly(tif_path)
            rgb, sy, sx = read_rgb_thumbnail(ds, self.rgb_indices, max_dim=2048)
            self.current_ds = ds
            self.current_thumb = rgb
            self.scale_y, self.scale_x = sy, sx
            self.canvas.show_image(rgb)
            self._refresh_all_overlays()
            
        except Exception as e:
            self._error(f"Error al cargar imagen: {e}\n{traceback.format_exc()}")

    def on_pick_item_changed(self, item):
        """Cuando cambia el checkbox de un pick"""
        row = self.list_picks.row(item)
        if 0 <= row < len(self.session_picks):
            self.session_picks[row]['visible'] = (item.checkState() == QtCore.Qt.Checked)
            self._refresh_all_overlays()

    # ------------------- Picking ------------------- #
    def on_canvas_click(self, event):
        if not self.capture_enabled:
            return
        if hasattr(self.toolbar, '_active') and self.toolbar._active:
            return
        if event.inaxes != self.canvas.ax or event.button != 1:
            return
        if self.current_ds is None or self.current_thumb is None:
            return
            
        grp_row = self.list_groups.currentRow()
        if grp_row < 0 or grp_row >= len(self.group_names):
            self.status.showMessage("ERROR: Selecciona un grupo primero", 3000)
            return
            
        img_row = self.list_images.currentRow()
        if self.df_roi is None or img_row < 0:
            return
            
        fecha_sel = self.df_roi.iloc[img_row]["Fecha"]
        j_th = int(round(event.xdata))
        i_th = int(round(event.ydata))
        H, W = self.current_thumb.shape[:2]
        
        if i_th < 0 or j_th < 0 or i_th >= H or j_th >= W:
            return
            
        j_full = int(j_th * self.scale_x)
        i_full = int(i_th * self.scale_y)
        
        try:
            spec = read_pixel_spectrum(self.current_ds, i_full, j_full)
            
            # CALCULAR HSL USANDO TU FUNCIÓN
            # Asumimos que spec tiene las bandas en orden: [B01, B02, B03, B04, ...]
            # Para RGB usamos B04 (Red), B03 (Green), B02 (Blue)
            if len(spec) >= 4:
                # Crear array RGB normalizado [0-1] usando B04, B03, B02
                rgb_values = np.array([
                    spec[3] / 10000.0,  # B04 - Red (normalizar Sentinel-2)
                    spec[2] / 10000.0,  # B03 - Green  
                    spec[1] / 10000.0   # B02 - Blue
                ])
                
                # Limitar valores entre 0 y 1
                rgb_values = np.clip(rgb_values, 0.0, 1.0)
                
                # Calcular HSL usando tu función
                hsl_result = f_rgb2hsl(rgb_values)
                H_val, S_val, L_val = hsl_result[0], hsl_result[1], hsl_result[2]
            else:
                H_val, S_val, L_val = 0.0, 0.0, 0.0
            
            rec = {
                "Fecha": fecha_sel.strftime("%Y-%m-%d"),
                "i": i_full, "j": j_full,
                "i_th": i_th, "j_th": j_th,
                "Ng": int(grp_row + 1),
                "Grupo": self.group_names[grp_row],
                "visible": True,
                # AGREGAR HSL CALCULADO
                "H": float(H_val),
                "S": float(S_val), 
                "L": float(L_val)
            }
            
            # Agregar TODAS las bandas espectrales en el orden correcto de Sentinel-2
            bandas_sentinel2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            
            # Asumiendo que 'spec' viene en el orden correcto de bandas
            for idx, banda in enumerate(bandas_sentinel2):
                if idx < len(spec):
                    rec[banda] = int(spec[idx])
                else:
                    rec[banda] = 0  # Valor por defecto si no hay suficientes bandas
                    
            # Insertar al inicio
            self.session_picks.insert(0, rec)
            item = QListWidgetItem(f"{rec['Fecha']}  i={rec['i']} j={rec['j']}  Ng={rec['Ng']} ({rec['Grupo']})")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.list_picks.insertItem(0, item)
            
            self._refresh_all_overlays()
            self.status.showMessage(f"Pick agregado: {rec['Grupo']} en ({i_full}, {j_full})", 3000)
            
        except Exception as e:
            self._error(f"Error al leer pixel: {e}")
    # ------------------- Overlays ------------------- #
    def _refresh_all_overlays(self):
        if self.current_thumb is None:
            return
        
        H, W = self.current_thumb.shape[:2]
        
        # 1) Mcal existente
        mcal_points = []
        mcal_colors = []

        if self.overlay_toggle and self.df_mcal is not None and len(self.group_names) > 0:
            row = self.list_images.currentRow()
            if row >= 0 and self.df_roi is not None:
                fecha_sel = self.df_roi.iloc[row]["Fecha"].date()
                df_day = self.df_mcal[self.df_mcal["Fecha"].dt.date == fecha_sel]
                
                print(f"=== DEBUG OVERLAY Mcal ===")
                print(f"Fecha: {fecha_sel}")
                print(f"Puntos en df_day: {len(df_day)}")
                
                # DEBUG: VERIFICAR COLORES PARA TODOS LOS GRUPOS
                print("\nMuestra de colores para todos los grupos:")
                for ng in range(1, len(self.group_names) + 1):
                    if ng <= len(self.group_names):
                        color = _rgba_color(ng - 1, len(self.group_names), alpha=200)
                        grupo_nombre = self.group_names[ng-1]
                        print(f"  Ng={ng} ({grupo_nombre}): color={color}")
                
                for idx, point in df_day.iterrows():
                    j_th = point["j"] / self.scale_x
                    i_th = point["i"] / self.scale_y
                    
                    if 0 <= j_th < W and 0 <= i_th < H:
                        mcal_points.append([j_th, i_th])
                        
                        ng_value = point["Ng"]
                        
                        # CORRECCIÓN ROBUSTA
                        if pd.isna(ng_value):
                            color = (1.0, 0.0, 0.0, 0.8)  # Rojo para NaN
                        elif ng_value < 1 or ng_value > len(self.group_names):
                            print(f"  Ng={ng_value} FUERA DE RANGO [1-{len(self.group_names)}]")
                            color = (1.0, 0.5, 0.0, 0.8)  # Naranja para fuera de rango
                        else:
                            color_index = int(ng_value) - 1
                            color = _rgba_color(color_index, len(self.group_names), alpha=200)
                        
                        mcal_colors.append(color)
                
                print(f"Total puntos en overlay: {len(mcal_points)}")
                print("=== FIN DEBUG ===\n")

        self.canvas.update_overlay_mcal(mcal_points, mcal_colors)
        
        # 2) Picks de sesión
        pick_points = []
        
        if self.session_picks:
            img_row = self.list_images.currentRow()
            if img_row >= 0 and self.df_roi is not None:
                fecha_sel = self.df_roi.iloc[img_row]["Fecha"].strftime("%Y-%m-%d")
                
                for idx in range(min(len(self.session_picks), self.list_picks.count())):
                    rec = self.session_picks[idx]
                    if rec.get("Fecha") != fecha_sel:
                        continue
                        
                    item = self.list_picks.item(idx)
                    if item and item.checkState() == QtCore.Qt.Checked:
                        j_th = rec.get("j_th", 0)
                        i_th = rec.get("i_th", 0)
                        pick_points.append([j_th, i_th])
        
        self.canvas.update_overlay_picks(pick_points)

    def _error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.status.showMessage(f"ERROR: {msg}", 5000)


# ----------------------------- Utilidades de color ----------------------------- #

def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Genera una paleta de colores equidistantes en el espacio de color HSV
    para cualquier número de grupos.
    """
    if num_colors <= 0:
        return [(31, 119, 180)]  # Color por defecto
    
    colors = []
    for i in range(num_colors):
        # Usar HSV para colores equidistantes, convertir a RGB
        hue = i / max(num_colors, 1)  # Distribuir en el círculo de color (0-1)
        saturation = 0.8  # Saturación alta para colores vibrantes
        value = 0.9       # Valor alto para buena visibilidad
        
        # Convertir HSV a RGB
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)
        
        if h_i == 0:
            r, g, b = value, t, p
        elif h_i == 1:
            r, g, b = q, value, p
        elif h_i == 2:
            r, g, b = p, value, t
        elif h_i == 3:
            r, g, b = p, q, value
        elif h_i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q
        
        # Convertir a 0-255 y redondear
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        
        colors.append((r, g, b))
    
    return colors

def _rgba_color(i: int, num_groups: int, alpha: int = 160) -> Tuple[float, float, float, float]:
    """
    Genera color RGBA basado en el índice y el número total de grupos.
    """
    if not hasattr(_rgba_color, 'palette_cache'):
        _rgba_color.palette_cache = {}
    
    # Cachear paletas por número de grupos
    if num_groups not in _rgba_color.palette_cache:
        _rgba_color.palette_cache[num_groups] = generate_color_palette(num_groups)
    
    palette = _rgba_color.palette_cache[num_groups]
    r, g, b = palette[i % len(palette)]
    return r/255.0, g/255.0, b/255.0, alpha/255.0

# Paleta por defecto para casos especiales
_DEFAULT_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
]


# ----------------------------- main ----------------------------- #

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = RoiViewer()
    w.resize(1300, 850)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()