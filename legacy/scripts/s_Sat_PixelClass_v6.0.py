# -*- coding: utf-8 -*-
"""
Roi Viewer + Picker (PyQt5 + Matplotlib + GDAL)
Cumple:
- Cargar config desde GUI (JSON con nameg y opcional rgb_indices [3,2,1] 0/1-based)
- Cargar ROI list desde GUI (CSV con columnas mínimas: Fecha, Ruta)
- Cargar Mcal_py.csv desde GUI (Fecha, i, j, Ng, ...)
- Lista de imágenes ordenadas por fecha
- Lista de grupos desde config
- Mostrar imagen seleccionada
- Overlay con puntos de Mcal por fecha (toggle ON/OFF) como "pixeles" coloreados (no círculos)
- Picking: seleccionar grupo en la lista y hacer click sobre la imagen para agregar puntos (se listan a la vista)

Notas:
- Este archivo muestra y permite selección básica **y persistencia a CSV** (merge opcional con Mcal existente) con escritura atómica.
- Overlay: Mcal (capa 1) + picks sesión (capa 2). Ambos son RGBA sobrepuestos con zorder explícito. Los picks nuevos se pintan en **rojo**.
- Añadido botón **Capturar: ON/OFF** para habilitar/deshabilitar picking. Si está OFF o si la toolbar está en modo zoom/pan, no se capturan clics.
- Lista de picks en **orden inverso (último arriba)**. Lista con checkboxes: selección/deselección controla visibilidad en el mapa.
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


# ----------------------------- Utilidades GDAL ----------------------------- #

def gdal_open_readonly(path: str):
    gdal.UseExceptions()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL no pudo abrir: {path}")
    return ds


def read_rgb_thumbnail(ds, rgb_indices: Tuple[int, int, int], max_dim: int = 2048) -> Tuple[np.ndarray, float, float]:
    """Lee un thumbnail RGB uint8 optimizado. Devuelve (rgb, scale_y, scale_x) donde
    scale_* = tamaño_full / tamaño_thumb. rgb_indices son 0-based.
    """
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
    """Lee todas las bandas en un único píxel (fila i, col j). Retorna lista de float."""
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
        self.overlay_mcal_artist = None
        self.overlay_picks_artist = None

    def show_image(self, img: np.ndarray):
        self.ax.clear()
        self.base_img_artist = self.ax.imshow(img, zorder=0)
        self.ax.set_xlabel("X [px]")
        self.ax.set_ylabel("Y [px]")
        self.ax.set_aspect("equal", adjustable="box")
        # placeholders de overlays con zorder fijo
        H, W = img.shape[:2]
        empty = np.zeros((H, W, 4), dtype=np.uint8)
        self.overlay_mcal_artist = self.ax.imshow(empty.copy(), zorder=1, interpolation='nearest')
        self.overlay_picks_artist = self.ax.imshow(empty.copy(), zorder=2, interpolation='nearest')
        self.draw_idle()

    def update_overlay_mcal(self, overlay_rgba: Optional[np.ndarray]):
        if self.overlay_mcal_artist is None:
            return
        if overlay_rgba is None:
            H, W = self.overlay_mcal_artist.get_array().shape[:2]
            overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        self.overlay_mcal_artist.set_data(overlay_rgba)
        self.draw_idle()

    def update_overlay_picks(self, overlay_rgba: Optional[np.ndarray]):
        if self.overlay_picks_artist is None:
            return
        if overlay_rgba is None:
            H, W = self.overlay_picks_artist.get_array().shape[:2]
            overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        self.overlay_picks_artist.set_data(overlay_rgba)
        self.draw_idle()

# ----------------------------- Ventana Principal ----------------------------- #

class RoiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Viewer/Picker (S2)")

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
        self.session_picks: List[Dict] = []  # cada rec incluye 'visible': bool

        # Widgets
        self.btn_load_cfg = QPushButton("Cargar Config")
        self.btn_load_roi = QPushButton("Cargar ROI list")
        self.btn_load_mcal = QPushButton("Cargar Mcal")
        self.btn_toggle_overlay = QPushButton("Puntos Mcal: ON"); self.btn_toggle_overlay.setCheckable(True); self.btn_toggle_overlay.setChecked(True)
        self.btn_capture = QPushButton("Capturar: ON"); self.btn_capture.setCheckable(True); self.btn_capture.setChecked(True)
        self.btn_save_csv = QPushButton("Guardar picks → CSV")

        self.lbl_cfg = QLabel("Config: —")
        self.lbl_roi = QLabel("ROI list: —")
        self.lbl_mcal = QLabel("Mcal: —")

        self.list_images = QListWidget()
        self.list_groups = QListWidget(); self.list_groups.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_picks = QListWidget()  # listado de picks con checkboxes

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
        top_bar.addStretch(1)

        info_bar = QHBoxLayout()
        info_bar.addWidget(self.lbl_cfg)
        info_bar.addWidget(self.lbl_roi)
        info_bar.addWidget(self.lbl_mcal)
        info_bar.addStretch(1)
        self.status = QStatusBar(); self.setStatusBar(self.status)

        # Panel izquierdo (listas)
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Imágenes (por fecha):"))
        left_panel.addWidget(self.list_images)
        left_panel.addWidget(QLabel("Grupos (selecciona para pick):"))
        left_panel.addWidget(self.list_groups)
        left_panel.addWidget(QLabel("Picks (sesión):"))
        left_panel.addWidget(self.list_picks)
        left_widget = QWidget(); left_widget.setLayout(left_panel)

        # Panel derecho (canvas + toolbar)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.canvas)
        right_panel.addWidget(self.toolbar)
        right_widget = QWidget(); right_widget.setLayout(right_panel)

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
        self.list_images.currentRowChanged.connect(self.on_select_image)
        self.list_picks.itemChanged.connect(self.on_pick_item_changed)

        # Picking: conectar al canvas (solo cuando captura ON y toolbar inactiva)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_click)

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
            self.list_groups.clear(); self.list_groups.addItems(self.group_names)
            self.lbl_cfg.setText(f"Config: {os.path.basename(path)}")
            # refrescar overlay si ya hay imagen
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
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.dropna(subset=["Fecha"]).copy()
            df = df.sort_values("Fecha").reset_index(drop=True)
            self.df_roi = df
            self.list_images.clear()
            for _, row in df.iterrows():
                fecha_str = row["Fecha"].strftime("%Y-%m-%d")
                ruta = str(row["Ruta"])  # puedes mostrar basename si prefieres
                self.list_images.addItem(f"{fecha_str} | {ruta}")
            self.lbl_roi.setText(f"ROI list: {os.path.basename(path)}")
        except Exception as e:
            self._error(f"Error al cargar ROI list: {e}{traceback.format_exc()}")

    def on_load_mcal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecciona Mcal_py.csv", "", "CSV (*.csv);;Todos (*.*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, comment="#")
            needed = {"Fecha", "i", "j", "Ng"}
            if not needed.issubset(df.columns):
                raise ValueError(f"Mcal debe contener columnas {needed}")
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.dropna(subset=["Fecha"]).copy()
            for c in ["i", "j", "Ng"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            self.df_mcal = df
            self.lbl_mcal.setText(f"Mcal: {os.path.basename(path)}")
            self._refresh_all_overlays()
        except Exception as e:
            self._error(f"Error al cargar Mcal: {e}")

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
            df_new = pd.DataFrame(self.session_picks)
            keep_cols = [c for c in df_new.columns if c in (['Fecha','i','j','Ng'] + [f'B{n:02d}' for n in range(1,13)] + ['B8A','B09','B11','B12'])]
            keep_cols = [c for c in ['Fecha','i','j','Ng'] if c in df_new.columns] + [c for c in keep_cols if c not in ('Fecha','i','j','Ng')]
            df_new = df_new[keep_cols].copy()
            if self.df_mcal is not None:
                df_all = pd.concat([self.df_mcal, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=['Fecha','i','j','Ng'], keep='last')
            else:
                df_all = df_new
            tmp = path + ".tmp"
            df_all.to_csv(tmp, index=False)
            os.replace(tmp, path)
            QMessageBox.information(self, "Guardar", f"Guardado: {path}")
        except Exception as e:
            self._error(f"Error al guardar: {e}")
        if not self.session_picks:
            QMessageBox.information(self, "Guardar", "No hay picks para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "Mcal_py_new.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            df_new = pd.DataFrame(self.session_picks)
            # filtra columnas visibles típicas
            keep_cols = [c for c in df_new.columns if c in (['Fecha','i','j','Ng'] + [f'B{n:02d}' for n in range(1,13)] + ['B8A','B09','B11','B12'])]
            keep_cols = [c for c in ['Fecha','i','j','Ng'] if c in df_new.columns] + [c for c in keep_cols if c not in ('Fecha','i','j','Ng')]
            df_new = df_new[keep_cols].copy()
            # merge opcional con Mcal existente
            if self.df_mcal is not None:
                df_all = pd.concat([self.df_mcal, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=['Fecha','i','j','Ng'], keep='last')
            else:
                df_all = df_new
            # escritura atómica
            tmp = path + ".tmp"
            df_all.to_csv(tmp, index=False)
            os.replace(tmp, path)
            QMessageBox.information(self, "Guardar", f"Guardado: {path}")
        except Exception as e:
            self._error(f"Error al guardar: {e}")
        self.overlay_toggle = checked
        self.btn_toggle_overlay.setText("Puntos Mcal: ON" if checked else "Puntos Mcal: OFF")
        self._refresh_all_overlays()

    def on_select_image(self, row: int):
        if row < 0 or self.df_roi is None or row >= len(self.df_roi):
            return
        rec = self.df_roi.iloc[row]
        tif_path = str(rec["Ruta"])
        if not os.path.isfile(tif_path):
            self._error(f"Archivo no existe: {tif_path}")
            return
        try:
            if self.current_ds is not None:
                self.current_ds = None
            ds = gdal_open_readonly(tif_path)
            rgb, sy, sx = read_rgb_thumbnail(ds, self.rgb_indices, max_dim=2048)
            self.current_ds = ds
            self.current_thumb = rgb
            self.scale_y, self.scale_x = sy, sx
            self.canvas.show_image(rgb)
            self._refresh_all_overlays()
        except Exception as e:
            self._error(f"Error al cargar imagen: {e}")

    # ------------------- Picking ------------------- #
    def on_pick_item_changed(self, item: QListWidgetItem):
        # Re-renderiza overlays según el check del ítem (visible/oculto).
        # Mantiene el orden LIFO tal como está en list_picks y session_picks.
        self._refresh_all_overlays()
    
    def on_canvas_click(self, event):
        # Bloquear captura si OFF o si toolbar activa (zoom/pan)
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
            QMessageBox.information(self, "Grupo", "Selecciona un grupo antes de hacer click.")
            return
        img_row = self.list_images.currentRow()
        if self.df_roi is None or img_row < 0:
            return
        fecha_sel = self.df_roi.iloc[img_row]["Fecha"]
        j_th = int(round(event.xdata)); i_th = int(round(event.ydata))
        H, W = self.current_thumb.shape[:2]
        if i_th < 0 or j_th < 0 or i_th >= H or j_th >= W:
            return
        j_full = int(j_th * self.scale_x); i_full = int(i_th * self.scale_y)
        spec = read_pixel_spectrum(self.current_ds, i_full, j_full)
        rec = {
            "Fecha": fecha_sel.strftime("%Y-%m-%d"),
            "i": i_full, "j": j_full,
            "i_th": i_th, "j_th": j_th,
            "Ng": int(grp_row + 1),
            "Grupo": self.group_names[grp_row],
            "visible": True
        }
        for bi, val in enumerate(spec, start=1):
            key = ("B%02d" % bi) if bi <= 12 else f"B{bi}"
            rec[key] = val
        # insertar al INICIO (último primero)
        self.session_picks.insert(0, rec)
        item = QListWidgetItem(f"{rec['Fecha']}  i={rec['i']} j={rec['j']}  Ng={rec['Ng']} ({rec['Grupo']})")
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.list_picks.insertItem(0, item)
        self._refresh_all_overlays()

    # ------------------- Overlays ------------------- #
    def _refresh_all_overlays(self):
        if self.current_thumb is None:
            return
        H, W = self.current_thumb.shape[:2]
        # 1) Mcal existente (capa 1)
        mcal_rgba = None
        if self.overlay_toggle:
            row = self.list_images.currentRow()
            if row >= 0 and self.df_roi is not None and self.df_mcal is not None and len(self.group_names) > 0:
                fecha_sel = self.df_roi.iloc[row]["Fecha"].date()
                df_day = self.df_mcal[self.df_mcal["Fecha"].dt.date == fecha_sel]
                if not df_day.empty:
                    mcal_rgba = np.zeros((H, W, 4), dtype=np.uint8)
                    js = (df_day["j"].astype(float) / self.scale_x).round().astype(int).clip(0, W-1).to_numpy()
                    is_ = (df_day["i"].astype(float) / self.scale_y).round().astype(int).clip(0, H-1).to_numpy()
                    ngs = df_day["Ng"].astype(int).to_numpy()
                    for j_th, i_th, ng in zip(js, is_, ngs):
                        r, g, b, a = _rgba_color(ng - 1, alpha=140)
                        mcal_rgba[i_th, j_th, :] = (r, g, b, a)
        self.canvas.update_overlay_mcal(mcal_rgba)

        # 2) Picks sesión (capa 2, rojo)
        picks_rgba = None
        if self.session_picks:
            picks_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            img_row = self.list_images.currentRow()
            if img_row >= 0 and self.df_roi is not None:
                fecha_sel = self.df_roi.iloc[img_row]["Fecha"].strftime("%Y-%m-%d")
                # recorrer lista en orden actual
                for idx in range(self.list_picks.count()):
                    item = self.list_picks.item(idx)
                    rec = self.session_picks[idx]
                    if rec.get("Fecha") != fecha_sel:
                        continue
                    vis = (item.checkState() == QtCore.Qt.Checked)
                    if not vis:
                        continue
                    i_th = int(rec.get("i_th", -1)); j_th = int(rec.get("j_th", -1))
                    if 0 <= i_th < H and 0 <= j_th < W:
                        # rojo sólido para picks
                        picks_rgba[i_th, j_th, :] = (255, 0, 0, 220)
        self.canvas.update_overlay_picks(picks_rgba)


    def _error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)


# ----------------------------- Utilidades de color ----------------------------- #

_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
]

def _rgba_color(i: int, alpha: int = 160) -> Tuple[int, int, int, int]:
    r, g, b = _PALETTE[i % len(_PALETTE)]
    return int(r), int(g), int(b), int(np.clip(alpha, 0, 255))


# ----------------------------- main ----------------------------- #

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = RoiViewer()
    w.resize(1300, 850)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
