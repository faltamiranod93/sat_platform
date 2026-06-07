# -*- coding: utf-8 -*-
"""
Roi Viewer + Picker (PyQt5 + Matplotlib + GDAL)
Con dock 'Gestión Mcal' + tabla, filtros, overlay RGBA por píxel y soft-delete.
centraliza el flujo “ver imagen ROI → seleccionar píxeles → asignar clase → mantener/limpiar dataset Mcal” con una UI simple y trazable por fecha y grupo.
ROI list debe tener {Fecha, Ruta} y Fecha se normaliza a día.
Mcal debe tener {Fecha, i, j, Ng}; el script completa bandas B01..B12 y H,S,L si faltan.
Guardado: escribe CSV final con columnas base en orden fijo; dedup por {Fecha,i,j,Ng}.
"""

import os
import sys
import json
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QWidget, QListWidget, QPushButton, QLabel, QHBoxLayout,
    QVBoxLayout, QSplitter, QSizePolicy, QListWidgetItem, QStatusBar,
    QDockWidget, QTableView, QComboBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

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
        self.mcal_scatter = None           # puntos Mcal activos
        self.mcal_sel_scatter = None       # puntos Mcal seleccionados (resaltados)
        self.picks_scatter = None          # picks de sesión

    def show_image(self, img: np.ndarray):
        self.ax.clear()
        self.base_img_artist = self.ax.imshow(img, zorder=0)
        self.ax.set_xlabel("X [px]")
        self.ax.set_ylabel("Y [px]")
        self.ax.set_aspect("equal", adjustable="box")

        # Scatter para puntos Mcal activos
        self.mcal_scatter = self.ax.scatter(
            [], [], s=20, alpha=0.7, zorder=10
        )

        # Scatter para puntos Mcal seleccionados (resaltados)
        self.mcal_sel_scatter = self.ax.scatter(
            [], [], s=80,
            facecolors='none', edgecolors='cyan',
            linewidths=1.5, zorder=11
        )

        # Scatter para picks de sesión
        self.picks_scatter = self.ax.scatter(
            [], [], s=30, alpha=0.9, zorder=12,
            color='red', marker='x', linewidth=2
        )

        self.draw_idle()

    def update_overlay_mcal(self, points: List[Tuple[float, float]], colors):
        """Actualiza puntos Mcal activos (scatter)."""
        if self.mcal_scatter is None:
            return
        if not points:
            self.mcal_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.mcal_scatter.set_offsets(points)
            if colors is not None:
                self.mcal_scatter.set_color(colors)
        self.draw_idle()

    def update_overlay_mcal_selection(self, points: List[Tuple[float, float]]):
        """Actualiza puntos Mcal seleccionados (resaltados)."""
        if self.mcal_sel_scatter is None:
            return
        if not points:
            self.mcal_sel_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.mcal_sel_scatter.set_offsets(points)
        self.draw_idle()

    def update_overlay_picks(self, points: List[Tuple[float, float]]):
        """Actualiza picks de sesión usando scatter plot"""
        if self.picks_scatter is None:
            return
        if not points:
            self.picks_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.picks_scatter.set_offsets(points)
        self.draw_idle()


# ----------------------------- Modelo de tabla para Mcal ----------------------------- #

class McalTableModel(QAbstractTableModel):
    COLS = ["Fecha", "Ng", "i", "j"]  # columnas visibles básicas

    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame(columns=self.COLS + ["_idx"])

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self.COLS)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        row = index.row()
        col = index.column()
        if row >= len(self._df):
            return None

        if role == Qt.DisplayRole:
            col_name = self.COLS[col]
            val = self._df.iloc[row][col_name]
            if col_name == "Fecha":
                if isinstance(val, (pd.Timestamp, np.datetime64)):
                    return pd.to_datetime(val).strftime("%Y-%m-%d")
                return str(val)
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.COLS[section]
        else:
            return str(section)


# ----------------------------- Ventana Principal ----------------------------- #

class RoiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Viewer/Picker (S2)")

        # Estado
        self.config: Dict = {}
        self.group_names: List[str] = []
        self.rgb_indices: Tuple[int, int, int] = (3, 2, 1)

        self.df_roi: Optional[pd.DataFrame] = None          # listado ROI (Fecha, Ruta)
        self.df_mcal: Optional[pd.DataFrame] = None
        self.df_mcal_all: Optional[pd.DataFrame] = None
        self.mcal_path: Optional[str] = None
        self.current_ds = None
        self.current_thumb: Optional[np.ndarray] = None
        self.scale_y: float = 1.0
        self.scale_x: float = 1.0
        self.overlay_toggle: bool = True
        self.capture_enabled: bool = True

        # picks de la sesión (independientes de Mcal)
        self.session_picks: List[Dict] = []

        # df “vista” actuales para la tabla
        self._df_mcal_active_view = pd.DataFrame()
        self._df_mcal_deleted_view = pd.DataFrame()

        # Widgets principales
        self.btn_load_cfg = QPushButton("Cargar Config")
        self.btn_load_roi = QPushButton("Cargar ROI list")
        self.btn_load_mcal = QPushButton("Cargar Mcal")

        self.btn_toggle_overlay = QPushButton("Puntos Mcal: ON")
        self.btn_toggle_overlay.setCheckable(True)
        self.btn_toggle_overlay.setChecked(True)

        self.btn_capture = QPushButton("Capturar: ON")
        self.btn_capture.setCheckable(True)
        self.btn_capture.setChecked(True)

        self.btn_save_csv = QPushButton("Guardar picks → CSV")
        self.btn_clear_picks = QPushButton("Limpiar Picks")

        self.lbl_cfg = QLabel("Config: —")
        self.lbl_roi = QLabel("ROI list: —")
        self.lbl_mcal = QLabel("Mcal: —")

        self.list_images = QListWidget()
        self.list_groups = QListWidget()
        self.list_groups.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_picks = QListWidget()

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Barra de estado
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Layout top bar
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

        # Central widget
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addLayout(top_bar)
        outer.addLayout(info_bar)
        outer.addWidget(splitter)
        self.setCentralWidget(central)

        # Dock Gestión Mcal
        self._build_mcal_dock()
        
        self.table_mcal_active.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_mcal_active.customContextMenuRequested.connect(self._open_mcal_active_context_menu)
        

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

        # Picking en canvas
        self.canvas.mpl_connect('button_release_event', self.on_canvas_click)

    # ------------------- Dock Gestión Mcal ------------------- #

    def _build_mcal_dock(self):
        self.dock_mcal = QDockWidget("Gestión Mcal", self)
        self.dock_mcal.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        w = QWidget()
        lay = QVBoxLayout(w)

        # Filtro por fecha
        lay.addWidget(QLabel("Filtro fecha Mcal:"))
        self.cmb_fecha_mode = QComboBox()
        self.cmb_fecha_mode.addItems(["Imagen actual", "Todas"])
        lay.addWidget(self.cmb_fecha_mode)

        # Filtro por Ng (checklist)
        lay.addWidget(QLabel("Filtro grupos (Ng):"))
        self.list_ng_filter = QListWidget()
        self.list_ng_filter.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        lay.addWidget(self.list_ng_filter)

        # Tabla Mcal activos
        lay.addWidget(QLabel("Puntos Mcal activos:"))
        self.table_mcal_active = QTableView()
        self.table_mcal_active.setSelectionBehavior(QTableView.SelectRows)
        self.table_mcal_active.setSelectionMode(QTableView.ExtendedSelection)
        self.table_mcal_active.setAlternatingRowColors(True)
        self.table_mcal_active.horizontalHeader().setStretchLastSection(True)
        self.mcal_model_active = McalTableModel()
        self.table_mcal_active.setModel(self.mcal_model_active)

        lay.addWidget(self.table_mcal_active)

        # Botones eliminar/restaurar
        btn_row = QHBoxLayout()
        self.btn_remove_mcal = QPushButton("Quitar puntos de Mcal")
        self.btn_restore_mcal = QPushButton("Restaurar puntos")
        btn_row.addWidget(self.btn_remove_mcal)
        btn_row.addWidget(self.btn_restore_mcal)
        lay.addLayout(btn_row)

        # Tabla Mcal eliminados
        lay.addWidget(QLabel("Puntos eliminados:"))
        self.table_mcal_deleted = QTableView()
        self.table_mcal_deleted.setSelectionBehavior(QTableView.SelectRows)
        self.table_mcal_deleted.setSelectionMode(QTableView.ExtendedSelection)
        self.table_mcal_deleted.setAlternatingRowColors(True)
        self.table_mcal_deleted.horizontalHeader().setStretchLastSection(True)
        self.mcal_model_deleted = McalTableModel()
        self.table_mcal_deleted.setModel(self.mcal_model_deleted)

        lay.addWidget(self.table_mcal_deleted)

        w.setLayout(lay)
        self.dock_mcal.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_mcal)

        # Conexiones
        self.cmb_fecha_mode.currentIndexChanged.connect(self._update_mcal_views)
        self.list_ng_filter.itemChanged.connect(self._update_mcal_views)
        self.table_mcal_active.selectionModel().selectionChanged.connect(self._update_selection_overlay)
        self.table_mcal_deleted.selectionModel().selectionChanged.connect(self._update_selection_overlay)
        self.btn_remove_mcal.clicked.connect(self._remove_selected_mcal_points)
        self.btn_restore_mcal.clicked.connect(self._restore_selected_mcal_points)

    # ------------------- Lifecycle ------------------- #

    def closeEvent(self, event):
        if self.current_ds is not None:
            self.current_ds = None
        event.accept()

    # ------------------- Actions ------------------- #
    
    def _open_mcal_active_context_menu(self, pos):
        if not self.group_names:
            return
        menu = QtWidgets.QMenu(self)
        sub = menu.addMenu("Cambiar grupo a…")
        for ng, name in enumerate(self.group_names):
            act = sub.addAction(f"{ng} - {name}")
            act.triggered.connect(lambda _=False, ng=ng: self._apply_ng_to_selected_active_rows(ng))
        menu.exec_(self.table_mcal_active.viewport().mapToGlobal(pos))


    def _apply_ng_to_selected_active_rows(self, new_ng: int):
        if self.df_mcal_all is None or self._df_mcal_active_view.empty:
            return
        if not (0 <= new_ng < len(self.group_names)):
            return

        sel = self.table_mcal_active.selectionModel().selectedRows()
        if not sel:
            return

        # mapear filas de vista -> índices reales en df_mcal_all usando _idx
        idxs = []
        for qidx in sel:
            r = qidx.row()
            if 0 <= r < len(self._df_mcal_active_view):
                idxs.append(int(self._df_mcal_active_view.iloc[r]["_idx"]))

        if not idxs:
            return

        self.df_mcal_all.loc[idxs, "Ng"] = int(new_ng)

        # refrescar tablas + overlays
        self._update_mcal_views()
        self._refresh_all_overlays()

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

            # Rellenar filtro Ng en dock
            self.list_ng_filter.clear()
            for i, gname in enumerate(self.group_names, start=0):
                item = QListWidgetItem(f"{i} - {gname}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.list_ng_filter.addItem(item)

            self.list_groups.clear()
            self.list_groups.addItems(self.group_names)
            self.lbl_cfg.setText(f"Config: {os.path.basename(path)}")

            self._update_mcal_views()
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

            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
            df = df.dropna(subset=["Fecha"]).copy()

            for c in ["i", "j", "Ng"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")

            bandas = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                      'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
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

            # Soft-delete flag
            if "deleted" not in df.columns:
                df["deleted"] = False

            self.df_mcal_all = df.reset_index(drop=True)
            self.df_mcal = self.df_mcal_all.drop(columns=['deleted']).copy()
            self.mcal_path = path
            self.lbl_mcal.setText(f"Mcal: {os.path.basename(path)}")

            self._update_mcal_views()
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
        """
        Guarda el Mcal completo (no sólo los picks nuevos), con esta lógica:

        Mcal_final = (Mcal_cargado SIN 'deleted'=True)  U  (picks de sesión)

        - Nunca se guarda la columna 'deleted' en el CSV.
        - Si no hay picks de sesión, igual se guarda (sólo con los cambios de borrado).
        - Si no hay Mcal cargado pero sí picks, se genera un Mcal nuevo desde cero.
        """
        if self.df_mcal_all is None and not self.session_picks:
            QMessageBox.information(
                self, "Guardar",
                "No hay datos Mcal ni picks en la sesión para guardar."
            )
            return

        # Nombre por defecto: si hay archivo cargado, usar ese; si no, Mcal_py.csv
        default_name = "Mcal_py.csv"
        if self.mcal_path is not None:
            default_name = self.mcal_path

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Mcal",
            default_name,
            "CSV (*.csv)"
        )
        if not path:
            return

        try:
            # -------------------------
            # 1) BASE: Mcal cargado (df_mcal_all) SIN 'deleted'
            # -------------------------
            base_cols = [
                'Fecha', 'i', 'j', 'Ng',
                'B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                'B07', 'B08', 'B8A', 'B09', 'B11', 'B12',
                'H', 'S', 'L'
            ]

            if self.df_mcal_all is not None and not self.df_mcal_all.empty:
                df_base = self.df_mcal_all.copy()

                # Filtrar borrados y quitar columna 'deleted'
                if 'deleted' in df_base.columns:
                    df_base = df_base[df_base['deleted'] != True].copy()
                    df_base = df_base.drop(columns=['deleted'])

                # Asegurar que todas las columnas base existan
                for c in base_cols:
                    if c not in df_base.columns:
                        df_base[c] = 0

                # Ordenar columnas
                df_base = df_base[base_cols]
                df_base['Fecha'] = pd.to_datetime(df_base['Fecha'], errors='coerce').dt.normalize()
            else:
                # No hay Mcal cargado: base vacía
                df_base = pd.DataFrame(columns=base_cols)

            # -------------------------
            # 2) NUEVOS PICKS DE SESIÓN
            # -------------------------
            if self.session_picks:
                rows = []
                for rec in self.session_picks:
                    row = {c: rec.get(c, 0) for c in base_cols}
                    rows.append(row)
                df_new = pd.DataFrame(rows)
                df_new['Fecha'] = pd.to_datetime(df_new['Fecha'], errors='coerce').dt.normalize()
            else:
                df_new = pd.DataFrame(columns=base_cols)

            # -------------------------
            # 3) UNIR BASE + NUEVOS Y LIMPIAR DUPLICADOS
            # -------------------------
            df_all = pd.concat([df_base, df_new], ignore_index=True)

            subset_keys = ['Fecha', 'i', 'j', 'Ng']
            # Sólo aplicar si todas estas columnas están presentes
            if all(c in df_all.columns for c in subset_keys):
                df_all = df_all.drop_duplicates(subset=subset_keys, keep='last')

            # Asegurar tipos (por si vienen como string)
            df_all['Fecha'] = pd.to_datetime(df_all['Fecha'], errors='coerce').dt.normalize()
            df_all['Ng'] = pd.to_numeric(df_all['Ng'], errors='coerce')
            df_all['i']  = pd.to_numeric(df_all['i'],  errors='coerce')
            df_all['j']  = pd.to_numeric(df_all['j'],  errors='coerce')

            # Ordenar
            df_all = df_all.sort_values(
                by=['Fecha', 'Ng', 'i', 'j'],
                ascending=[True, True, True, True],
                kind='mergesort'   # estable (útil si quieres reproducibilidad)
            ).reset_index(drop=True)

            # Asegurar orden columnas
            df_all = df_all[base_cols]

            # -------------------------
            # 4) GUARDAR A DISCO (SIN 'deleted')
            # -------------------------
            tmp = path + ".tmp"
            df_all.to_csv(tmp, index=False)
            os.replace(tmp, path)

            # -------------------------
            # 5) ACTUALIZAR ESTADO EN MEMORIA
            # -------------------------
            # Ahora el Mcal "oficial" es df_all (sin columna deleted)
            self.df_mcal = df_all.copy()

            # df_mcal_all vuelve a tener 'deleted' pero todo en False (nuevo estado limpio)
            self.df_mcal_all = df_all.copy()
            self.df_mcal_all['deleted'] = False

            self.mcal_path = path
            self.lbl_mcal.setText(f"Mcal: {os.path.basename(path)}")

            # Refrescar tablas y overlays
            self._update_mcal_views()
            self._refresh_all_overlays()

            QMessageBox.information(
                self,
                "Guardar",
                f"Mcal guardado en:\n{path}\n\n"
                f"Picks añadidos en esta sesión: {len(self.session_picks)}"
            )

        except Exception as e:
            self._error(f"Error al guardar: {e}\n{traceback.format_exc()}")



    def on_clear_picks(self):
        if self.session_picks:
            reply = QMessageBox.question(
                self, "Limpiar Picks",
                f"¿Eliminar todos los {len(self.session_picks)} picks de sesión?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.session_picks.clear()
                self.list_picks.clear()
                self.canvas.update_overlay_picks([])

    def on_select_image(self, row: int):
        if row < 0 or self.df_roi is None or row >= len(self.df_roi):
            return

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

            self._update_mcal_views()
            self._refresh_all_overlays()

        except Exception as e:
            self._error(f"Error al cargar imagen: {e}\n{traceback.format_exc()}")

    def on_pick_item_changed(self, item):
        # picks de sesión: dejar como está (solo overlay de picks)
        self._refresh_all_overlays()

    # ------------------- Picking en canvas ------------------- #

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

            if len(spec) >= 4:
                rgb_values = np.array([
                    spec[3] / 10000.0,  # B04
                    spec[2] / 10000.0,  # B03
                    spec[1] / 10000.0   # B02
                ])
                rgb_values = np.clip(rgb_values, 0.0, 1.0)
                hsl_result = f_rgb2hsl(rgb_values)
                H_val, S_val, L_val = hsl_result[0], hsl_result[1], hsl_result[2]
            else:
                H_val, S_val, L_val = 0.0, 0.0, 0.0

            rec = {
                "Fecha": fecha_sel.strftime("%Y-%m-%d"),
                "i": i_full, "j": j_full,
                "i_th": i_th, "j_th": j_th,
                "Ng": int(grp_row),  # 0-based (config_bandas_v2.json)
                "Grupo": self.group_names[grp_row],
                "visible": True,
                "H": float(H_val),
                "S": float(S_val),
                "L": float(L_val)
            }

            bandas_sentinel2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                                'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            for idx, banda in enumerate(bandas_sentinel2):
                if idx < len(spec):
                    rec[banda] = int(spec[idx])
                else:
                    rec[banda] = 0

            self.session_picks.insert(0, rec)
            item = QListWidgetItem(
                f"{rec['Fecha']}  i={rec['i']} j={rec['j']}  Ng={rec['Ng']} ({rec['Grupo']})"
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_picks.insertItem(0, item)

            # Overlay picks (en thumbnail coords)
            pick_points = []
            for p in self.session_picks:
                if p.get("Fecha") == fecha_sel.strftime("%Y-%m-%d"):
                    pick_points.append([p["j_th"], p["i_th"]])
            self.canvas.update_overlay_picks(pick_points)

            self.status.showMessage(
                f"Pick agregado: {rec['Grupo']} en ({i_full}, {j_full})", 3000
            )

        except Exception as e:
            self._error(f"Error al leer pixel: {e}")

    # ------------------- Gestión Mcal (dock) ------------------- #

    def _selected_ng_list(self) -> List[int]:
        """Ng seleccionados en el checklist. Si ninguno marcado, devuelve todos."""
        ngs = []
        for row in range(self.list_ng_filter.count()):
            item = self.list_ng_filter.item(row)
            if item.checkState() == Qt.Checked:
                # item text: "1 - Grupo"
                ng = row  # 0-based (Ng == row)
                ngs.append(ng)
        if not ngs and self.group_names:
            ngs = list(range(0, len(self.group_names)))
        return ngs

    def _build_active_view_df(self) -> pd.DataFrame:
        """DataFrame Mcal activo filtrado por fecha y Ng + índice global _idx."""
        if self.df_mcal_all is None or self.df_mcal_all.empty:
            return pd.DataFrame(columns=["Fecha","Ng","i","j","_idx"])

        df = self.df_mcal_all.copy()
        df = df[~df["deleted"]]

        # Filtro fecha
        mode = self.cmb_fecha_mode.currentText()
        if mode == "Imagen actual" and self.df_roi is not None:
            row = self.list_images.currentRow()
            if row >= 0:
                fecha_im = self.df_roi.iloc[row]["Fecha"].date()
                df = df[df["Fecha"].dt.date == fecha_im]

        # Filtro Ng
        ngs = self._selected_ng_list()
        df = df[df["Ng"].isin(ngs)]

        if df.empty:
            return pd.DataFrame(columns=["Fecha","Ng","i","j","_idx"])

        df_tmp = df.reset_index().rename(columns={"index": "_idx"})
        return df_tmp[["Fecha","Ng","i","j","_idx"]]

    def _build_deleted_view_df(self) -> pd.DataFrame:
        if self.df_mcal_all is None or self.df_mcal_all.empty:
            return pd.DataFrame(columns=["Fecha","Ng","i","j","_idx"])
        df = self.df_mcal_all[self.df_mcal_all["deleted"]].copy()
        if df.empty:
            return pd.DataFrame(columns=["Fecha","Ng","i","j","_idx"])
        df_tmp = df.reset_index().rename(columns={"index": "_idx"})
        return df_tmp[["Fecha","Ng","i","j","_idx"]]

    def _update_mcal_views(self):
        """Actualiza tabla activos/eliminados según filtros y refresca overlays."""
        active = self._build_active_view_df()
        deleted = self._build_deleted_view_df()

        self._df_mcal_active_view = active
        self._df_mcal_deleted_view = deleted

        self.mcal_model_active.set_dataframe(active)
        self.mcal_model_deleted.set_dataframe(deleted)

        self._refresh_all_overlays()

    def _remove_selected_mcal_points(self):
        if self.df_mcal_all is None or self._df_mcal_active_view.empty:
            return

        sel = self.table_mcal_active.selectionModel().selectedRows()
        if not sel:
            return

        idx_global = []
        for idx in sel:
            row = idx.row()
            if 0 <= row < len(self._df_mcal_active_view):
                gidx = int(self._df_mcal_active_view.iloc[row]["_idx"])
                idx_global.append(gidx)

        if not idx_global:
            return

        self.df_mcal_all.loc[idx_global, "deleted"] = True
        self._update_mcal_views()

    def _restore_selected_mcal_points(self):
        if self.df_mcal_all is None or self._df_mcal_deleted_view.empty:
            return

        sel = self.table_mcal_deleted.selectionModel().selectedRows()
        if not sel:
            return

        idx_global = []
        for idx in sel:
            row = idx.row()
            if 0 <= row < len(self._df_mcal_deleted_view):
                gidx = int(self._df_mcal_deleted_view.iloc[row]["_idx"])
                idx_global.append(gidx)

        if not idx_global:
            return

        self.df_mcal_all.loc[idx_global, "deleted"] = False
        self._update_mcal_views()

    def _update_selection_overlay(self, *_args):
        """Redibuja capa de selección en función de lo seleccionado en tabla activos."""
        if self.current_thumb is None:
            self.canvas.update_overlay_mcal_selection([])
            return

        H, W = self.current_thumb.shape[:2]
        sel_points = []

        if self._df_mcal_active_view.empty:
            self.canvas.update_overlay_mcal_selection([])
            return

        sel = self.table_mcal_active.selectionModel().selectedRows()
        if not sel:
            self.canvas.update_overlay_mcal_selection([])
            return

        for idx in sel:
            row = idx.row()
            if 0 <= row < len(self._df_mcal_active_view):
                i = int(self._df_mcal_active_view.iloc[row]["i"])
                j = int(self._df_mcal_active_view.iloc[row]["j"])

                i_th = int(round(i / self.scale_y))
                j_th = int(round(j / self.scale_x))

                if 0 <= i_th < H and 0 <= j_th < W:
                    sel_points.append([j_th, i_th])

        # estos puntos se dibujan como círculos grandes cyan (definido en MplCanvas)
        self.canvas.update_overlay_mcal_selection(sel_points)


    # ------------------- Overlays ------------------- #

    def _refresh_all_overlays(self):
        """Redibuja overlay de Mcal + selección + picks (modo scatter)."""
        if self.current_thumb is None:
            return

        H, W = self.current_thumb.shape[:2]

        # ---------- 1) PUNTOS MCAL ACTIVOS (SCATTER) ----------
        mcal_points = []
        mcal_colors = []

        if self.overlay_toggle and self.df_mcal_all is not None and len(self.group_names) > 0:
            df = self._df_mcal_active_view
            if not df.empty:
                n_groups = len(self.group_names)
                for _, row in df.iterrows():
                    i = int(row["i"])
                    j = int(row["j"])
                    ng = int(row["Ng"])

                    # pasar de coordenadas full a thumbnail
                    i_th = int(round(i / self.scale_y))
                    j_th = int(round(j / self.scale_x))

                    if 0 <= i_th < H and 0 <= j_th < W:
                        # color según grupo
                        if 0 <= ng < n_groups:
                            r, g, b, a = _rgba_color(ng, n_groups, alpha=200)
                            color = (r, g, b, a)
                        else:
                            # Ng fuera de rango -> naranja
                            color = (1.0, 0.65, 0.0, 0.8)

                        mcal_points.append([j_th, i_th])
                        mcal_colors.append(color)

        # Actualizar scatter de Mcal
        self.canvas.update_overlay_mcal(mcal_points, mcal_colors)

        # ---------- 2) SELECCIÓN MCAL (SCATTER) ----------
        self._update_selection_overlay()

        # ---------- 3) PICKS DE SESIÓN (SCATTER) ----------
        pick_points = []
        if self.session_picks and self.df_roi is not None:
            img_row = self.list_images.currentRow()
            if img_row >= 0:
                fecha_sel = self.df_roi.iloc[img_row]["Fecha"].strftime("%Y-%m-%d")
                for p in self.session_picks:
                    if p.get("Fecha") == fecha_sel:
                        pick_points.append([p["j_th"], p["i_th"]])

        self.canvas.update_overlay_picks(pick_points)
        
    def _error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        try:
            self.status.showMessage(f"ERROR: {msg}", 5000)
        except:
            pass




# ----------------------------- Utilidades de color ----------------------------- #

def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    if num_colors <= 0:
        return [(31, 119, 180)]
    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)
        saturation = 0.8
        value = 0.9

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

        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _rgba_color(i: int, num_groups: int, alpha: int = 160) -> Tuple[float, float, float, float]:
    if not hasattr(_rgba_color, 'palette_cache'):
        _rgba_color.palette_cache = {}

    if num_groups not in _rgba_color.palette_cache:
        _rgba_color.palette_cache[num_groups] = generate_color_palette(num_groups)

    palette = _rgba_color.palette_cache[num_groups]
    r, g, b = palette[i % len(palette)]
    return r / 255.0, g / 255.0, b / 255.0, alpha / 255.0


# ----------------------------- main ----------------------------- #

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = RoiViewer()
    w.resize(1500, 900)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

