# -*- coding: utf-8 -*-
"""
q_Sat_SpectralSignature.py

Re-implementación PyQt5 + Matplotlib del visor de ROI/firmas espectrales.
"""
import os
import sys
import json
import math
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from osgeo import gdal

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ----------------------------- Hardcoded paths -----------------------------
path = r"C:/Users/felip/Desktop/Msc-UTFSM/Laguna-Seca/"
ruta_funciones = r"C:/Users/felip/Desktop/Msc-UTFSM/00_Codigos/Python"
ver = "v6"

# ----------------------------- Utilidades SD -------------------------------
if not os.path.exists(ruta_funciones):
    raise FileNotFoundError(f"La carpeta de funciones no existe: {ruta_funciones}")

sys.path.append(ruta_funciones)
_funciones_importadas = []
for _fn in os.listdir(ruta_funciones):
    if _fn.startswith("f_") and _fn.endswith(".py"):
        try:
            mod = __import__(_fn[:-3])
            for _attr in dir(mod):
                if not _attr.startswith("__") and callable(getattr(mod, _attr)):
                    globals()[_attr] = getattr(mod, _attr)
                    _funciones_importadas.append(_attr)
        except Exception as _e:
            print(f"No se pudo importar {_fn}: {_e}")

if 'read_csv_file' not in globals():
    def read_csv_file(csv_path: str, comment: str = '#') -> pd.DataFrame:
        return pd.read_csv(csv_path, comment=comment)

# ----------------------------- Config bandas -------------------------------
CONFIG_BANDAS = os.path.join(ruta_funciones, "config_bandas_v3.json")
if not os.path.exists(CONFIG_BANDAS):
    raise FileNotFoundError(f"Falta config_bandas.json en {ruta_funciones}")

with open(CONFIG_BANDAS, 'r', encoding='utf-8') as f:
    _cfg_raw = json.load(f)

def _rgb_to_hex(rgb):
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def _bands_from_legacy(cfg_old):
    band_names = [b['name'] for b in cfg_old['bands']]
    lambdas   = np.array([b['lambda'] for b in cfg_old['bands']], dtype=float)
    band_index = {b: i+1 for i, b in enumerate(band_names)}
    classmap_colors = cfg_old.get('groups', {}).get('classmap', {})
    mcal_group_colors = cfg_old.get('groups', {}).get('mcal', {})
    presets = cfg_old.get('presets', {"TrueColor":[4,3,2], "FalseColor":[8,4,3], "SWIR":[12,8,4]})
    return band_names, lambdas, band_index, classmap_colors, mcal_group_colors, presets

def _bands_from_current(cfg_new):
    required = ("Nband", "lam", "Nband_sort")
    if not all(k in cfg_new for k in required):
        raise KeyError("Faltan llaves requeridas: Nband, lam, Nband_sort")
    lam_map = {bn: float(lv) for bn, lv in zip(cfg_new["Nband"], cfg_new["lam"])}
    band_names = list(cfg_new["Nband_sort"])
    lambdas = np.array([lam_map[b] for b in band_names], dtype=float)
    band_index = {b: i+1 for i, b in enumerate(band_names)}
    # colores MCAL desde nameg/color
    nameg = list(cfg_new.get("nameg", []))
    color = list(cfg_new.get("color", []))
    if len(nameg) != len(color):
        raise ValueError("Longitudes distintas entre nameg y color")
    mcal_group_colors = {str(g): _rgb_to_hex(rgb) for g, rgb in zip(nameg, color)}
    classmap_colors = {}
    def _preset(bnames): return [band_index[b] for b in bnames]
    presets = {
        "TrueColor": _preset(["B04","B03","B02"]),
        "FalseColor": _preset(["B08","B04","B03"]),
        "SWIR": _preset(["B12","B11","B04"]),
    }
    return band_names, lambdas, band_index, classmap_colors, mcal_group_colors, presets

if "bands" in _cfg_raw:
    BAND_NAMES, LAMBDA_NM, BAND_INDEX, CLASSMAP_COLORS, MCAL_GROUP_COLORS, PRESETS = _bands_from_legacy(_cfg_raw)
else:
    BAND_NAMES, LAMBDA_NM, BAND_INDEX, CLASSMAP_COLORS, MCAL_GROUP_COLORS, PRESETS = _bands_from_current(_cfg_raw)

CURRENT_PRESET = "TrueColor"

# ----------------------------- Paleta de colores por grupo (Ng) -----------------------------

def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Misma lógica que en el visor ROI:
    genera colores equiespaciados en HSV y los entrega como RGB (0-255).
    """
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


def _rgb_to_hex(rgb):
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def get_group_color_hex(ng_value) -> str:
    """
    Dado un Ng (1, 2, 3, ...), devuelve el color hex de la paleta HSV,
    consistente con el visor ROI.
    """
    try:
        ng_int = int(ng_value)
    except Exception:
        return "#FFEE00"  # fallback

    if NUM_GROUPS <= 0:
        return "#FFEE00"

    idx = (ng_int - 1) % NUM_GROUPS
    return _rgb_to_hex(_PALETTE_GROUPS[idx])

def get_group_name(ng_value) -> str:
    """
    Dado un Ng (1, 2, 3, ...), devuelve el nombre del grupo desde GROUP_NAMES.
    """
    try:
        ng_int = int(ng_value)
        if 1 <= ng_int <= len(GROUP_NAMES):
            return GROUP_NAMES[ng_int - 1]
        else:
            return f"Grupo {ng_int}"
    except (ValueError, TypeError):
        return f"Grupo {ng_value}"

# Número de grupos (Ng) según config_bandas (nameg)
GROUP_NAMES = _cfg_raw.get("nameg", [])
NUM_GROUPS = len(GROUP_NAMES) if GROUP_NAMES else 0

# Paleta global por índice de grupo (Ng = 1..NUM_GROUPS)
_PALETTE_GROUPS = generate_color_palette(NUM_GROUPS) if NUM_GROUPS > 0 else []

# ----------------------------- Datos de entrada ----------------------------
CSV_ROI = os.path.join(path, "02-Space-Facilities", "00-ROI-utm.csv")
CSV_LIST = os.path.join(path, "02-Space-Facilities", "04-ROI-MOD.csv")
CSV_MCAL = os.path.join(path, "Mcal_py.csv")
CSV_MCAL_HSL = os.path.join(path, f"McalHSL_mod_{ver}_py.csv")

for p in [CSV_LIST, CONFIG_BANDAS]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe requerido: {p}")

# ------------------------------ Modelos UI ---------------------------------
MCAL_SOURCES = {
"Mcal_py.csv": CSV_MCAL,
"McalHSL_mod": CSV_MCAL_HSL,
}
CURRENT_MCAL_KEY = "Mcal_py.csv"

@dataclass
class PixelSel:
    pid: str
    i: int
    j: int

@dataclass
class ProjectState:
    active_date: Optional[str]
    xaxis_mode: str
    show_mcal: bool
    mcal_filter_by_date: bool
    preset: str
    selected_pixels: List[PixelSel]
    loaded_paths: Dict[str, str]

# ------------------------------ Helpers GDAL -------------------------------
def open_dataset(path_tif: str) -> gdal.Dataset:
    ds = gdal.Open(path_tif, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL no pudo abrir: {path_tif}")
    return ds

def read_band_sample(ds: gdal.Dataset, band_idx: int, i: int, j: int) -> float:
    band = ds.GetRasterBand(band_idx)
    arr = band.ReadAsArray(j, i, 1, 1)
    val = float(arr[0, 0]) if arr is not None else np.nan
    return val

def build_rgb(ds: gdal.Dataset, preset_name: str = 'TrueColor') -> np.ndarray:
    idxs = PRESETS.get(preset_name, PRESETS['TrueColor'])
    H, W = ds.RasterYSize, ds.RasterXSize
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for k, band_idx in enumerate(idxs):
        band = ds.GetRasterBand(band_idx)
        arr = band.ReadAsArray().astype(np.float32)
        m = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
        m = m if m > 0 else 1.0
        rgb[:, :, k] = np.clip(arr / m, 0, 1)
    return rgb

# ---------------------------- Lector de CSVs -------------------------------
def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_roi = read_csv_file(CSV_ROI, '#') if os.path.exists(CSV_ROI) else pd.DataFrame()
    df_list = read_csv_file(CSV_LIST, '#')
    df_mcal = read_csv_file(CSV_MCAL, '#') if os.path.exists(CSV_MCAL) else pd.DataFrame()
    df_mcal_hsl = read_csv_file(CSV_MCAL_HSL, '#') if os.path.exists(CSV_MCAL_HSL) else pd.DataFrame()
    return df_roi, df_list, df_mcal, df_mcal_hsl

# ------------------------------- Canvas Img --------------------------------
class ImageCanvas(FigureCanvas):
    pixelClicked = QtCore.pyqtSignal(int, int)  # i, j

    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        self._img_artist = None
        self._mcal_scatter = None
        self._hover_annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", ec="0.7"),
            arrowprops=dict(arrowstyle="->")
        )
        self._hover_annot.set_visible(False)
        self._mcal_df = pd.DataFrame()
        self._mcal_labels = []  # una etiqueta por punto
        self._cid = self.mpl_connect('button_press_event', self._on_click)
        self._hid = self.mpl_connect('motion_notify_event', self._on_hover)
        self.img_shape = None  # (H, W)

    def show_image(self, rgb: np.ndarray):
        self.ax.clear(); self.ax.set_axis_off()
        self._img_artist = self.ax.imshow(rgb, interpolation='nearest')
        self.img_shape = rgb.shape[:2]
        self.draw_idle()

    def show_classmap(self, cm: np.ndarray, palette: Dict[str, str]):
        self.ax.clear(); self.ax.set_axis_off()
        maxc = int(np.nanmax(cm)) if np.size(cm) > 0 else 0
        lut = np.zeros((maxc + 1, 3), dtype=float)
        for k in range(maxc + 1):
            hexc = palette.get(str(k), '#000000')
            c = QtGui.QColor(hexc)
            lut[k] = [c.redF(), c.greenF(), c.blueF()]
        rgb = lut[np.clip(cm, 0, maxc).astype(int)]
        self._img_artist = self.ax.imshow(rgb, interpolation='nearest')
        self.img_shape = cm.shape
        self.draw_idle()

    def set_mcal_points(self, df: pd.DataFrame, labels: List[str]):
        # espera columnas i, j y opcionalmente Ng
        self._mcal_df = df.copy()
        self._mcal_labels = list(labels)
        if self._mcal_scatter is not None:
            self._mcal_scatter.remove()
            self._mcal_scatter = None
        if df.empty:
            self.draw_idle(); return
        yy = df['i'].astype(int).to_numpy()
        xx = df['j'].astype(int).to_numpy()

        if 'Ng' in df.columns:
            colors = [get_group_color_hex(ng) for ng in df['Ng']]
        else:
            colors = '#FFFF00'  # fallback único

        self._mcal_scatter = self.ax.scatter(
            xx, yy,
            s=20,
            c=colors,
            edgecolors='k',
            linewidths=0.3
        )
        self.draw_idle()

    def clear_mcal(self):
        if self._mcal_scatter is not None:
            self._mcal_scatter.remove()
            self._mcal_scatter = None
            self.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.img_shape is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata)); i = int(round(event.ydata))
        H, W = self.img_shape
        if 0 <= i < H and 0 <= j < W:
            self.pixelClicked.emit(i, j)  # i=fila, j=col

    def _on_hover(self, event):
        # tooltip de MCAL (Ng-nameg)
        if self._mcal_scatter is None or event.inaxes != self.ax:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False); self.draw_idle()
            return
        contains, ind = self._mcal_scatter.contains(event)
        if contains and 'ind' in ind and len(ind['ind']) > 0:
            idx = ind['ind'][0]
            if 0 <= idx < len(self._mcal_labels):
                pos = self._mcal_scatter.get_offsets()[idx]
                self._hover_annot.xy = (pos[0], pos[1])
                self._hover_annot.set_text(self._mcal_labels[idx])
                self._hover_annot.set_visible(True)
                self.draw_idle()
        else:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False); self.draw_idle()

# ------------------------------- Canvas Firmas ------------------------------

class SignatureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.grid(True, alpha=0.25)
        self.x_mode = 'lambda'

    def set_xmode(self, mode: str):
        self.x_mode = mode
        self.draw_idle()

    def clear(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.draw_idle()

    def plot_signature(
        self,
        xvals: np.ndarray,
        yvals: np.ndarray,
        label: str,
        std_upper: np.ndarray = None,
        std_lower: np.ndarray = None,
        color: Optional[str] = None,
        marker: str = 'o',
        linestyle: str = '-'
    ):
        """Plotear firma espectral con bandas de desviación estándar."""
        # Plotear línea promedio
        if color is not None:
            line = self.ax.plot(
                xvals, yvals,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=label,
                color=color
            )
            line_color = color
        else:
            line = self.ax.plot(
                xvals, yvals,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=label
            )
            line_color = line[0].get_color()

        # Banda de desviación estándar
        if std_upper is not None and std_lower is not None:
            self.ax.fill_between(
                xvals, std_lower, std_upper,
                alpha=0.25,
                color=line_color
            )

        self.ax.legend(loc='best', fontsize=8)
        if self.x_mode == 'lambda':
            self.ax.set_xlabel('Wavelength λ (nm)')
        else:
            self.ax.set_xlabel('Band')
            self.ax.set_xticks(np.arange(len(BAND_NAMES)))
            self.ax.set_xticklabels(BAND_NAMES, rotation=45, ha='right')
        self.ax.set_ylabel('Reflectance (scaled 0–10000)')
        self.ax.set_ylim([0, 10000])
        self.draw_idle()

# ------------------------------- Ventana firmas ----------------------------
class SignatureWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Firmas espectrales")
        self.resize(900, 500)

        self.canvas = SignatureCanvas(self)
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "i", "j", "Fecha"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.SelectedClicked)

        self.chk_lambda = QtWidgets.QCheckBox("Eje X en λ (nm)")
        self.chk_lambda.setChecked(True)
        self.chk_lambda.stateChanged.connect(self._toggle_xmode)

        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.canvas, 2)
        hl.addWidget(self.table, 1)
        layout.addLayout(hl)
        layout.addWidget(self.chk_lambda)

    def _toggle_xmode(self, st):
        self.canvas.set_xmode('lambda' if st==Qt.Checked else 'band')

    def update_table(self, rows: List[Tuple[str,int,int,str]]):
        self.table.setRowCount(len(rows))
        for r, (pid, i, j, fecha) in enumerate(rows):
            for c, val in enumerate([pid, i, j, fecha]):
                item = QtWidgets.QTableWidgetItem(str(val))
                if c>0: item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(r, c, item)
                
# ------------------------------- Ventana firmas Promedio ----------------------------

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']

class AverageSignatureWindow(QtWidgets.QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Firmas Especiales Promedio")
        self.resize(1000, 600)
        
        self.canvas = SignatureCanvas(self)
        self.setup_controls()
        self.setup_layout()
        
    def setup_controls(self):
        # Selector de grupos
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        
        # Selector de fechas
        self.date_list = QtWidgets.QListWidget()
        self.date_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        
        # Botones
        self.btn_plot = QtWidgets.QPushButton("Plotear Firmas Promedio")
        self.btn_plot.clicked.connect(self.plot_average_signatures)
        
        self.btn_select_all_groups = QtWidgets.QPushButton("Seleccionar Todos los Grupos")
        self.btn_select_all_groups.clicked.connect(self.select_all_groups)
        
        self.btn_clear_groups = QtWidgets.QPushButton("Limpiar Selección Grupos")
        self.btn_clear_groups.clicked.connect(self.clear_groups)
        
        self.btn_select_all_dates = QtWidgets.QPushButton("Seleccionar Todas las Fechas")
        self.btn_select_all_dates.clicked.connect(self.select_all_dates)
        
        self.btn_clear_dates = QtWidgets.QPushButton("Limpiar Selección Fechas")
        self.btn_clear_dates.clicked.connect(self.clear_dates)
        
        # Estadísticas
        self.stats_label = QtWidgets.QLabel("Selecciona grupos y fechas para ver estadísticas")
        
        # Checkbox para eje X
        self.chk_lambda = QtWidgets.QCheckBox("Eje X en λ (nm)")
        self.chk_lambda.setChecked(True)
        self.chk_lambda.stateChanged.connect(self._toggle_xmode)
        
    def setup_layout(self):
        layout = QtWidgets.QHBoxLayout(self)
        
        # Panel izquierdo - controles
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Grupos
        left_layout.addWidget(QtWidgets.QLabel("Grupos (selección múltiple):"))
        left_layout.addWidget(self.group_list)
        btn_group_layout = QtWidgets.QHBoxLayout()
        btn_group_layout.addWidget(self.btn_select_all_groups)
        btn_group_layout.addWidget(self.btn_clear_groups)
        left_layout.addLayout(btn_group_layout)
        
        # Fechas
        left_layout.addWidget(QtWidgets.QLabel("Fechas (selección múltiple):"))
        left_layout.addWidget(self.date_list)
        btn_date_layout = QtWidgets.QHBoxLayout()
        btn_date_layout.addWidget(self.btn_select_all_dates)
        btn_date_layout.addWidget(self.btn_clear_dates)
        left_layout.addLayout(btn_date_layout)
        
        left_layout.addWidget(self.stats_label)
        left_layout.addWidget(self.btn_plot)
        left_layout.addWidget(self.chk_lambda)
        left_layout.addStretch()
        
        # Panel derecho - canvas
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(self.canvas)
        
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)
        
    def _toggle_xmode(self, st):
        self.canvas.set_xmode('lambda' if st == Qt.Checked else 'band')
        
    def load_groups_and_dates(self):
        """Cargar grupos y fechas disponibles desde los datos MCAL COMPLETOS"""
        # Usar datos MCAL completos sin filtrar
        df_mcal = self.main_window._current_mcal_complete()
        
        print(f"DEBUG: Total de registros MCAL: {len(df_mcal)}")
        print(f"DEBUG: Columnas disponibles: {df_mcal.columns.tolist()}")
        
        # Cargar grupos únicos - CORREGIDO
        if 'Ng' in df_mcal.columns:
            print(f"DEBUG: Valores únicos en 'Ng': {df_mcal['Ng'].unique()}")
            
            # Obtener grupos únicos directamente de la columna Ng
            unique_ng = sorted(df_mcal['Ng'].dropna().unique())
            print(f"DEBUG: Grupos Ng únicos: {unique_ng}")
            self.group_list.clear()
            for ng_val in unique_ng:
                    gname = get_group_name(ng_val)
                    text = f"Ng{int(ng_val)} - {gname}"
                    item = QtWidgets.QListWidgetItem(text)
                    item.setData(Qt.UserRole, int(ng_val))  # guardamos Ng numérico
                    self.group_list.addItem(item)
            print(f"DEBUG: Total de grupos cargados: {self.group_list.count()}")
        else:
            print("DEBUG: No se encontró columna 'Ng'")
            # Intentar encontrar columnas alternativas
            group_cols = [col for col in df_mcal.columns if 'group' in col.lower() or 'ng' in col.lower()]
            print(f"DEBUG: Columnas que podrían ser grupos: {group_cols}")
            
            if group_cols:
                group_items = self.load_groups_alternative(df_mcal, group_cols[0])
                self.group_list.clear()
                self.group_list.addItems(group_items)
        
        # Cargar fechas únicas (de TODOS los datos MCAL)
        if 'Fecha' in df_mcal.columns:
            dates = sorted(df_mcal['Fecha'].astype(str).unique())
            self.date_list.clear()
            self.date_list.addItems(dates)
            print(f"DEBUG: Fechas cargadas: {len(dates)}")
        else:
            print("DEBUG: No se encontró columna 'Fecha'")
            date_cols = [col for col in df_mcal.columns if 'fecha' in col.lower() or 'date' in col.lower()]
            print(f"DEBUG: Columnas que podrían ser fechas: {date_cols}")
            
        # Seleccionar automáticamente algunas opciones por defecto
        self.select_default_items()

    def load_groups_alternative(self, df_mcal, group_col):
        """Método alternativo para cargar grupos"""
        print(f"DEBUG: Cargando grupos desde columna: {group_col}")
        unique_vals = df_mcal[group_col].dropna().unique()
        print(f"DEBUG: Valores únicos en {group_col}: {unique_vals}")
        
        group_items = []
        for val in unique_vals:
            group_str = f"{val}"
            group_items.append(group_str)
        
        return group_items
        
    def select_default_items(self):
        """Seleccionar algunos ítems por defecto para facilitar el uso"""
        # Seleccionar primeros 3 grupos
        for i in range(min(3, self.group_list.count())):
            self.group_list.item(i).setSelected(True)
        
        # Seleccionar todas las fechas
        self.select_all_dates()
            
    def select_all_groups(self):
        for i in range(self.group_list.count()):
            self.group_list.item(i).setSelected(True)
            
    def clear_groups(self):
        for i in range(self.group_list.count()):
            self.group_list.item(i).setSelected(False)
            
    def select_all_dates(self):
        for i in range(self.date_list.count()):
            self.date_list.item(i).setSelected(True)
            
    def clear_dates(self):
        for i in range(self.date_list.count()):
            self.date_list.item(i).setSelected(False)
            
    def plot_average_signatures(self):
        """Plotear firmas espectrales promedio con desviación estándar."""
        # Ítems de grupo seleccionados (cada uno tiene Ng en UserRole)
        selected_items = self.group_list.selectedItems()
        selected_dates = [item.text() for item in self.date_list.selectedItems()]

        # Debug
        print("DEBUG: Grupos seleccionados:",
              [i.text() for i in selected_items])
        print("DEBUG: Fechas seleccionadas:", selected_dates)

        if not selected_items or not selected_dates:
            QtWidgets.QMessageBox.warning(
                self,
                "Advertencia",
                "Selecciona al menos un grupo y una fecha"
            )
            return

        # Usar datos MCAL COMPLETOS para el cálculo de promedios
        df_mcal = self.main_window._current_mcal_complete()
        if df_mcal.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "Advertencia",
                "No hay datos MCAL disponibles"
            )
            return

        # Limpiar canvas
        self.canvas.clear()
        plots_created = 0

        # Para cada grupo seleccionado
        for item in selected_items:
            # Ng numérico desde UserRole (si falla, intentar parsear desde el texto)
            ng_val = item.data(Qt.UserRole)
            if ng_val is None:
                text = item.text()  # p.ej. "Ng1 - Agua clara"
                try:
                    if text.startswith("Ng"):
                        ng_val = int(text[2:].split()[0])
                    else:
                        ng_val = int(text)
                except Exception as e:
                    print(f"DEBUG: No se pudo obtener Ng de '{text}': {e}")
                    continue

            gname = get_group_name(ng_val)
            color_hex = get_group_color_hex(ng_val)
            group_label_text = item.text()  # "Ng1 - Agua clara", etc.

            print(f"DEBUG: Procesando grupo Ng={ng_val} ({gname})")

            # Para cada fecha seleccionada, cambiar sólo el marcador
            for date_idx, date_str in enumerate(selected_dates):
                marker = MARKERS[date_idx % len(MARKERS)]
                print(f"DEBUG:   Fecha {date_str} con marker '{marker}'")

                # Filtrar datos por grupo (Ng) y fecha
                try:
                    mask = (
                        df_mcal['Ng'].astype(str) == str(ng_val)
                    ) & (
                        df_mcal['Fecha'].astype(str) == str(date_str)
                    )
                    group_data = df_mcal[mask]

                    print(
                        f"DEBUG:   Encontrados {len(group_data)} registros para "
                        f"Ng={ng_val}, Fecha={date_str}"
                    )

                    if len(group_data) > 0:
                        sample_ng = group_data['Ng'].iloc[0]
                        sample_fecha = group_data['Fecha'].iloc[0]
                        print(
                            f"DEBUG:   Muestra - Ng: {sample_ng} "
                            f"(tipo: {type(sample_ng)}), "
                            f"Fecha: {sample_fecha} (tipo: {type(sample_fecha)})"
                        )

                except Exception as e:
                    print(f"DEBUG:   Error filtrando datos: {e}")
                    continue

                if group_data.empty:
                    print(
                        f"DEBUG:   No hay datos para grupo Ng={ng_val} "
                        f"({gname}) en fecha {date_str}"
                    )
                    continue

                # ---------- Extraer valores de bandas ----------
                band_data_cols = []
                valid_samples = 0

                for band_name in BAND_NAMES:
                    if band_name in group_data.columns:
                        band_values = group_data[band_name].astype(float).to_numpy()
                        valid_count = np.sum(
                            ~np.isnan(band_values) & np.isfinite(band_values)
                        )
                        if valid_count > 0:
                            valid_samples = max(valid_samples, valid_count)
                        else:
                            # todo NaN -> se mantiene pero luego no aportará al promedio
                            pass
                    else:
                        print(
                            f"DEBUG:   Banda {band_name} no encontrada en datos, "
                            "rellenando con NaN"
                        )
                        band_values = np.full(len(group_data), np.nan, dtype=float)

                    band_data_cols.append(band_values)

                if valid_samples == 0:
                    print(
                        f"DEBUG:   No hay muestras válidas para "
                        f"Ng={ng_val} en {date_str}"
                    )
                    continue

                # band_data: shape (muestras, bandas)
                band_data = np.vstack(band_data_cols).T

                print(f"DEBUG:   Forma de band_data: {band_data.shape}")
                print(f"DEBUG:   Muestras válidas (max por banda): {valid_samples}")

                # ---------- Estadísticas ----------
                mean_vals = np.nanmean(band_data, axis=0)
                std_vals = np.nanstd(band_data, axis=0)

                print(f"DEBUG:   Valores promedio: {mean_vals}")
                print(f"DEBUG:   Desviaciones estándar: {std_vals}")

                # Clipping
                mean_vals = np.clip(mean_vals, 0, 10000)
                upper_std = np.clip(mean_vals + std_vals, 0, 10000)
                lower_std = np.clip(mean_vals - std_vals, 0, 10000)

                # ---------- Eje X ----------
                if self.canvas.x_mode == 'lambda':
                    xvals = LAMBDA_NM
                else:
                    xvals = np.arange(len(BAND_NAMES))

                # Etiqueta informativa
                n_samples = len(group_data)
                label = f"{group_label_text} ({date_str}) - n={n_samples}"

                # ---------- Plot ----------
                self.canvas.plot_signature(
                    xvals,
                    mean_vals,
                    label,
                    upper_std,
                    lower_std,
                    color=color_hex,
                    marker=marker
                )
                plots_created += 1
                print(
                    f"DEBUG:   Plot creado para {label} con color {color_hex} "
                    f"y marker '{marker}'"
                )

        print(f"DEBUG: Total de plots creados: {plots_created}")

        if plots_created == 0:
            print("DEBUG: Revisando datos disponibles...")
            sample_ng_values = df_mcal['Ng'].unique()[:5]
            sample_fecha_values = df_mcal['Fecha'].unique()[:5]
            print(f"DEBUG: Valores de Ng en datos: {sample_ng_values}")
            print(f"DEBUG: Valores de Fecha en datos: {sample_fecha_values}")

            QtWidgets.QMessageBox.information(
                self,
                "Información",
                "No se encontraron datos para las combinaciones seleccionadas"
            )
        else:
            self.stats_label.setText(
                f"Se crearon {plots_created} firmas espectrales"
            )


# ------------------------------- Main Window -------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Spectral Signature (PyQt5)")
        self.resize(1300, 800)

        self.df_roi, self.df_list, self.df_mcal, self.df_mcal_hsl = load_dataframes()
        self._validate_list()

        self.active_date: Optional[str] = None
        self.active_path: Optional[str] = None
        self.ds: Optional[gdal.Dataset] = None
        self.is_classmap: bool = False
        self.show_mcal: bool = True
        self.mcal_filter_by_date: bool = True
        self.selected_pixels: List[PixelSel] = []

        self._build_ui()
        self._build_menus()
        self._load_first()

    def _validate_list(self):
        if self.df_list is None or self.df_list.empty:
            raise RuntimeError("04-ROI-MOD.csv está vacío o no se pudo leer")
        bad = []
        for _, row in self.df_list.iterrows():
            ruta = row.get('Ruta') or row.get('ruta')
            if not ruta or not os.path.exists(ruta):
                bad.append(ruta)
        if bad:
            print(f"Advertencia: {len(bad)} rutas no existen. Se omitirán en UI.")
            self.df_list = self.df_list[self.df_list['Ruta'].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
        if self.df_list.empty:
            raise RuntimeError("No hay rutas válidas en 04-ROI-MOD.csv")

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, central)

        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)

        self.list_dates = QtWidgets.QListWidget()
        for f in self.df_list['Fecha'].astype(str).unique():
            self.list_dates.addItem(f)
        self.list_dates.currentTextChanged.connect(self._on_date_changed)

        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.addItems(list(PRESETS.keys()))
        self.cmb_preset.setCurrentText(CURRENT_PRESET)
        self.cmb_preset.currentTextChanged.connect(self._redraw_image)

        self.chk_mcal = QtWidgets.QCheckBox("Mostrar Mcal")
        self.chk_mcal.setChecked(True)
        self.chk_mcal.stateChanged.connect(self._toggle_mcal)

        self.chk_mcal_filter = QtWidgets.QCheckBox("Filtrar Mcal por fecha activa")
        self.chk_mcal_filter.setChecked(True)
        self.chk_mcal_filter.stateChanged.connect(self._toggle_mcal_filter)

        vleft.addWidget(QtWidgets.QLabel("Fechas"))
        vleft.addWidget(self.list_dates)
        vleft.addWidget(QtWidgets.QLabel("Preset RGB"))
        vleft.addWidget(self.cmb_preset)
        vleft.addWidget(self.chk_mcal)
        vleft.addWidget(self.chk_mcal_filter)
        vleft.addStretch(1)

        center = QtWidgets.QWidget()
        vcenter = QtWidgets.QVBoxLayout(center)

        self.img_canvas = ImageCanvas(center)
        self.img_canvas.pixelClicked.connect(self._on_pixel_clicked)
        vcenter.addWidget(self.img_canvas)

        right = QtWidgets.QWidget()
        vright = QtWidgets.QVBoxLayout(right)

        self.btn_open_sigs = QtWidgets.QPushButton("Abrir ventana de Firmas")
        self.btn_open_sigs.clicked.connect(self._open_sig_window)

        self.btn_save_png = QtWidgets.QPushButton("Guardar PNG (RGB actual)")
        self.btn_save_png.clicked.connect(self._save_png_current)

        self.btn_save_all = QtWidgets.QPushButton("Guardar TODAS las imágenes (lote)")
        self.btn_save_all.clicked.connect(self._save_all_images)

        self.btn_export_csv = QtWidgets.QPushButton("Exportar firmas CSV (sesión)")
        self.btn_export_csv.clicked.connect(self._export_csv)

        self.btn_save_proj = QtWidgets.QPushButton("Guardar proyecto (*.json)")
        self.btn_save_proj.clicked.connect(self._save_project)

        self.btn_load_proj = QtWidgets.QPushButton("Cargar proyecto (*.json)")
        self.btn_load_proj.clicked.connect(self._load_project)

        vright.addWidget(self.btn_open_sigs)
        vright.addWidget(self.btn_save_png)
        vright.addWidget(self.btn_save_all)
        vright.addWidget(self.btn_export_csv)
        vright.addSpacing(15)
        vright.addWidget(self.btn_save_proj)
        vright.addWidget(self.btn_load_proj)
        vright.addStretch(1)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([220, 800, 220])

        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(splitter)

        self.sig_win = SignatureWindow(self)
        self.status = self.statusBar()

    def _build_menus(self):
        bar = self.menuBar()
        m_file = bar.addMenu("Archivo")
        act_exit = QtWidgets.QAction("Salir", self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_data = bar.addMenu("Datos")
        # MCAL
        m_mcal = m_data.addMenu("Seleccionar MCAL")
        act_mcal_std = QtWidgets.QAction("Usar Mcal_py.csv", self)
        act_mcal_std.triggered.connect(lambda: self._set_mcal_source('Mcal_py.csv'))
        act_mcal_hsl = QtWidgets.QAction("Usar McalHSL_mod", self)
        act_mcal_hsl.triggered.connect(lambda: self._set_mcal_source('McalHSL_mod'))
        act_mcal_custom = QtWidgets.QAction("Cargar MCAL desde archivo...", self)
        act_mcal_custom.triggered.connect(self._load_mcal_from_file)
        m_mcal.addActions([act_mcal_std, act_mcal_hsl, act_mcal_custom])

        # Lista de imágenes
        m_list = m_data.addMenu("Seleccionar lista de imágenes")
        act_list_default = QtWidgets.QAction("Usar 04-ROI-MOD.csv", self)
        act_list_default.triggered.connect(self._use_default_list)
        act_list_custom = QtWidgets.QAction("Cargar lista CSV...", self)
        act_list_custom.triggered.connect(self._load_list_from_file)
        m_list.addActions([act_list_default, act_list_custom])

        m_view = bar.addMenu("Ver")
        act_view_avg_sigs = QtWidgets.QAction("Firmas espectrales promedio (MCAL)", self)
        act_view_avg_sigs.triggered.connect(self._open_avg_sig_window)
        m_view.addAction(act_view_avg_sigs)

    def _open_avg_sig_window(self):
        """Abrir ventana de firmas espectrales promedio"""
        if not hasattr(self, 'avg_sig_win'):
            self.avg_sig_win = AverageSignatureWindow(self)
        
        # Cargar datos en la ventana
        self.avg_sig_win.load_groups_and_dates()
        
        # Mostrar ventana
        self.avg_sig_win.show()
        self.avg_sig_win.raise_()


    def _load_first(self):
        self.list_dates.setCurrentRow(0)

    def _on_date_changed(self, fecha: str):
        self.active_date = fecha
        subset = self.df_list[self.df_list['Fecha'].astype(str)==fecha]
        if subset.empty:
            self.status.showMessage(f"Sin ruta para fecha {fecha}")
            return
        self.active_path = subset.iloc[0]['Ruta']
        self._open_dataset_and_draw()

    def _toggle_mcal(self, st):
        self.show_mcal = (st == Qt.Checked)
        self._plot_mcal()

    def _toggle_mcal_filter(self, st):
        self.mcal_filter_by_date = (st == Qt.Checked)
        self._plot_mcal()

    def _redraw_image(self):
        if self.ds is None: return
        self._draw_image_or_classmap()
        self._plot_mcal()

    def _open_dataset_and_draw(self):
        if self.ds is not None:
            self.ds = None
        try:
            ds = open_dataset(self.active_path)
            self.ds = ds
            self.is_classmap = ("CLASSMAP" in os.path.basename(self.active_path).upper())
            self._draw_image_or_classmap()
            self._plot_mcal()
            H, W = ds.RasterYSize, ds.RasterXSize
            self.status.showMessage(f"{os.path.basename(self.active_path)} | {W}x{H} | {'CLASSMAP' if self.is_classmap else 'multibanda'}")
        except Exception as e:
            self.status.showMessage(f"Error al cargar: {e}")
            traceback.print_exc()

    def _draw_image_or_classmap(self):
        if self.ds is None: return
        if self.is_classmap:
            band = self.ds.GetRasterBand(1)
            cm = band.ReadAsArray()
            self.img_canvas.show_classmap(cm, CLASSMAP_COLORS)
        else:
            preset = self.cmb_preset.currentText()
            rgb = build_rgb(self.ds, preset)
            self.img_canvas.show_image(rgb)

    def _current_mcal_complete(self) -> pd.DataFrame:
        """Obtener datos MCAL completos sin filtrar por fecha"""
        return getattr(self, "current_mcal_df", self.df_mcal).copy()

    def _current_mcal_filtered(self) -> pd.DataFrame:
        """Obtener datos MCAL filtrados por fecha activa (si está habilitado)"""
        df = self._current_mcal_complete()
        if self.mcal_filter_by_date and self.active_date is not None and 'Fecha' in df.columns:
            df = df[df['Fecha'].astype(str) == str(self.active_date)]
        return df

    def _set_mcal_source(self, key: str):
        if key == 'Mcal_py.csv':
            self.current_mcal_df = self.df_mcal.copy()
        elif key == 'McalHSL_mod':
            self.current_mcal_df = self.df_mcal_hsl.copy()
        self._plot_mcal()

    def _load_mcal_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar MCAL CSV", "", "CSV (*.csv)")
        if fn:
            self.current_mcal_df = read_csv_file(fn, '#')
            self._plot_mcal()
            self.status.showMessage(f"MCAL cargado: {fn}")

    def _use_default_list(self):
        self.df_roi, self.df_list, _, _ = load_dataframes()
        self._refresh_dates_list()

    def _load_list_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar lista de imágenes CSV", "", "CSV (*.csv)")
        if fn:
            self.df_list = read_csv_file(fn, '#')
            self._validate_list()
            self._refresh_dates_list()
            self.status.showMessage(f"Lista cargada: {fn}")

    def _refresh_dates_list(self):
        self.list_dates.clear()
        for f in self.df_list['Fecha'].astype(str).unique():
            self.list_dates.addItem(f)
        if self.list_dates.count() > 0:
            self.list_dates.setCurrentRow(0)


    def _plot_mcal(self):
        if not self.show_mcal:
            self.img_canvas.clear_mcal(); return
        if self.ds is None:
            self.img_canvas.clear_mcal(); return
        df = self._current_mcal_filtered()
        if df is None or df.empty:
            self.img_canvas.clear_mcal(); return
        labels = []
        for _, r in df.iterrows():
            ng = r.get('Ng', '')
            gname = get_group_name(ng)
            labels.append(f"Ng{ng} - {gname}")
        self.img_canvas.set_mcal_points(df, labels)

    def _on_pixel_clicked(self, i: int, j: int):
        pid = f"P{len(self.selected_pixels)+1}"
        self.selected_pixels.append(PixelSel(pid, i, j))
        self._update_signatures_view()

    def _get_signature(self, ds: gdal.Dataset, i: int, j: int) -> np.ndarray:
        nb = ds.RasterCount
        vals = np.zeros(nb, dtype=float)
        for b in range(1, nb+1):
            vals[b-1] = read_band_sample(ds, b, i, j)
        vals = np.clip(vals, 0, 10000)
        return vals

    def _update_signatures_view(self):
        if self.active_date is None: return
        rows = []
        self.sig_win.canvas.clear()

        fechas = list(self.df_list['Fecha'].astype(str).unique())

        for ps in self.selected_pixels:
            for fecha in fechas:
                subset = self.df_list[self.df_list['Fecha'].astype(str)==fecha]
                if subset.empty: continue
                ruta = subset.iloc[0]['Ruta']
                try:
                    ds = open_dataset(ruta)
                    vals = self._get_signature(ds, ps.i, ps.j)
                except Exception:
                    continue
                if self.sig_win.canvas.x_mode == 'lambda':
                    xvals = LAMBDA_NM
                else:
                    xvals = np.arange(len(BAND_NAMES))
                label = f"{ps.pid} ({ps.i},{ps.j}) - {fecha}"
                self.sig_win.canvas.plot_signature(xvals, vals, label)
                rows.append((ps.pid, ps.i, ps.j, fecha))

        self.sig_win.update_table(rows)

    def _open_mcal_sig_window(self):
        """Abrir ventana para ver firmas espectrales del MCAL"""
        if not hasattr(self, 'mcal_sig_win'):
            self.mcal_sig_win = SignatureWindow(self)
            self.mcal_sig_win.setWindowTitle("Firmas espectrales del MCAL")
        
        # Limpiar el canvas
        self.mcal_sig_win.canvas.clear()
        
        # Obtener datos MCAL filtrados
        df_mcal = self._current_mcal_filtered()
        if df_mcal.empty:
            self.status.showMessage("No hay datos MCAL para mostrar")
            return
        
        # Preparar datos para la tabla
        rows = []
        
        # Plotear firmas espectrales del MCAL
        for idx, row in df_mcal.iterrows():
            # Asumiendo que las columnas de bandas están nombradas como B01, B02, etc.
            band_vals = []
            for band_name in BAND_NAMES:
                if band_name in df_mcal.columns:
                    band_vals.append(row[band_name])
                else:
                    band_vals.append(np.nan)
            
            band_vals = np.array(band_vals, dtype=float)
            band_vals = np.clip(band_vals, 0, 10000)
            
            # Crear etiqueta
            ng = row.get('Ng', '')
            gname = get_group_name(ng)
            fecha = row.get('Fecha', '')
            label = f"Ng{ng} - {gname} ({fecha})"
            color_hex = get_group_color_hex(ng)
            
            # Plotear
            if self.mcal_sig_win.canvas.x_mode == 'lambda':
                xvals = LAMBDA_NM
            else:
                xvals = np.arange(len(BAND_NAMES))
            
            self.mcal_sig_win.canvas.plot_signature(xvals, band_vals, label, color=color_hex)
            
            # Agregar a tabla
            i_val = row.get('i', '')
            j_val = row.get('j', '')
            rows.append((f"Ng{ng} - {gname}", i_val, j_val, str(fecha)))
        
        # Actualizar tabla
        self.mcal_sig_win.update_table(rows)
        
        # Mostrar ventana
        self.mcal_sig_win.show()
        self.mcal_sig_win.raise_()
        self.status.showMessage(f"Mostrando {len(df_mcal)} firmas del MCAL")

    def _open_sig_window(self):
        if not self.sig_win.isVisible():
            self.sig_win.show()
        self._update_signatures_view()

    def _export_csv(self):
        if not self.selected_pixels:
            self.status.showMessage("No hay píxeles seleccionados")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar firmas CSV", "firmas_sesion.csv", "CSV (*.csv)")
        if not fn: return
        rows = []
        fechas = list(self.df_list['Fecha'].astype(str).unique())
        for ps in self.selected_pixels:
            for fecha in fechas:
                subset = self.df_list[self.df_list['Fecha'].astype(str)==fecha]
                if subset.empty: continue
                ruta = subset.iloc[0]['Ruta']
                try:
                    ds = open_dataset(ruta)
                    vals = self._get_signature(ds, ps.i, ps.j)
                except Exception:
                    continue
                for b_idx, bname in enumerate(BAND_NAMES):
                    rows.append({
                        'pixel_id': ps.pid,
                        'i': ps.i,
                        'j': ps.j,
                        'Fecha': fecha,
                        'band': bname,
                        'lambda_nm': LAMBDA_NM[b_idx],
                        'reflectance_scaled': float(vals[b_idx])
                    })
        if rows:
            pd.DataFrame(rows).to_csv(fn, index=False)
            self.status.showMessage(f"Firmas exportadas: {fn}")

    def _save_png_current(self):
        if self.ds is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar PNG", "rgb_actual.png", "PNG (*.png)")
        if not fn: return
        if self.is_classmap:
            band = self.ds.GetRasterBand(1)
            cm = band.ReadAsArray()
            maxc = int(np.nanmax(cm)) if np.size(cm)>0 else 0
            lut = np.zeros((maxc+1, 3), dtype=np.uint8)
            for k in range(maxc+1):
                c = QtGui.QColor(CLASSMAP_COLORS.get(str(k), '#000000'))
                lut[k] = [int(c.red()*255/255), int(c.green()*255/255), int(c.blue()*255/255)]
            rgb_img = lut[cm.clip(min=0).astype(int)]
        else:
            preset = self.cmb_preset.currentText()
            rgb = build_rgb(self.ds, preset)
            rgb_img = (rgb*255).astype(np.uint8)
        plt.imsave(fn, rgb_img)
        self.status.showMessage(f"PNG guardado: {fn}")

    def _save_all_images(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino")
        if not folder: return
        for fecha in self.df_list['Fecha'].astype(str).unique():
            ruta = self.df_list[self.df_list['Fecha'].astype(str)==fecha].iloc[0]['Ruta']
            try:
                ds = open_dataset(ruta)
                base = os.path.splitext(os.path.basename(ruta))[0]
                out = os.path.join(folder, f"{base}_RGB_{self.cmb_preset.currentText()}.png")
                if "CLASSMAP" in base.upper():
                    band = ds.GetRasterBand(1)
                    cm = band.ReadAsArray()
                    maxc = int(np.nanmax(cm)) if np.size(cm)>0 else 0
                    lut = np.zeros((maxc+1, 3), dtype=np.uint8)
                    for k in range(maxc+1):
                        c = QtGui.QColor(CLASSMAP_COLORS.get(str(k), '#000000'))
                        lut[k] = [int(c.red()*255/255), int(c.green()*255/255), int(c.blue()*255/255)]
                    rgb_img = lut[cm.clip(min=0).astype(int)]
                else:
                    rgb = build_rgb(ds, self.cmb_preset.currentText())
                    rgb_img = (rgb*255).astype(np.uint8)
                plt.imsave(out, rgb_img)
                self.status.showMessage(f"Guardada {out}")
            except Exception as e:
                print(f"Error guardando {ruta}: {e}")
        self.status.showMessage("Lote completado")

    def _save_project(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar proyecto", "proyecto.json", "JSON (*.json)")
        if not fn: return
        state = ProjectState(
            active_date=self.active_date,
            xaxis_mode='lambda' if self.sig_win.chk_lambda.isChecked() else 'band',
            show_mcal=self.show_mcal,
            mcal_filter_by_date=self.mcal_filter_by_date,
            preset=self.cmb_preset.currentText(),
            selected_pixels=self.selected_pixels,
            loaded_paths={f: self.df_list[self.df_list['Fecha'].astype(str)==f].iloc[0]['Ruta'] for f in self.df_list['Fecha'].astype(str).unique()}
        )
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
        self.status.showMessage(f"Proyecto guardado: {fn}")

    def _load_project(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Cargar proyecto", "", "JSON (*.json)")
        if not fn: return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self.active_date = d.get('active_date')
            xmode = d.get('xaxis_mode','lambda')
            self.sig_win.chk_lambda.setChecked(xmode=='lambda')
            self.show_mcal = bool(d.get('show_mcal', True))
            self.chk_mcal.setChecked(self.show_mcal)
            self.mcal_filter_by_date = bool(d.get('mcal_filter_by_date', True))
            self.chk_mcal_filter.setChecked(self.mcal_filter_by_date)
            preset = d.get('preset', 'TrueColor')
            if preset in PRESETS: self.cmb_preset.setCurrentText(preset)
            sps = []
            for sp in d.get('selected_pixels', []):
                sps.append(PixelSel(sp['pid'], int(sp['i']), int(sp['j'])))
            self.selected_pixels = sps
            if self.active_date:
                items = self.list_dates.findItems(self.active_date, Qt.MatchExactly)
                if items:
                    self.list_dates.setCurrentItem(items[0])
            self._update_signatures_view()
            self.status.showMessage(f"Proyecto cargado: {fn}")
        except Exception as e:
            self.status.showMessage(f"Error al cargar proyecto: {e}")

# --------------------------------- Main ------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()