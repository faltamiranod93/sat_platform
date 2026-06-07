# -*- coding: utf-8 -*-
"""g_Sat_SpectralSignature_v2_ng0based_inputs.py

Variante de g_Sat_SpectralSignature_v2.py con:
  - Convención Ng 0-based (Ng=0 == "no class"), consistente con config_bandas_v2.json.
  - Colores por clase desde config (fallback HSV si faltan).
  - Carga de inputs (config_bandas, MCAL, ROI list) vía menú/diálogos, estilo PixelClass.
  - Elimina dependencias duras de rutas hardcodeadas; intenta autodetectar archivos en el
    directorio del script o CWD, y si no existen, deja la UI lista para cargar manualmente.

NOTA: Mantiene la estructura original (globals) para minimizar deuda técnica.
"""

import os
import sys
import json
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


# ---------------------------------------------------------------------------
# Utilidades (CSV robusto)
# ---------------------------------------------------------------------------

def read_csv_file(csv_path: str, comment: str = '#') -> pd.DataFrame:
    """Lector robusto: intenta separadores comunes sin romper el pipeline."""
    last_err = None
    for sep in (',', ';', '\t'):
        try:
            df = pd.read_csv(csv_path, sep=sep, comment=comment)
            if len(df.columns) > 1:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return pd.read_csv(csv_path, comment=comment)


# ---------------------------------------------------------------------------
# Autodetección de paths (sin hardcode)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


DEFAULT_CONFIG = _first_existing([
    os.path.join(SCRIPT_DIR, 'config_bandas_v2.json'),
    os.path.join(os.getcwd(), 'config_bandas_v2.json'),
    os.path.join(SCRIPT_DIR, 'config_bandas_v3.json'),
    os.path.join(os.getcwd(), 'config_bandas_v3.json'),
])

DEFAULT_MCAL = _first_existing([
    os.path.join(SCRIPT_DIR, 'Mcal_py.csv'),
    os.path.join(os.getcwd(), 'Mcal_py.csv'),
])

DEFAULT_MCAL_HSL = _first_existing([
    os.path.join(SCRIPT_DIR, 'McalHSL_mod_v5_py.csv'),
    os.path.join(SCRIPT_DIR, 'McalHSL_mod_v4_py.csv'),
    os.path.join(os.getcwd(), 'McalHSL_mod_v5_py.csv'),
    os.path.join(os.getcwd(), 'McalHSL_mod_v4_py.csv'),
])

DEFAULT_LIST = _first_existing([
    os.path.join(SCRIPT_DIR, '04-ROI-MOD.csv'),
    os.path.join(SCRIPT_DIR, '02-Space-Facilities', '04-ROI-MOD.csv'),
    os.path.join(os.getcwd(), '04-ROI-MOD.csv'),
    os.path.join(os.getcwd(), '02-Space-Facilities', '04-ROI-MOD.csv'),
])


# ---------------------------------------------------------------------------
# Config bandas (recargable)
# ---------------------------------------------------------------------------

_cfg_raw: Dict = {}
BAND_NAMES: List[str] = []
LAMBDA_NM: np.ndarray = np.array([], dtype=float)
BAND_INDEX: Dict[str, int] = {}
CLASSMAP_COLORS: Dict[str, str] = {}
PRESETS: Dict[str, List[int]] = {}

GROUP_NAMES: List[str] = []
GROUP_COLORS_HEX: List[str] = []
NUM_GROUPS: int = 0


def _rgb_to_hex(rgb) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _bands_from_legacy(cfg_old: Dict):
    band_names = [b['name'] for b in cfg_old['bands']]
    lambdas = np.array([b['lambda'] for b in cfg_old['bands']], dtype=float)
    band_index = {b: i + 1 for i, b in enumerate(band_names)}
    classmap_colors = cfg_old.get('groups', {}).get('classmap', {})
    presets = cfg_old.get('presets', {
        "TrueColor": [4, 3, 2],
        "FalseColor": [8, 4, 3],
        "SWIR": [12, 8, 4]
    })

    # Para legacy, intentamos mapear a 0-based si hay 'nameg'/'color'; si no, solo nombres.
    nameg = list(cfg_old.get("nameg", []))
    color = list(cfg_old.get("color", []))
    group_colors_hex = [_rgb_to_hex(c) for c in color] if (nameg and color and len(nameg) == len(color)) else []

    return band_names, lambdas, band_index, classmap_colors, presets, nameg, group_colors_hex


def _bands_from_current(cfg_new: Dict):
    required = ("Nband", "lam", "Nband_sort")
    if not all(k in cfg_new for k in required):
        raise KeyError("Faltan llaves requeridas: Nband, lam, Nband_sort")

    lam_map = {bn: float(lv) for bn, lv in zip(cfg_new["Nband"], cfg_new["lam"])}
    band_names = list(cfg_new["Nband_sort"])
    lambdas = np.array([lam_map[b] for b in band_names], dtype=float)
    band_index = {b: i + 1 for i, b in enumerate(band_names)}  # GDAL bands 1-based

    nameg = list(cfg_new.get("nameg", []))
    color = list(cfg_new.get("color", []))
    if nameg and color and len(nameg) != len(color):
        raise ValueError("Longitudes distintas entre nameg y color")

    # Colores de classmap y MCAL por Ng (0-based)
    classmap_colors = {str(i): _rgb_to_hex(rgb) for i, rgb in enumerate(color)} if color else {}
    group_colors_hex = [_rgb_to_hex(rgb) for rgb in color] if color else []

    def _preset(bnames):
        return [band_index[b] for b in bnames]

    presets = {
        "TrueColor": _preset(["B04", "B03", "B02"]),
        "FalseColor": _preset(["B08", "B04", "B03"]),
        "SWIR": _preset(["B12", "B11", "B04"]),
    }

    return band_names, lambdas, band_index, classmap_colors, presets, nameg, group_colors_hex


def reload_config(config_path: str):
    """Recarga config_bandas y actualiza globals.

    Convención: Ng es 0-based (índice directo en nameg/color).
    """
    global _cfg_raw, BAND_NAMES, LAMBDA_NM, BAND_INDEX, CLASSMAP_COLORS, PRESETS
    global GROUP_NAMES, GROUP_COLORS_HEX, NUM_GROUPS

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    _cfg_raw = cfg

    if "bands" in cfg:
        (BAND_NAMES, LAMBDA_NM, BAND_INDEX,
         CLASSMAP_COLORS, PRESETS, GROUP_NAMES, GROUP_COLORS_HEX) = _bands_from_legacy(cfg)
    else:
        (BAND_NAMES, LAMBDA_NM, BAND_INDEX,
         CLASSMAP_COLORS, PRESETS, GROUP_NAMES, GROUP_COLORS_HEX) = _bands_from_current(cfg)

    NUM_GROUPS = len(GROUP_NAMES) if GROUP_NAMES else 0


# Cargar config por defecto si existe
if DEFAULT_CONFIG:
    reload_config(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Colores y nombres por Ng (0-based)
# ---------------------------------------------------------------------------

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


_PALETTE_FALLBACK = generate_color_palette(NUM_GROUPS) if NUM_GROUPS > 0 else []


def get_group_color_hex(ng_value) -> str:
    """Color por clase.

    Regla:
      - Si el config trae "color", usamos ese (consistencia PixelClass/ClassMap).
      - Si no, fallback HSV estable.

    Convención: Ng 0-based.
    """
    try:
        ng_int = int(ng_value)
    except Exception:
        return "#FFEE00"

    if NUM_GROUPS <= 0:
        return "#FFEE00"

    if GROUP_COLORS_HEX and 0 <= ng_int < len(GROUP_COLORS_HEX):
        return GROUP_COLORS_HEX[ng_int]

    # fallback HSV
    idx = ng_int % NUM_GROUPS
    return _rgb_to_hex(_PALETTE_FALLBACK[idx])


def get_group_name(ng_value) -> str:
    """Nombre de grupo desde config (0-based)."""
    try:
        ng_int = int(ng_value)
    except Exception:
        return f"Grupo {ng_value}"

    if 0 <= ng_int < len(GROUP_NAMES):
        return GROUP_NAMES[ng_int]
    return f"Grupo {ng_int}"


# ---------------------------------------------------------------------------
# Modelos UI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers GDAL
# ---------------------------------------------------------------------------

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
    if not PRESETS:
        raise RuntimeError("PRESETS no está cargado (falta config_bandas).")
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


# ---------------------------------------------------------------------------
# Canvas Imagen
# ---------------------------------------------------------------------------

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
        self._mcal_labels = []
        self._cid = self.mpl_connect('button_press_event', self._on_click)
        self._hid = self.mpl_connect('motion_notify_event', self._on_hover)
        self.img_shape = None

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
            colors = '#FFFF00'

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
            self.pixelClicked.emit(i, j)

    def _on_hover(self, event):
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


# ---------------------------------------------------------------------------
# Canvas Firmas
# ---------------------------------------------------------------------------

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
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

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
        self.canvas.set_xmode('lambda' if st == Qt.Checked else 'band')

    def update_table(self, rows: List[Tuple[str, int, int, str]]):
        self.table.setRowCount(len(rows))
        for r, (pid, i, j, fecha) in enumerate(rows):
            for c, val in enumerate([pid, i, j, fecha]):
                item = QtWidgets.QTableWidgetItem(str(val))
                self.table.setItem(r, c, item)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Spectral Signature (PyQt5)")
        self.resize(1300, 800)

        # Paths cargados (estilo PixelClass)
        self.config_path: Optional[str] = DEFAULT_CONFIG
        self.list_path: Optional[str] = DEFAULT_LIST
        self.mcal_path: Optional[str] = DEFAULT_MCAL
        self.mcal_hsl_path: Optional[str] = DEFAULT_MCAL_HSL

        self.df_list = pd.DataFrame()
        self.df_mcal = pd.DataFrame()
        self.df_mcal_hsl = pd.DataFrame()

        self.active_date: Optional[str] = None
        self.active_path: Optional[str] = None
        self.ds: Optional[gdal.Dataset] = None
        self.is_classmap: bool = False
        self.show_mcal: bool = True
        self.mcal_filter_by_date: bool = True
        self.selected_pixels: List[PixelSel] = []

        self._build_ui()
        self._build_menus()
        self._load_defaults_if_any()


    # --------------------- carga inicial ---------------------
    def _load_defaults_if_any(self):
        # Config
        if self.config_path and os.path.exists(self.config_path):
            try:
                reload_config(self.config_path)
            except Exception as e:
                self._warn(f"No pude cargar config por defecto: {e}")

        self._update_labels()

        # Lista de imágenes
        if self.list_path and os.path.exists(self.list_path):
            try:
                self.df_list = read_csv_file(self.list_path, '#')
                self._validate_list_soft()
                self._refresh_dates_list()
            except Exception as e:
                self._warn(f"No pude cargar ROI list por defecto: {e}")

        # MCAL
        if self.mcal_path and os.path.exists(self.mcal_path):
            try:
                self.df_mcal = read_csv_file(self.mcal_path, '#')
            except Exception as e:
                self._warn(f"No pude cargar MCAL por defecto: {e}")
        if self.mcal_hsl_path and os.path.exists(self.mcal_hsl_path):
            try:
                self.df_mcal_hsl = read_csv_file(self.mcal_hsl_path, '#')
            except Exception as e:
                self._warn(f"No pude cargar MCAL_HSL por defecto: {e}")

        self._set_mcal_source('Mcal_py.csv' if not self.df_mcal.empty else 'McalHSL_mod')

        # Si ya hay fechas válidas, cargar la primera
        if self.list_dates.count() > 0:
            self.list_dates.setCurrentRow(0)


    def _validate_list_soft(self):
        """Validación no-fatal: deja UI operativa."""
        if self.df_list is None or self.df_list.empty:
            self.statusBar().showMessage("ROI list vacío. Carga un 04-ROI-MOD.csv")
            return
        # Filtrar rutas inexistentes
        if 'Ruta' not in self.df_list.columns or 'Fecha' not in self.df_list.columns:
            raise RuntimeError("ROI list debe tener columnas 'Fecha' y 'Ruta'")
        ok = self.df_list['Ruta'].apply(lambda p: isinstance(p, str) and os.path.exists(p))
        n_bad = int((~ok).sum())
        if n_bad > 0:
            self.statusBar().showMessage(f"Advertencia: {n_bad} rutas no existen; se omiten")
        self.df_list = self.df_list[ok].copy()


    # --------------------- UI ---------------------
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, central)

        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)

        self.lbl_config = QtWidgets.QLabel("Config: (no cargado)")
        self.lbl_list = QtWidgets.QLabel("ROI list: (no cargado)")
        self.lbl_mcal = QtWidgets.QLabel("MCAL: (no cargado)")

        self.list_dates = QtWidgets.QListWidget()
        self.list_dates.currentTextChanged.connect(self._on_date_changed)

        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.currentTextChanged.connect(self._redraw_image)

        self.chk_mcal = QtWidgets.QCheckBox("Mostrar Mcal")
        self.chk_mcal.setChecked(True)
        self.chk_mcal.stateChanged.connect(self._toggle_mcal)

        self.chk_mcal_filter = QtWidgets.QCheckBox("Filtrar Mcal por fecha activa")
        self.chk_mcal_filter.setChecked(True)
        self.chk_mcal_filter.stateChanged.connect(self._toggle_mcal_filter)

        vleft.addWidget(self.lbl_config)
        vleft.addWidget(self.lbl_list)
        vleft.addWidget(self.lbl_mcal)
        vleft.addSpacing(10)
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

        vright.addWidget(self.btn_open_sigs)
        vright.addWidget(self.btn_save_png)
        vright.addStretch(1)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([260, 800, 220])

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

        act_cfg = QtWidgets.QAction("Cargar config_bandas (JSON)...", self)
        act_cfg.triggered.connect(self._load_config_from_file)
        m_data.addAction(act_cfg)

        m_mcal = m_data.addMenu("Seleccionar MCAL")
        act_mcal_std = QtWidgets.QAction("Usar Mcal_py.csv", self)
        act_mcal_std.triggered.connect(lambda: self._set_mcal_source('Mcal_py.csv'))
        act_mcal_hsl = QtWidgets.QAction("Usar McalHSL_mod", self)
        act_mcal_hsl.triggered.connect(lambda: self._set_mcal_source('McalHSL_mod'))
        act_mcal_custom = QtWidgets.QAction("Cargar MCAL desde archivo...", self)
        act_mcal_custom.triggered.connect(self._load_mcal_from_file)
        m_mcal.addActions([act_mcal_std, act_mcal_hsl, act_mcal_custom])

        m_list = m_data.addMenu("Seleccionar lista de imágenes")
        act_list_custom = QtWidgets.QAction("Cargar lista CSV...", self)
        act_list_custom.triggered.connect(self._load_list_from_file)
        m_list.addAction(act_list_custom)


    # --------------------- loaders ---------------------
    def _update_labels(self):
        self.lbl_config.setText(f"Config: {os.path.basename(self.config_path) if self.config_path else '(no cargado)'}")
        self.lbl_list.setText(f"ROI list: {os.path.basename(self.list_path) if self.list_path else '(no cargado)'}")
        self.lbl_mcal.setText(f"MCAL: {os.path.basename(self.mcal_path) if self.mcal_path else '(no cargado)'}")

        # presets
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        self.cmb_preset.addItems(list(PRESETS.keys()) if PRESETS else [])
        if PRESETS and 'TrueColor' in PRESETS:
            self.cmb_preset.setCurrentText('TrueColor')
        self.cmb_preset.blockSignals(False)


    def _warn(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Aviso", msg)


    def _load_config_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar config_bandas", "", "JSON (*.json)")
        if not fn:
            return
        try:
            reload_config(fn)
            self.config_path = fn
            self._update_labels()
            # Replot
            self._redraw_image()
            self._plot_mcal()
            self.status.showMessage(f"Config cargado: {fn}")
        except Exception as e:
            self._warn(f"Error al cargar config: {e}\n\n{traceback.format_exc()}")


    def _load_mcal_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar MCAL CSV", "", "CSV (*.csv)")
        if not fn:
            return
        try:
            df = read_csv_file(fn, '#')
            # Decide si es HSL_mod o Mcal_py por columnas
            if all(c in df.columns for c in ['H', 'S', 'L']):
                self.df_mcal_hsl = df
                self.mcal_hsl_path = fn
                self._set_mcal_source('McalHSL_mod')
            else:
                self.df_mcal = df
                self.mcal_path = fn
                self._set_mcal_source('Mcal_py.csv')
            self._update_labels()
            self._plot_mcal()
            self.status.showMessage(f"MCAL cargado: {fn}")
        except Exception as e:
            self._warn(f"Error al cargar MCAL: {e}\n\n{traceback.format_exc()}")


    def _load_list_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Seleccionar lista de imágenes CSV", "", "CSV (*.csv)")
        if not fn:
            return
        try:
            self.df_list = read_csv_file(fn, '#')
            self.list_path = fn
            self._validate_list_soft()
            self._refresh_dates_list()
            self._update_labels()
            self.status.showMessage(f"Lista cargada: {fn}")
            if self.list_dates.count() > 0:
                self.list_dates.setCurrentRow(0)
        except Exception as e:
            self._warn(f"Error al cargar lista: {e}\n\n{traceback.format_exc()}")


    def _refresh_dates_list(self):
        self.list_dates.clear()
        if self.df_list is None or self.df_list.empty or 'Fecha' not in self.df_list.columns:
            return
        for f in self.df_list['Fecha'].astype(str).unique():
            self.list_dates.addItem(f)


    # --------------------- eventos ---------------------
    def _on_date_changed(self, fecha: str):
        self.active_date = fecha
        if self.df_list is None or self.df_list.empty:
            return
        subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
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
        if self.ds is None:
            return
        self._draw_image_or_classmap()
        self._plot_mcal()


    def _open_dataset_and_draw(self):
        try:
            self.ds = open_dataset(self.active_path)
            self.is_classmap = ("CLASSMAP" in os.path.basename(self.active_path).upper())
            self._draw_image_or_classmap()
            self._plot_mcal()
            H, W = self.ds.RasterYSize, self.ds.RasterXSize
            self.status.showMessage(f"{os.path.basename(self.active_path)} | {W}x{H} | {'CLASSMAP' if self.is_classmap else 'multibanda'}")
        except Exception as e:
            self.status.showMessage(f"Error al cargar: {e}")
            traceback.print_exc()


    def _draw_image_or_classmap(self):
        if self.ds is None:
            return
        if self.is_classmap:
            band = self.ds.GetRasterBand(1)
            cm = band.ReadAsArray()
            self.img_canvas.show_classmap(cm, CLASSMAP_COLORS)
        else:
            preset = self.cmb_preset.currentText() or 'TrueColor'
            rgb = build_rgb(self.ds, preset)
            self.img_canvas.show_image(rgb)


    def _current_mcal_complete(self) -> pd.DataFrame:
        return getattr(self, "current_mcal_df", self.df_mcal).copy()


    def _current_mcal_filtered(self) -> pd.DataFrame:
        df = self._current_mcal_complete()
        if self.mcal_filter_by_date and self.active_date is not None and 'Fecha' in df.columns:
            df = df[df['Fecha'].astype(str) == str(self.active_date)]
        return df


    def _set_mcal_source(self, key: str):
        if key == 'Mcal_py.csv':
            self.current_mcal_df = self.df_mcal.copy()
        elif key == 'McalHSL_mod':
            self.current_mcal_df = self.df_mcal_hsl.copy()
        else:
            self.current_mcal_df = self.df_mcal.copy()
        self._plot_mcal()


    def _plot_mcal(self):
        if not self.show_mcal or self.ds is None:
            self.img_canvas.clear_mcal();
            return
        df = self._current_mcal_filtered()
        if df is None or df.empty:
            self.img_canvas.clear_mcal();
            return
        labels = []
        for _, r in df.iterrows():
            ng = r.get('Ng', '')
            gname = get_group_name(ng)
            labels.append(f"Ng{ng} - {gname}")
        self.img_canvas.set_mcal_points(df, labels)


    def _on_pixel_clicked(self, i: int, j: int):
        pid = f"P{len(self.selected_pixels) + 1}"
        self.selected_pixels.append(PixelSel(pid, i, j))
        self._update_signatures_view()


    def _get_signature(self, ds: gdal.Dataset, i: int, j: int) -> np.ndarray:
        nb = ds.RasterCount
        vals = np.zeros(nb, dtype=float)
        for b in range(1, nb + 1):
            vals[b - 1] = read_band_sample(ds, b, i, j)
        vals = np.clip(vals, 0, 10000)
        return vals


    def _update_signatures_view(self):
        if self.df_list is None or self.df_list.empty:
            return
        rows = []
        self.sig_win.canvas.clear()

        fechas = list(self.df_list['Fecha'].astype(str).unique())

        for ps in self.selected_pixels:
            for fecha in fechas:
                subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
                if subset.empty:
                    continue
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


    def _open_sig_window(self):
        if not self.sig_win.isVisible():
            self.sig_win.show()
        self._update_signatures_view()


    def _save_png_current(self):
        if self.ds is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar PNG", "rgb_actual.png", "PNG (*.png)")
        if not fn:
            return
        if self.is_classmap:
            band = self.ds.GetRasterBand(1)
            cm = band.ReadAsArray()
            maxc = int(np.nanmax(cm)) if np.size(cm) > 0 else 0
            lut = np.zeros((maxc + 1, 3), dtype=np.uint8)
            for k in range(maxc + 1):
                c = QtGui.QColor(CLASSMAP_COLORS.get(str(k), '#000000'))
                lut[k] = [c.red(), c.green(), c.blue()]
            rgb_img = lut[np.clip(cm, 0, maxc).astype(int)]
        else:
            preset = self.cmb_preset.currentText() or 'TrueColor'
            rgb = build_rgb(self.ds, preset)
            rgb_img = (rgb * 255).astype(np.uint8)
        plt.imsave(fn, rgb_img)
        self.status.showMessage(f"PNG guardado: {fn}")


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
