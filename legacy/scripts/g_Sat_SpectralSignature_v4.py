# -*- coding: utf-8 -*-
"""
g_Sat_SpectralSignature_v4.py

Visor de imágenes satelitales y firmas espectrales (PyQt5 + Matplotlib).

Esta versión toma como base la portabilidad y consistencia de v3, e incorpora
varias funciones analíticas de v2:
  - Convención Ng 0-based (Ng=0 == "no class").
  - Colores por clase desde config_bandas (fallback HSV si faltan).
  - Carga flexible de config, ROI list y MCAL desde diálogos.
  - Firmas espectrales por píxel en todas las fechas.
  - Firmas promedio por grupo/fecha con desviación estándar.
  - Exportación de firmas de sesión a CSV.
  - Guardado/carga de proyecto en JSON.
  - Exportación PNG de la vista actual y guardado por lote.

Supuestos principales:
  - 04-ROI-MOD.csv contiene columnas 'Fecha' y 'Ruta'.
  - Los CSV MCAL que se quieran usar para firmas promedio deben contener las
    columnas espectrales definidas en BAND_NAMES (por ejemplo B01, B02, ...).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilidades de IO
# ---------------------------------------------------------------------------

def read_csv_file(csv_path: str, comment: str = '#') -> pd.DataFrame:
    """Lector robusto de CSV con separadores comunes."""
    last_err = None
    for sep in (',', ';', '\t'):
        try:
            df = pd.read_csv(csv_path, sep=sep, comment=comment)
            if len(df.columns) > 1:
                return df
        except Exception as exc:  # pragma: no cover - defensivo
            last_err = exc
    if last_err is not None:
        raise last_err
    return pd.read_csv(csv_path, comment=comment)


def _first_existing(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = _first_existing([
    os.path.join(SCRIPT_DIR, 'config_bandas_v2.json'),
    os.path.join(SCRIPT_DIR, 'config_bandas_v3.json'),
    os.path.join(SCRIPT_DIR, 'config_bandas.json'),
    os.path.join(os.getcwd(), 'config_bandas_v2.json'),
    os.path.join(os.getcwd(), 'config_bandas_v3.json'),
    os.path.join(os.getcwd(), 'config_bandas.json'),
])

DEFAULT_LIST = _first_existing([
    os.path.join(SCRIPT_DIR, '04-ROI-MOD.csv'),
    os.path.join(SCRIPT_DIR, '02-Space-Facilities', '04-ROI-MOD.csv'),
    os.path.join(os.getcwd(), '04-ROI-MOD.csv'),
    os.path.join(os.getcwd(), '02-Space-Facilities', '04-ROI-MOD.csv'),
])

DEFAULT_MCAL = _first_existing([
    os.path.join(SCRIPT_DIR, 'Mcal_py.csv'),
    os.path.join(os.getcwd(), 'Mcal_py.csv'),
])

DEFAULT_MCAL_HSL = _first_existing([
    os.path.join(SCRIPT_DIR, 'McalHSL_mod_v6_py.csv'),
    os.path.join(SCRIPT_DIR, 'McalHSL_mod_v5_py.csv'),
    os.path.join(SCRIPT_DIR, 'McalHSL_mod_v4_py.csv'),
    os.path.join(os.getcwd(), 'McalHSL_mod_v6_py.csv'),
    os.path.join(os.getcwd(), 'McalHSL_mod_v5_py.csv'),
    os.path.join(os.getcwd(), 'McalHSL_mod_v4_py.csv'),
])


# ---------------------------------------------------------------------------
# Configuración de bandas y grupos
# ---------------------------------------------------------------------------

_cfg_raw: Dict = {}
BAND_NAMES: List[str] = []
LAMBDA_NM: np.ndarray = np.array([], dtype=float)
BAND_INDEX: Dict[str, int] = {}
CLASSMAP_COLORS: Dict[str, str] = {}
PRESETS: Dict[str, List[int]] = {}
BAND_SEQUENCES: Dict[str, List[str]] = {}
GROUP_NAMES: List[str] = []
GROUP_COLORS_HEX: List[str] = []
NUM_GROUPS: int = 0
_PALETTE_FALLBACK: List[Tuple[int, int, int]] = []


def _rgb_to_hex(rgb) -> str:
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def generate_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    if num_colors <= 0:
        return [(31, 119, 180)]

    colors: List[Tuple[int, int, int]] = []
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


def _bands_from_legacy(cfg_old: Dict):
    band_names = [b['name'] for b in cfg_old['bands']]
    lambdas = np.array([b['lambda'] for b in cfg_old['bands']], dtype=float)
    band_index = {b: i + 1 for i, b in enumerate(band_names)}
    classmap_colors = cfg_old.get('groups', {}).get('classmap', {})
    presets = cfg_old.get('presets', {
        'TrueColor': [4, 3, 2],
        'FalseColor': [8, 4, 3],
        'SWIR': [12, 8, 4],
    })

    nameg = list(cfg_old.get('nameg', []))
    color = list(cfg_old.get('color', []))
    group_colors_hex = [_rgb_to_hex(c) for c in color] if (nameg and color and len(nameg) == len(color)) else []

    return band_names, lambdas, band_index, classmap_colors, presets, nameg, group_colors_hex


def _bands_from_current(cfg_new: Dict):
    required = ('Nband', 'lam', 'Nband_sort')
    if not all(k in cfg_new for k in required):
        raise KeyError('Faltan llaves requeridas: Nband, lam, Nband_sort')

    lam_map = {bn: float(lv) for bn, lv in zip(cfg_new['Nband'], cfg_new['lam'])}

    # Secuencias disponibles
    band_sequences: Dict[str, List[str]] = {}
    if cfg_new.get('Nband_sort'):
        band_sequences['Nband_sort'] = list(cfg_new['Nband_sort'])
    if cfg_new.get('Nband_filter'):
        band_sequences['Nband_filter'] = list(cfg_new['Nband_filter'])

    # Secuencia por defecto
    band_names = list(band_sequences.get('Nband_sort', cfg_new['Nband']))
    lambdas = np.array([lam_map[b] for b in band_names], dtype=float)
    band_index = {b: i + 1 for i, b in enumerate(band_names)}

    nameg = list(cfg_new.get('nameg', []))
    color = list(cfg_new.get('color', []))
    if nameg and color and len(nameg) != len(color):
        raise ValueError('Longitudes distintas entre nameg y color')

    classmap_colors = {str(i): _rgb_to_hex(rgb) for i, rgb in enumerate(color)} if color else {}
    group_colors_hex = [_rgb_to_hex(rgb) for rgb in color] if color else []

    def _preset(bnames: List[str]) -> List[int]:
        return [band_index[b] for b in bnames if b in band_index]

    presets = {
        'TrueColor': _preset(['B04', 'B03', 'B02']),
        'FalseColor': _preset(['B08', 'B04', 'B03']),
        'SWIR': _preset(['B12', 'B11', 'B04']),
    }

    return band_names, lambdas, band_index, classmap_colors, presets, nameg, group_colors_hex, band_sequences

def reload_config(config_path: str):
    global _cfg_raw, BAND_NAMES, LAMBDA_NM, BAND_INDEX, CLASSMAP_COLORS, PRESETS
    global GROUP_NAMES, GROUP_COLORS_HEX, NUM_GROUPS, _PALETTE_FALLBACK, BAND_SEQUENCES

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    _cfg_raw = cfg

    if 'bands' in cfg:
        (BAND_NAMES, LAMBDA_NM, BAND_INDEX,
         CLASSMAP_COLORS, PRESETS, GROUP_NAMES, GROUP_COLORS_HEX) = _bands_from_legacy(cfg)

        BAND_SEQUENCES = {
            'Nband_sort': list(BAND_NAMES)
        }
    else:
        (BAND_NAMES, LAMBDA_NM, BAND_INDEX,
         CLASSMAP_COLORS, PRESETS, GROUP_NAMES, GROUP_COLORS_HEX,
         BAND_SEQUENCES) = _bands_from_current(cfg)

    NUM_GROUPS = len(GROUP_NAMES) if GROUP_NAMES else 0
    _PALETTE_FALLBACK = generate_color_palette(NUM_GROUPS) if NUM_GROUPS > 0 else []

if DEFAULT_CONFIG:
    reload_config(DEFAULT_CONFIG)


def get_group_color_hex(ng_value) -> str:
    try:
        ng_int = int(ng_value)
    except Exception:
        return '#FFEE00'

    if NUM_GROUPS <= 0:
        return '#FFEE00'

    if GROUP_COLORS_HEX and 0 <= ng_int < len(GROUP_COLORS_HEX):
        return GROUP_COLORS_HEX[ng_int]

    idx = ng_int % NUM_GROUPS
    return _rgb_to_hex(_PALETTE_FALLBACK[idx])


def get_group_name(ng_value) -> str:
    try:
        ng_int = int(ng_value)
    except Exception:
        return f'Grupo {ng_value}'

    if 0 <= ng_int < len(GROUP_NAMES):
        return GROUP_NAMES[ng_int]
    return f'Grupo {ng_int}'



def get_band_names_for_mode(mode: str) -> List[str]:
    """Devuelve la secuencia de bandas a usar según el config activo."""
    mode = (mode or 'Nband_sort').strip()
    if mode == 'Nband_filter':
        band_names = list(_cfg_raw.get('Nband_filter', []))
        if band_names:
            return band_names
    return list(_cfg_raw.get('Nband_sort', BAND_NAMES)) or list(BAND_NAMES)


def get_lambda_for_band_names(band_names: List[str]) -> np.ndarray:
    if not band_names:
        return np.array([], dtype=float)
    if 'Nband' in _cfg_raw and 'lam' in _cfg_raw:
        lam_map = {bn: float(lv) for bn, lv in zip(_cfg_raw.get('Nband', []), _cfg_raw.get('lam', []))}
        return np.array([lam_map.get(b, np.nan) for b in band_names], dtype=float)
    if 'bands' in _cfg_raw:
        lam_map = {b.get('name'): float(b.get('lambda', np.nan)) for b in _cfg_raw.get('bands', [])}
        return np.array([lam_map.get(b, np.nan) for b in band_names], dtype=float)
    return np.array([], dtype=float)


def get_available_band_modes() -> List[str]:
    modes = ['Nband_sort']
    if _cfg_raw.get('Nband_filter'):
        modes.append('Nband_filter')
    return modes


# ---------------------------------------------------------------------------
# Modelos auxiliares
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
    avg_xaxis_mode: str
    show_mcal: bool
    mcal_filter_by_date: bool
    preset: str
    selected_pixels: List[Dict]
    config_path: Optional[str]
    list_path: Optional[str]
    mcal_path: Optional[str]
    mcal_hsl_path: Optional[str]
    current_mcal_key: str


# ---------------------------------------------------------------------------
# Helpers GDAL y datos
# ---------------------------------------------------------------------------

def open_dataset(path_tif: str) -> gdal.Dataset:
    ds = gdal.Open(path_tif, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'GDAL no pudo abrir: {path_tif}')
    return ds


def read_band_sample(ds: gdal.Dataset, band_idx: int, i: int, j: int) -> float:
    band = ds.GetRasterBand(band_idx)
    arr = band.ReadAsArray(j, i, 1, 1)
    return float(arr[0, 0]) if arr is not None else np.nan


def build_rgb(ds: gdal.Dataset, preset_name: str = 'TrueColor') -> np.ndarray:
    if not PRESETS:
        raise RuntimeError('PRESETS no está cargado (falta config_bandas).')

    idxs = PRESETS.get(preset_name, PRESETS.get('TrueColor', []))
    if not idxs:
        raise RuntimeError('No hay preset RGB disponible.')

    h, w = ds.RasterYSize, ds.RasterXSize
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for k, band_idx in enumerate(idxs[:3]):
        band = ds.GetRasterBand(band_idx)
        arr = band.ReadAsArray().astype(np.float32)
        m = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
        m = m if m > 0 else 1.0
        rgb[:, :, k] = np.clip(arr / m, 0, 1)
    return rgb


def get_band_columns_from_df(df: pd.DataFrame) -> List[str]:
    return [b for b in BAND_NAMES if b in df.columns]


def extract_band_matrix(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    cols = get_band_columns_from_df(df)
    if not cols:
        return [], np.empty((0, 0), dtype=float)
    matrix = df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    return cols, matrix


def get_available_band_sequence_names() -> List[str]:
    names = []
    for key in ('Nband_sort', 'Nband_filter'):
        if key in BAND_SEQUENCES and BAND_SEQUENCES[key]:
            names.append(key)

    for key, value in BAND_SEQUENCES.items():
        if key not in names and value:
            names.append(key)

    if not names:
        names = ['Nband_sort']
    return names


def get_band_sequence(seq_name: Optional[str]) -> List[str]:
    if seq_name in BAND_SEQUENCES and BAND_SEQUENCES[seq_name]:
        seq = BAND_SEQUENCES[seq_name]
    elif 'Nband_sort' in BAND_SEQUENCES and BAND_SEQUENCES['Nband_sort']:
        seq = BAND_SEQUENCES['Nband_sort']
    else:
        seq = list(BAND_NAMES)

    return [b for b in seq if b in BAND_NAMES]


def get_lambda_for_sequence(seq_name: Optional[str]) -> np.ndarray:
    seq = get_band_sequence(seq_name)
    idx = [BAND_NAMES.index(b) for b in seq if b in BAND_NAMES]
    return np.array([LAMBDA_NM[i] for i in idx], dtype=float)

def qcolor_to_rgb255(color: QtGui.QColor) -> Tuple[int, int, int]:
    return color.red(), color.green(), color.blue()


# ---------------------------------------------------------------------------
# Canvas de imagen
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
            '', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
            bbox=dict(boxstyle='round', fc='w', ec='0.7'),
            arrowprops=dict(arrowstyle='->')
        )
        self._hover_annot.set_visible(False)
        self._mcal_labels: List[str] = []
        self.img_shape: Optional[Tuple[int, int]] = None

        self.mpl_connect('button_press_event', self._on_click)
        self.mpl_connect('motion_notify_event', self._on_hover)

    def show_image(self, rgb: np.ndarray):
        self.ax.clear()
        self.ax.set_axis_off()
        self._img_artist = self.ax.imshow(rgb, interpolation='nearest')
        self.img_shape = rgb.shape[:2]
        self.draw_idle()

    def show_classmap(self, cm: np.ndarray, palette: Dict[str, str]):
        self.ax.clear()
        self.ax.set_axis_off()
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
        self.clear_mcal()
        self._mcal_labels = list(labels)
        if df.empty:
            self.draw_idle()
            return

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
            linewidths=0.3,
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
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        h, w = self.img_shape
        if 0 <= i < h and 0 <= j < w:
            self.pixelClicked.emit(i, j)

    def _on_hover(self, event):
        if self._mcal_scatter is None or event.inaxes != self.ax:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.draw_idle()
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
        elif self._hover_annot.get_visible():
            self._hover_annot.set_visible(False)
            self.draw_idle()


# ---------------------------------------------------------------------------
# Canvas de firmas
# ---------------------------------------------------------------------------

class SignatureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)
        self.ax.grid(True, alpha=0.25)
        self.x_mode = 'lambda'
        self.band_labels: List[str] = list(BAND_NAMES)

    def set_xmode(self, mode: str):
        self.x_mode = mode
        self.draw_idle()

    def set_band_labels(self, labels: List[str]):
        self.band_labels = list(labels)
        self.draw_idle()

    def clear(self):
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.draw_idle()

    def _apply_axis_style(self):
        if self.x_mode == 'lambda':
            self.ax.set_xlabel('Wavelength λ (nm)')
        else:
            self.ax.set_xlabel('Band')
            self.ax.set_xticks(np.arange(len(self.band_labels)))
            self.ax.set_xticklabels(self.band_labels, rotation=45, ha='right')
        self.ax.set_ylabel('Reflectance (scaled 0–10000)')
        self.ax.set_ylim([0, 10000])

    def plot_signature(
        self,
        xvals: np.ndarray,
        yvals: np.ndarray,
        label: str,
        std_upper: Optional[np.ndarray] = None,
        std_lower: Optional[np.ndarray] = None,
        color: Optional[str] = None,
        marker: str = 'o',
        linestyle: str = '-',
    ):
        if color is not None:
            line = self.ax.plot(
                xvals, yvals,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=label,
                color=color,
            )
            line_color = color
        else:
            line = self.ax.plot(
                xvals, yvals,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4,
                label=label,
            )
            line_color = line[0].get_color()

        if std_upper is not None and std_lower is not None:
            self.ax.fill_between(xvals, std_lower, std_upper, alpha=0.25, color=line_color)

        self.ax.legend(loc='best', fontsize=8)
        self._apply_axis_style()
        self.draw_idle()
        

class SignatureWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, title: str = 'Firmas espectrales', clear_callback=None, pixel_mode: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 560)
        self.clear_callback = clear_callback
        self.pixel_mode = pixel_mode

        self.canvas = SignatureCanvas(self)
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['ID', 'i', 'j', 'Fecha'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.chk_lambda = QtWidgets.QCheckBox('Eje X en λ (nm)')
        self.chk_lambda.setChecked(True)
        self.chk_lambda.stateChanged.connect(self._toggle_xmode)
        
        self.cmb_band_sequence = QtWidgets.QComboBox(self)
        self.reload_band_sequences()
        
        self.cmb_band_mode = QtWidgets.QComboBox()
        self.cmb_band_mode.addItems(get_available_band_modes())
        self.cmb_band_mode.setCurrentText('Nband_sort')

        self.btn_clear = QtWidgets.QPushButton('Limpiar gráfico')
        self.btn_clear.clicked.connect(self._clear_graph)

        self.chk_all_dates = None
        if self.pixel_mode:
            self.chk_all_dates = QtWidgets.QCheckBox('Mostrar todas las fechas')
            self.chk_all_dates.setChecked(False)

        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.canvas, 2)
        hl.addWidget(self.table, 1)
        layout.addLayout(hl)

        seq_layout = QtWidgets.QHBoxLayout()
        seq_layout.addWidget(QtWidgets.QLabel('Secuencia de bandas:'))
        seq_layout.addWidget(self.cmb_band_sequence)
        layout.addLayout(seq_layout)

        layout.addWidget(self.chk_lambda)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.chk_lambda)
        controls.addWidget(QtWidgets.QLabel('Secuencia de bandas:'))
        controls.addWidget(self.cmb_band_mode)
        if self.chk_all_dates is not None:
            controls.addWidget(self.chk_all_dates)
        controls.addStretch(1)
        controls.addWidget(self.btn_clear)
        layout.addLayout(controls)

    def _toggle_xmode(self, st):
        self.canvas.set_xmode('lambda' if st == Qt.Checked else 'band')

    def current_band_mode(self) -> str:
        return self.cmb_band_mode.currentText() or 'Nband_sort'

    def _clear_graph(self):
        self.canvas.clear()
        self.update_table([])
        if callable(self.clear_callback):
            self.clear_callback()

    def update_table(self, rows: List[Tuple[str, int, int, str]]):
        self.table.setRowCount(len(rows))
        for r, (pid, i, j, fecha) in enumerate(rows):
            for c, val in enumerate([pid, i, j, fecha]):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))

    def reload_band_sequences(self):
        current = self.cmb_band_sequence.currentText() if hasattr(self, 'cmb_band_sequence') else ''
        names = get_available_band_sequence_names()

        self.cmb_band_sequence.blockSignals(True)
        self.cmb_band_sequence.clear()
        self.cmb_band_sequence.addItems(names)

        if current in names:
            self.cmb_band_sequence.setCurrentText(current)
        elif 'Nband_sort' in names:
            self.cmb_band_sequence.setCurrentText('Nband_sort')
        elif names:
            self.cmb_band_sequence.setCurrentIndex(0)

        self.cmb_band_sequence.blockSignals(False)

    def get_selected_band_sequence_name(self) -> str:
        text = self.cmb_band_sequence.currentText().strip()
        return text if text else 'Nband_sort'

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']


class AverageSignatureWindow(QtWidgets.QDialog):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle('Firmas promedio MCAL')
        self.resize(1080, 620)

        self.canvas = SignatureCanvas(self)
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.group_list.setUniformItemSizes(True)
        self.group_list.setSpacing(1)
        self.group_list.setStyleSheet('QListWidget::item { padding: 2px 4px; }')
        self.group_list.setMaximumHeight(190)
        self.date_list = QtWidgets.QListWidget()
        self.date_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.date_list.setMaximumHeight(190)

        self.btn_plot = QtWidgets.QPushButton('Plotear firmas promedio')
        self.btn_plot.clicked.connect(self.plot_average_signatures)
        self.btn_clear_plot = QtWidgets.QPushButton('Limpiar gráfico')
        self.btn_clear_plot.clicked.connect(self._clear_plot)
        self.btn_select_all_groups = QtWidgets.QPushButton('Seleccionar todos los grupos')
        self.btn_select_all_groups.clicked.connect(self.select_all_groups)
        self.btn_clear_groups = QtWidgets.QPushButton('Limpiar grupos')
        self.btn_clear_groups.clicked.connect(self.clear_groups)
        self.btn_select_all_dates = QtWidgets.QPushButton('Seleccionar todas las fechas')
        self.btn_select_all_dates.clicked.connect(self.select_all_dates)
        self.btn_clear_dates = QtWidgets.QPushButton('Limpiar fechas')
        self.btn_clear_dates.clicked.connect(self.clear_dates)

        self.stats_label = QtWidgets.QLabel('Selecciona grupos y fechas para graficar.')
        self.chk_lambda = QtWidgets.QCheckBox('Eje X en λ (nm)')
        self.chk_lambda.setChecked(True)
        self.chk_lambda.stateChanged.connect(self._toggle_xmode)
        self.cmb_band_mode = QtWidgets.QComboBox()
        self.cmb_band_mode.addItems(get_available_band_modes())
        self.cmb_band_mode.setCurrentText('Nband_sort')
        
        self.cmb_band_sequence = QtWidgets.QComboBox()
        self.reload_band_sequences()

        self._build_layout()

    def _build_layout(self):
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel('Grupos'))
        left_layout.addWidget(self.group_list)

        group_btns = QtWidgets.QHBoxLayout()
        group_btns.addWidget(self.btn_select_all_groups)
        group_btns.addWidget(self.btn_clear_groups)
        left_layout.addLayout(group_btns)

        left_layout.addWidget(QtWidgets.QLabel('Fechas'))
        left_layout.addWidget(self.date_list)
        date_btns = QtWidgets.QHBoxLayout()
        date_btns.addWidget(self.btn_select_all_dates)
        date_btns.addWidget(self.btn_clear_dates)
        left_layout.addLayout(date_btns)

        left_layout.addWidget(self.stats_label)
        left_layout.addWidget(self.btn_plot)
        left_layout.addWidget(self.btn_clear_plot)
        left_layout.addWidget(QtWidgets.QLabel('Secuencia de bandas'))
        left_layout.addWidget(self.cmb_band_sequence)
        left_layout.addWidget(self.chk_lambda)
        
        band_row = QtWidgets.QHBoxLayout()
        band_row.addWidget(QtWidgets.QLabel('Secuencia de bandas:'))
        band_row.addWidget(self.cmb_band_mode)
        left_layout.addLayout(band_row)
        left_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(self.canvas)

        layout.addWidget(left_panel, 0)
        layout.addWidget(right_panel, 1)
        
    def reload_band_sequences(self):
        current = self.cmb_band_sequence.currentText() if hasattr(self, 'cmb_band_sequence') else ''
        names = get_available_band_sequence_names()

        self.cmb_band_sequence.blockSignals(True)
        self.cmb_band_sequence.clear()
        self.cmb_band_sequence.addItems(names)

        if current in names:
            self.cmb_band_sequence.setCurrentText(current)
        elif 'Nband_sort' in names:
            self.cmb_band_sequence.setCurrentText('Nband_sort')
        elif names:
            self.cmb_band_sequence.setCurrentIndex(0)

        self.cmb_band_sequence.blockSignals(False)

    def get_selected_band_sequence_name(self) -> str:
        text = self.cmb_band_sequence.currentText().strip()
        return text if text else 'Nband_sort'

    def _toggle_xmode(self, st):
        self.canvas.set_xmode('lambda' if st == Qt.Checked else 'band')

    def _clear_plot(self):
        self.canvas.clear()

    def current_band_mode(self) -> str:
        return self.cmb_band_mode.currentText() or 'Nband_sort'

    def load_groups_and_dates(self):
        df = self.main_window._current_mcal_complete()
        self.group_list.clear()
        self.date_list.clear()

        if df.empty:
            self.stats_label.setText('No hay datos MCAL cargados.')
            return

        if 'Ng' in df.columns:
            unique_ng = sorted(pd.to_numeric(df['Ng'], errors='coerce').dropna().astype(int).unique())
            for ng_val in unique_ng:
                gname = get_group_name(ng_val)
                item = QtWidgets.QListWidgetItem(f'Ng{ng_val} - {gname}')
                item.setData(Qt.UserRole, int(ng_val))
                item.setBackground(QtGui.QColor(get_group_color_hex(ng_val)))
                qtext = QtGui.QColor(get_group_color_hex(ng_val))
                luminance = 0.299 * qtext.red() + 0.587 * qtext.green() + 0.114 * qtext.blue()
                item.setForeground(QtGui.QBrush(Qt.black if luminance > 150 else Qt.white))
                self.group_list.addItem(item)

        if 'Fecha' in df.columns:
            for fecha in sorted(df['Fecha'].astype(str).unique()):
                self.date_list.addItem(fecha)

        self.select_default_items()
        self.stats_label.setText(f'Registros MCAL cargados: {len(df)}')

    def select_default_items(self):
        for i in range(min(3, self.group_list.count())):
            self.group_list.item(i).setSelected(True)
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
        selected_items = self.group_list.selectedItems()
        selected_dates = [item.text() for item in self.date_list.selectedItems()]

        if not selected_items or not selected_dates:
            QtWidgets.QMessageBox.warning(self, 'Advertencia', 'Selecciona al menos un grupo y una fecha.')
            return

        df_mcal = self.main_window._current_mcal_complete()

        band_names = get_band_sequence(self.current_band_mode())
        band_names = [b for b in band_names if b in df_mcal.columns]

        if not band_names:
            QtWidgets.QMessageBox.warning(
                self,
                'Advertencia',
                'El MCAL activo no contiene columnas espectrales compatibles con la secuencia seleccionada.',
            )
            return

        self.canvas.set_band_labels(band_names)
        xvals = get_lambda_for_sequence(self.get_selected_band_sequence_name()) if self.canvas.x_mode == 'lambda' else np.arange(len(band_names))

        for item in selected_items:
            ng_val = item.data(Qt.UserRole)
            if ng_val is None:
                continue
            color_hex = get_group_color_hex(ng_val)
            group_text = item.text()

            for date_idx, date_str in enumerate(selected_dates):
                marker = MARKERS[date_idx % len(MARKERS)]
                mask = (
                    pd.to_numeric(df_mcal['Ng'], errors='coerce') == int(ng_val)
                ) & (
                    df_mcal['Fecha'].astype(str) == str(date_str)
                )
                group_data = df_mcal.loc[mask].copy()
                if group_data.empty:
                    continue

                band_data = group_data.reindex(columns=band_cols).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
                if band_data.size == 0 or np.all(~np.isfinite(band_data)):
                    continue

                mean_vals = np.nanmean(band_data, axis=0)
                std_vals = np.nanstd(band_data, axis=0)
                mean_vals = np.clip(mean_vals, 0, 10000)
                upper_std = np.clip(mean_vals + std_vals, 0, 10000)
                lower_std = np.clip(mean_vals - std_vals, 0, 10000)

                label = f'{group_text} ({date_str}) - n={len(group_data)}'
                self.canvas.plot_signature(
                    xvals,
                    mean_vals,
                    label,
                    std_upper=upper_std,
                    std_lower=lower_std,
                    color=color_hex,
                    marker=marker,
                )
                plots_created += 1

        if plots_created == 0:
            self.stats_label.setText('No se encontraron combinaciones con datos válidos.')
            QtWidgets.QMessageBox.information(self, 'Información', 'No se encontraron datos para las combinaciones seleccionadas.')
        else:
            self.stats_label.setText(f'Se crearon {plots_created} firmas promedio.')


class McalSignatureWindow(QtWidgets.QDialog):
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle('Firmas del MCAL activo')
        self.resize(1120, 660)

        self.canvas = SignatureCanvas(self)
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.group_list.setMaximumHeight(190)
        self.date_list = QtWidgets.QListWidget()
        self.date_list.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
        self.date_list.setMaximumHeight(190)
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Grupo', 'i', 'j', 'Fecha'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(180)

        self.info_label = QtWidgets.QLabel(
            'Este visor muestra firmas espectrales de registros individuales del MCAL activo. '
            'Para evitar que la aplicación se quede pensando, primero filtra por grupos/fechas '
            'y opcionalmente limita la cantidad máxima de registros a graficar.'
        )
        self.info_label.setWordWrap(True)

        self.btn_plot = QtWidgets.QPushButton('Plotear firmas MCAL')
        self.btn_plot.clicked.connect(self.plot_mcal_signatures)
        self.btn_clear_plot = QtWidgets.QPushButton('Limpiar gráfico')
        self.btn_clear_plot.clicked.connect(self._clear_plot)
        self.btn_select_all_groups = QtWidgets.QPushButton('Seleccionar todos los grupos')
        self.btn_select_all_groups.clicked.connect(self.select_all_groups)
        self.btn_clear_groups = QtWidgets.QPushButton('Limpiar grupos')
        self.btn_clear_groups.clicked.connect(self.clear_groups)
        self.btn_select_all_dates = QtWidgets.QPushButton('Seleccionar todas las fechas')
        self.btn_select_all_dates.clicked.connect(self.select_all_dates)
        self.btn_clear_dates = QtWidgets.QPushButton('Limpiar fechas')
        self.btn_clear_dates.clicked.connect(self.clear_dates)

        self.chk_lambda = QtWidgets.QCheckBox('Eje X en λ (nm)')
        self.chk_lambda.setChecked(True)
        self.chk_lambda.stateChanged.connect(self._toggle_xmode)
        self.cmb_band_mode = QtWidgets.QComboBox()
        self.cmb_band_mode.addItems(get_available_band_modes())
        self.cmb_band_mode.setCurrentText('Nband_sort')
        self.spn_max_records = QtWidgets.QSpinBox()
        self.spn_max_records.setRange(1, 5000)
        self.spn_max_records.setValue(150)
        self.spn_max_records.setSingleStep(25)

        self.stats_label = QtWidgets.QLabel('Sin graficar.')
        self._build_layout()

    def _build_layout(self):
        layout = QtWidgets.QHBoxLayout(self)
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(self.info_label)
        left_layout.addWidget(QtWidgets.QLabel('Grupos'))
        left_layout.addWidget(self.group_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_select_all_groups)
        row.addWidget(self.btn_clear_groups)
        left_layout.addLayout(row)
        left_layout.addWidget(QtWidgets.QLabel('Fechas'))
        left_layout.addWidget(self.date_list)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_select_all_dates)
        row.addWidget(self.btn_clear_dates)
        left_layout.addLayout(row)
        left_layout.addWidget(self.chk_lambda)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Secuencia de bandas:'))
        row.addWidget(self.cmb_band_mode)
        left_layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Máx. registros:'))
        row.addWidget(self.spn_max_records)
        left_layout.addLayout(row)
        left_layout.addWidget(self.btn_plot)
        left_layout.addWidget(self.btn_clear_plot)
        left_layout.addWidget(self.stats_label)
        left_layout.addWidget(self.table)
        left_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(self.canvas)
        layout.addWidget(left_panel, 0)
        layout.addWidget(right_panel, 1)

    def _toggle_xmode(self, st):
        self.canvas.set_xmode('lambda' if st == Qt.Checked else 'band')

    def _clear_plot(self):
        self.canvas.clear()
        self.table.setRowCount(0)
        self.stats_label.setText('Gráfico limpiado.')

    def current_band_mode(self) -> str:
        return self.cmb_band_mode.currentText() or 'Nband_sort'

    def load_groups_and_dates(self):
        df = self.main_window._current_mcal_complete()
        self.group_list.clear()
        self.date_list.clear()
        if df.empty:
            self.stats_label.setText('No hay datos MCAL cargados.')
            return
        if 'Ng' in df.columns:
            unique_ng = sorted(pd.to_numeric(df['Ng'], errors='coerce').dropna().astype(int).unique())
            for ng_val in unique_ng:
                gname = get_group_name(ng_val)
                item = QtWidgets.QListWidgetItem(f'Ng{ng_val} - {gname}')
                item.setData(Qt.UserRole, int(ng_val))
                item.setBackground(QtGui.QColor(get_group_color_hex(ng_val)))
                qtext = QtGui.QColor(get_group_color_hex(ng_val))
                luminance = 0.299 * qtext.red() + 0.587 * qtext.green() + 0.114 * qtext.blue()
                item.setForeground(QtGui.QBrush(Qt.black if luminance > 150 else Qt.white))
                self.group_list.addItem(item)
        if 'Fecha' in df.columns:
            for fecha in sorted(df['Fecha'].astype(str).unique()):
                self.date_list.addItem(fecha)
        for i in range(min(2, self.group_list.count())):
            self.group_list.item(i).setSelected(True)
        self.select_all_dates()
        self.stats_label.setText(f'Registros MCAL cargados: {len(df)}')

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

    def plot_mcal_signatures(self):
        selected_items = self.group_list.selectedItems()
        selected_dates = [item.text() for item in self.date_list.selectedItems()]
        if not selected_items or not selected_dates:
            QtWidgets.QMessageBox.warning(self, 'Advertencia', 'Selecciona al menos un grupo y una fecha.')
            return
        df_mcal = self.main_window._current_mcal_complete()
        band_names = get_band_names_for_mode(self.current_band_mode())
        if not band_names or not all(b in df_mcal.columns for b in band_names):
            QtWidgets.QMessageBox.warning(self, 'Advertencia', 'El MCAL activo no contiene las columnas espectrales requeridas para la secuencia seleccionada.')
            return
        selected_ng = {int(item.data(Qt.UserRole)) for item in selected_items if item.data(Qt.UserRole) is not None}
        mask = pd.to_numeric(df_mcal['Ng'], errors='coerce').isin(selected_ng) & df_mcal['Fecha'].astype(str).isin(selected_dates)
        filtered = df_mcal.loc[mask].copy()
        if filtered.empty:
            QtWidgets.QMessageBox.information(self, 'Información', 'No hay registros MCAL para la combinación seleccionada.')
            return
        max_records = int(self.spn_max_records.value())
        if len(filtered) > max_records:
            filtered = filtered.head(max_records).copy()
        lambda_vals = get_lambda_for_band_names(band_names)
        self.canvas.set_band_context(band_names, lambda_vals)
        self.canvas.clear()
        self.table.setRowCount(0)
        xvals = lambda_vals if self.canvas.x_mode == 'lambda' else np.arange(len(band_names))
        rows = []
        plotted = 0
        for _, row in filtered.iterrows():
            vals = pd.to_numeric(row.reindex(band_names), errors='coerce').to_numpy(dtype=float)
            if np.all(~np.isfinite(vals)):
                continue
            vals = np.clip(vals, 0, 10000)
            ng = int(pd.to_numeric(row.get('Ng', np.nan), errors='coerce')) if pd.notna(pd.to_numeric(row.get('Ng', np.nan), errors='coerce')) else None
            color_hex = get_group_color_hex(ng) if ng is not None else None
            gname = get_group_name(ng) if ng is not None else 'Sin grupo'
            fecha = str(row.get('Fecha', ''))
            label = f'Ng{ng} - {gname} ({fecha}) [{plotted+1}]'
            self.canvas.plot_signature(xvals, vals, label, color=color_hex, marker='o', linestyle='-')
            rows.append((f'Ng{ng} - {gname}', row.get('i', ''), row.get('j', ''), fecha))
            plotted += 1
        self.table.setRowCount(len(rows))
        for r, vals in enumerate(rows):
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(val)))
        self.stats_label.setText(f'Firmas graficadas: {plotted} de {len(filtered)} registros filtrados.')


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROI Spectral Signature v4 (PyQt5)')
        self.resize(1380, 840)

        self.config_path: Optional[str] = DEFAULT_CONFIG
        self.list_path: Optional[str] = DEFAULT_LIST
        self.mcal_path: Optional[str] = DEFAULT_MCAL
        self.mcal_hsl_path: Optional[str] = DEFAULT_MCAL_HSL
        self.current_mcal_key: str = 'Mcal_py.csv'

        self.df_list = pd.DataFrame()
        self.df_mcal = pd.DataFrame()
        self.df_mcal_hsl = pd.DataFrame()
        self.current_mcal_df = pd.DataFrame()

        self.active_date: Optional[str] = None
        self.active_path: Optional[str] = None
        self.ds: Optional[gdal.Dataset] = None
        self.is_classmap: bool = False
        self.show_mcal: bool = True
        self.mcal_filter_by_date: bool = True
        self.selected_pixels: List[PixelSel] = []

        self.sig_win = SignatureWindow(self, 'Firmas espectrales por píxel', clear_callback=self._clear_selected_pixels, pixel_mode=True)
        self.mcal_sig_win = McalSignatureWindow(self, self)
        self.avg_sig_win = AverageSignatureWindow(self, self)

        self._build_ui()
        self._build_menus()
        self._connect_aux_windows()
        self._load_defaults_if_any()

    def _connect_aux_windows(self):
        self.sig_win.chk_lambda.stateChanged.connect(lambda _=None: self._update_signatures_view())
        self.sig_win.cmb_band_mode.currentTextChanged.connect(lambda _=None: self._update_signatures_view())
        if self.sig_win.chk_all_dates is not None:
            self.sig_win.chk_all_dates.stateChanged.connect(lambda _=None: self._update_signatures_view())
        self.avg_sig_win.chk_lambda.stateChanged.connect(lambda _=None: self.avg_sig_win.plot_average_signatures() if self.avg_sig_win.isVisible() and self.avg_sig_win.canvas.ax.lines else None)
        self.avg_sig_win.cmb_band_mode.currentTextChanged.connect(lambda _=None: self.avg_sig_win.plot_average_signatures() if self.avg_sig_win.isVisible() and self.avg_sig_win.canvas.ax.lines else None)
        self.mcal_sig_win.chk_lambda.stateChanged.connect(lambda _=None: None)
        self.mcal_sig_win.cmb_band_mode.currentTextChanged.connect(lambda _=None: None)

    # ------------------------- carga inicial -------------------------
    def _load_defaults_if_any(self):
        if self.config_path and os.path.exists(self.config_path):
            try:
                reload_config(self.config_path)
            except Exception as exc:
                self._warn(f'No pude cargar config por defecto: {exc}')

        self._update_labels()

        if self.list_path and os.path.exists(self.list_path):
            try:
                self.df_list = read_csv_file(self.list_path, '#')
                self._validate_list_soft()
                self._refresh_dates_list()
            except Exception as exc:
                self._warn(f'No pude cargar ROI list por defecto: {exc}')

        if self.mcal_path and os.path.exists(self.mcal_path):
            try:
                self.df_mcal = read_csv_file(self.mcal_path, '#')
            except Exception as exc:
                self._warn(f'No pude cargar MCAL por defecto: {exc}')

        if self.mcal_hsl_path and os.path.exists(self.mcal_hsl_path):
            try:
                self.df_mcal_hsl = read_csv_file(self.mcal_hsl_path, '#')
            except Exception as exc:
                self._warn(f'No pude cargar MCAL_HSL por defecto: {exc}')

        if not self.df_mcal.empty:
            self._set_mcal_source('Mcal_py.csv')
        elif not self.df_mcal_hsl.empty:
            self._set_mcal_source('McalHSL_mod')
        else:
            self.current_mcal_df = pd.DataFrame()

        if self.list_dates.count() > 0:
            self.list_dates.setCurrentRow(0)

    def _validate_list_soft(self):
        if self.df_list is None or self.df_list.empty:
            self.statusBar().showMessage('ROI list vacío. Carga un 04-ROI-MOD.csv')
            return

        if 'Ruta' not in self.df_list.columns or 'Fecha' not in self.df_list.columns:
            raise RuntimeError("ROI list debe tener columnas 'Fecha' y 'Ruta'")

        ok = self.df_list['Ruta'].apply(lambda p: isinstance(p, str) and os.path.exists(p))
        n_bad = int((~ok).sum())
        if n_bad > 0:
            self.statusBar().showMessage(f'Advertencia: {n_bad} rutas no existen; se omiten')
        self.df_list = self.df_list[ok].copy()

    # ------------------------- UI -------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, central)

        left = QtWidgets.QWidget()
        vleft = QtWidgets.QVBoxLayout(left)

        self.lbl_config = QtWidgets.QLabel('Config: (no cargado)')
        self.lbl_list = QtWidgets.QLabel('ROI list: (no cargado)')
        self.lbl_mcal = QtWidgets.QLabel('MCAL: (no cargado)')

        self.list_dates = QtWidgets.QListWidget()
        self.list_dates.currentTextChanged.connect(self._on_date_changed)

        self.cmb_preset = QtWidgets.QComboBox()
        self.cmb_preset.currentTextChanged.connect(self._redraw_image)

        self.chk_mcal = QtWidgets.QCheckBox('Mostrar Mcal')
        self.chk_mcal.setChecked(True)
        self.chk_mcal.stateChanged.connect(self._toggle_mcal)

        self.chk_mcal_filter = QtWidgets.QCheckBox('Filtrar Mcal por fecha activa')
        self.chk_mcal_filter.setChecked(True)
        self.chk_mcal_filter.stateChanged.connect(self._toggle_mcal_filter)

        vleft.addWidget(self.lbl_config)
        vleft.addWidget(self.lbl_list)
        vleft.addWidget(self.lbl_mcal)
        vleft.addSpacing(10)
        vleft.addWidget(QtWidgets.QLabel('Fechas'))
        vleft.addWidget(self.list_dates)
        vleft.addWidget(QtWidgets.QLabel('Preset RGB'))
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

        self.btn_open_sigs = QtWidgets.QPushButton('Abrir firmas por píxel')
        self.btn_open_sigs.clicked.connect(self._open_sig_window)
        self.btn_open_mcal_sigs = QtWidgets.QPushButton('Abrir firmas del MCAL')
        self.btn_open_mcal_sigs.clicked.connect(self._open_mcal_sig_window)
        self.btn_open_avg_sigs = QtWidgets.QPushButton('Abrir firmas promedio')
        self.btn_open_avg_sigs.clicked.connect(self._open_avg_sig_window)
        self.btn_clear_pixels = QtWidgets.QPushButton('Limpiar píxeles seleccionados')
        self.btn_clear_pixels.clicked.connect(self._clear_selected_pixels)
        self.btn_export_csv = QtWidgets.QPushButton('Exportar firmas sesión a CSV')
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_save_png = QtWidgets.QPushButton('Guardar PNG (vista actual)')
        self.btn_save_png.clicked.connect(self._save_png_current)
        self.btn_save_all = QtWidgets.QPushButton('Guardar todas las vistas PNG')
        self.btn_save_all.clicked.connect(self._save_all_images)
        self.btn_save_proj = QtWidgets.QPushButton('Guardar proyecto (*.json)')
        self.btn_save_proj.clicked.connect(self._save_project)
        self.btn_load_proj = QtWidgets.QPushButton('Cargar proyecto (*.json)')
        self.btn_load_proj.clicked.connect(self._load_project)

        for widget in [
            self.btn_open_sigs,
            self.btn_open_mcal_sigs,
            self.btn_open_avg_sigs,
            self.btn_clear_pixels,
            self.btn_export_csv,
            self.btn_save_png,
            self.btn_save_all,
            self.btn_save_proj,
            self.btn_load_proj,
        ]:
            vright.addWidget(widget)
        vright.addStretch(1)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([280, 860, 260])

        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(splitter)

        self.status = self.statusBar()

    def _build_menus(self):
        bar = self.menuBar()

        m_file = bar.addMenu('Archivo')
        act_exit = QtWidgets.QAction('Salir', self)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        m_data = bar.addMenu('Datos')

        act_cfg = QtWidgets.QAction('Cargar config_bandas (JSON)...', self)
        act_cfg.triggered.connect(self._load_config_from_file)
        m_data.addAction(act_cfg)

        m_mcal = m_data.addMenu('Seleccionar MCAL')
        act_mcal_std = QtWidgets.QAction('Usar Mcal_py.csv', self)
        act_mcal_std.triggered.connect(lambda: self._set_mcal_source('Mcal_py.csv'))
        act_mcal_hsl = QtWidgets.QAction('Usar McalHSL_mod', self)
        act_mcal_hsl.triggered.connect(lambda: self._set_mcal_source('McalHSL_mod'))
        act_mcal_custom = QtWidgets.QAction('Cargar MCAL desde archivo...', self)
        act_mcal_custom.triggered.connect(self._load_mcal_from_file)
        m_mcal.addActions([act_mcal_std, act_mcal_hsl, act_mcal_custom])

        m_list = m_data.addMenu('Seleccionar lista de imágenes')
        act_list_custom = QtWidgets.QAction('Cargar lista CSV...', self)
        act_list_custom.triggered.connect(self._load_list_from_file)
        m_list.addAction(act_list_custom)

        m_view = bar.addMenu('Ver')
        act_view_sig = QtWidgets.QAction('Firmas por píxel', self)
        act_view_sig.triggered.connect(self._open_sig_window)
        act_view_mcal_sig = QtWidgets.QAction('Firmas del MCAL', self)
        act_view_mcal_sig.triggered.connect(self._open_mcal_sig_window)
        act_view_avg = QtWidgets.QAction('Firmas promedio (MCAL)', self)
        act_view_avg.triggered.connect(self._open_avg_sig_window)
        m_view.addActions([act_view_sig, act_view_mcal_sig, act_view_avg])

    # ------------------------- utilidades UI -------------------------
    def _warn(self, msg: str):
        QtWidgets.QMessageBox.warning(self, 'Aviso', msg)

    def _update_labels(self):
        self.lbl_config.setText(f"Config: {os.path.basename(self.config_path) if self.config_path else '(no cargado)'}")
        self.lbl_list.setText(f"ROI list: {os.path.basename(self.list_path) if self.list_path else '(no cargado)'}")
        current_name = None
        if self.current_mcal_key == 'Mcal_py.csv' and self.mcal_path:
            current_name = os.path.basename(self.mcal_path)
        elif self.current_mcal_key == 'McalHSL_mod' and self.mcal_hsl_path:
            current_name = os.path.basename(self.mcal_hsl_path)
        self.lbl_mcal.setText(f"MCAL: {current_name if current_name else '(no cargado)'}")

        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        self.cmb_preset.addItems(list(PRESETS.keys()) if PRESETS else [])
        if PRESETS and 'TrueColor' in PRESETS:
            self.cmb_preset.setCurrentText('TrueColor')
        elif self.cmb_preset.count() > 0:
            self.cmb_preset.setCurrentIndex(0)
        self.cmb_preset.blockSignals(False)

    def _refresh_dates_list(self):
        self.list_dates.clear()
        if self.df_list is None or self.df_list.empty or 'Fecha' not in self.df_list.columns:
            return
        for fecha in self.df_list['Fecha'].astype(str).unique():
            self.list_dates.addItem(fecha)

    # ------------------------- loaders -------------------------
    def _load_config_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Seleccionar config_bandas', '', 'JSON (*.json)')
        if not fn:
            return
        try:
            reload_config(fn)
            self.config_path = fn
            self._update_labels()
            self.sig_win.reload_band_sequences()
            self.mcal_sig_win.reload_band_sequences()
            self.avg_sig_win.reload_band_sequences()
            self._redraw_image()
            self._plot_mcal()
            self.status.showMessage(f'Config cargado: {fn}')
        except Exception as exc:
            self._warn(f'Error al cargar config: {exc}\n\n{traceback.format_exc()}')

    def _load_mcal_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Seleccionar MCAL CSV', '', 'CSV (*.csv)')
        if not fn:
            return
        try:
            df = read_csv_file(fn, '#')
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
            self.status.showMessage(f'MCAL cargado: {fn}')
        except Exception as exc:
            self._warn(f'Error al cargar MCAL: {exc}\n\n{traceback.format_exc()}')

    def _load_list_from_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Seleccionar lista de imágenes CSV', '', 'CSV (*.csv)')
        if not fn:
            return
        try:
            self.df_list = read_csv_file(fn, '#')
            self.list_path = fn
            self._validate_list_soft()
            self._refresh_dates_list()
            self._update_labels()
            self.status.showMessage(f'Lista cargada: {fn}')
            if self.list_dates.count() > 0:
                self.list_dates.setCurrentRow(0)
        except Exception as exc:
            self._warn(f'Error al cargar lista: {exc}\n\n{traceback.format_exc()}')

    def _set_mcal_source(self, key: str):
        if key == 'Mcal_py.csv':
            self.current_mcal_df = self.df_mcal.copy()
        elif key == 'McalHSL_mod':
            self.current_mcal_df = self.df_mcal_hsl.copy()
        else:
            self.current_mcal_df = self.df_mcal.copy()
            key = 'Mcal_py.csv'
        self.current_mcal_key = key
        self._update_labels()
        self._plot_mcal()

    # ------------------------- eventos de vista -------------------------
    def _on_date_changed(self, fecha: str):
        self.active_date = fecha
        if self.df_list is None or self.df_list.empty:
            return
        subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
        if subset.empty:
            self.status.showMessage(f'Sin ruta para fecha {fecha}')
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
            self.is_classmap = ('CLASSMAP' in os.path.basename(self.active_path).upper())
            self._draw_image_or_classmap()
            self._plot_mcal()
            h, w = self.ds.RasterYSize, self.ds.RasterXSize
            kind = 'CLASSMAP' if self.is_classmap else 'multibanda'
            self.status.showMessage(f'{os.path.basename(self.active_path)} | {w}x{h} | {kind}')
        except Exception as exc:
            self.status.showMessage(f'Error al cargar: {exc}')
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
        return self.current_mcal_df.copy() if hasattr(self, 'current_mcal_df') else self.df_mcal.copy()

    def _current_mcal_filtered(self) -> pd.DataFrame:
        df = self._current_mcal_complete()
        if self.mcal_filter_by_date and self.active_date is not None and 'Fecha' in df.columns:
            df = df[df['Fecha'].astype(str) == str(self.active_date)]
        return df

    def _plot_mcal(self):
        if not self.show_mcal or self.ds is None:
            self.img_canvas.clear_mcal()
            return
        df = self._current_mcal_filtered()
        if df is None or df.empty:
            self.img_canvas.clear_mcal()
            return
        required = {'i', 'j'}
        if not required.issubset(set(df.columns)):
            self.img_canvas.clear_mcal()
            self.status.showMessage('El MCAL activo no contiene columnas i/j para dibujar puntos.')
            return
        labels = []
        for _, row in df.iterrows():
            ng = row.get('Ng', '')
            gname = get_group_name(ng)
            fecha = row.get('Fecha', '')
            labels.append(f'Ng{ng} - {gname} | {fecha}')
        self.img_canvas.set_mcal_points(df, labels)

    # ------------------------- firmas por píxel -------------------------
    def _on_pixel_clicked(self, i: int, j: int):
        pid = f'P{len(self.selected_pixels) + 1}'
        self.selected_pixels.append(PixelSel(pid, i, j))
        self._update_signatures_view()

    def _clear_selected_pixels(self):
        self.selected_pixels = []
        if hasattr(self, 'sig_win'):
            self.sig_win.canvas.clear()
            self.sig_win.update_table([])
        self.status.showMessage('Selección de píxeles limpiada.')

    def _get_signature(self, ds: gdal.Dataset, i: int, j: int) -> np.ndarray:
        nb = ds.RasterCount
        vals = np.zeros(nb, dtype=float)
        for b in range(1, nb + 1):
            vals[b - 1] = read_band_sample(ds, b, i, j)
        return np.clip(vals, 0, 10000)

    def _update_signatures_view(self):
        if self.df_list is None or self.df_list.empty:
            return

        rows: List[Tuple[str, int, int, str]] = []
        self.sig_win.canvas.clear()

        band_cols = get_band_sequence(self.sig_win.get_selected_band_sequence_name())
        self.sig_win.canvas.set_band_labels(band_cols)

        xvals = get_lambda_for_sequence(self.sig_win.get_selected_band_sequence_name()) \
            if self.sig_win.canvas.x_mode == 'lambda' else np.arange(len(band_cols))

        fechas = list(self.df_list['Fecha'].astype(str).unique())

        for px in self.selected_pixels:
            for fecha in fechas:
                subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
                if subset.empty:
                    continue
                ruta = subset.iloc[0]['Ruta']
                try:
                    ds = open_dataset(ruta)
                    vals_full = self._get_signature(ds, px.i, px.j)
                    vals = np.array([vals_full[BAND_NAMES.index(b)] for b in band_cols], dtype=float)
                except Exception:
                    continue

                label = f'{px.pid} ({px.i},{px.j}) - {fecha}'
                self.sig_win.canvas.plot_signature(xvals, vals, label)
                rows.append((px.pid, px.i, px.j, fecha))

        self.sig_win.update_table(rows)

    def _open_sig_window(self):
        if not self.sig_win.isVisible():
            self.sig_win.show()
        self._update_signatures_view()
        self.sig_win.raise_()

    # ------------------------- firmas MCAL -------------------------
    def _open_mcal_sig_window(self):
        df_mcal = self._current_mcal_filtered()
        if df_mcal.empty:
            self.status.showMessage('No hay datos MCAL para mostrar.')
            return

        band_cols = get_band_sequence(self.mcal_sig_win.get_selected_band_sequence_name())
        band_cols = [b for b in band_cols if b in df_mcal.columns]

        if not band_cols:
            self._warn('El MCAL activo no contiene columnas espectrales compatibles con la secuencia seleccionada.')
            return

        self.mcal_sig_win.canvas.clear()
        self.mcal_sig_win.canvas.set_band_labels(band_cols)

        rows: List[Tuple[str, int, int, str]] = []
        xvals = get_lambda_for_sequence(self.mcal_sig_win.get_selected_band_sequence_name()) \
            if self.mcal_sig_win.canvas.x_mode == 'lambda' else np.arange(len(band_cols))

        for _, row in df_mcal.iterrows():
            vals = pd.to_numeric(row.reindex(band_cols), errors='coerce').to_numpy(dtype=float)
            vals = np.clip(vals, 0, 10000)

            ng = row.get('Ng', '')
            gname = get_group_name(ng)
            fecha = str(row.get('Fecha', ''))
            color_hex = get_group_color_hex(ng)
            label = f'Ng{ng} - {gname} ({fecha})'

            self.mcal_sig_win.canvas.plot_signature(xvals, vals, label, color=color_hex)
            rows.append((f'Ng{ng} - {gname}', row.get('i', ''), row.get('j', ''), fecha))

        self.mcal_sig_win.update_table(rows)
        self.mcal_sig_win.show()
        self.mcal_sig_win.raise_()
        self.status.showMessage(f'Mostrando {len(df_mcal)} firmas del MCAL.')

    def _open_avg_sig_window(self):
        self.avg_sig_win.load_groups_and_dates()
        self.avg_sig_win.show()
        self.avg_sig_win.raise_()

    # ------------------------- exportación -------------------------
    def _export_csv(self):
        if not self.selected_pixels:
            self.status.showMessage('No hay píxeles seleccionados.')
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar firmas CSV', 'firmas_sesion.csv', 'CSV (*.csv)')
        if not fn:
            return

        rows: List[Dict] = []
        fechas = list(self.df_list['Fecha'].astype(str).unique())
        for px in self.selected_pixels:
            for fecha in fechas:
                subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
                if subset.empty:
                    continue
                ruta = subset.iloc[0]['Ruta']
                try:
                    ds = open_dataset(ruta)
                    vals = self._get_signature(ds, px.i, px.j)
                except Exception:
                    continue
                for b_idx, band_name in enumerate(BAND_NAMES):
                    rows.append({
                        'pixel_id': px.pid,
                        'i': px.i,
                        'j': px.j,
                        'Fecha': fecha,
                        'band': band_name,
                        'lambda_nm': float(LAMBDA_NM[b_idx]) if b_idx < len(LAMBDA_NM) else np.nan,
                        'reflectance_scaled': float(vals[b_idx]) if b_idx < len(vals) else np.nan,
                    })
        if rows:
            pd.DataFrame(rows).to_csv(fn, index=False)
            self.status.showMessage(f'Firmas exportadas: {fn}')

    def _save_png_current(self):
        if self.ds is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar PNG', 'rgb_actual.png', 'PNG (*.png)')
        if not fn:
            return
        rgb_img = self._make_current_rgb_img(self.ds, self.is_classmap)
        plt.imsave(fn, rgb_img)
        self.status.showMessage(f'PNG guardado: {fn}')

    def _make_current_rgb_img(self, ds: gdal.Dataset, is_classmap: bool) -> np.ndarray:
        if is_classmap:
            band = ds.GetRasterBand(1)
            cm = band.ReadAsArray()
            maxc = int(np.nanmax(cm)) if np.size(cm) > 0 else 0
            lut = np.zeros((maxc + 1, 3), dtype=np.uint8)
            for k in range(maxc + 1):
                c = QtGui.QColor(CLASSMAP_COLORS.get(str(k), '#000000'))
                lut[k] = [c.red(), c.green(), c.blue()]
            return lut[np.clip(cm, 0, maxc).astype(int)]
        preset = self.cmb_preset.currentText() or 'TrueColor'
        rgb = build_rgb(ds, preset)
        return (rgb * 255).astype(np.uint8)

    def _save_all_images(self):
        if self.df_list.empty:
            self.status.showMessage('No hay lista de imágenes cargada.')
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Seleccionar carpeta destino')
        if not folder:
            return

        for fecha in self.df_list['Fecha'].astype(str).unique():
            subset = self.df_list[self.df_list['Fecha'].astype(str) == fecha]
            if subset.empty:
                continue
            ruta = subset.iloc[0]['Ruta']
            try:
                ds = open_dataset(ruta)
                base = os.path.splitext(os.path.basename(ruta))[0]
                is_classmap = 'CLASSMAP' in base.upper()
                suffix = 'CLASSMAP' if is_classmap else f'RGB_{self.cmb_preset.currentText()}'
                out = os.path.join(folder, f'{base}_{suffix}.png')
                rgb_img = self._make_current_rgb_img(ds, is_classmap)
                plt.imsave(out, rgb_img)
            except Exception as exc:
                print(f'Error guardando {ruta}: {exc}')
        self.status.showMessage('Guardado por lote completado.')

    # ------------------------- proyecto -------------------------
    def _save_project(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar proyecto', 'proyecto.json', 'JSON (*.json)')
        if not fn:
            return
        state = ProjectState(
            active_date=self.active_date,
            xaxis_mode='lambda' if self.sig_win.chk_lambda.isChecked() else 'band',
            avg_xaxis_mode='lambda' if self.avg_sig_win.chk_lambda.isChecked() else 'band',
            show_mcal=self.show_mcal,
            mcal_filter_by_date=self.mcal_filter_by_date,
            preset=self.cmb_preset.currentText(),
            selected_pixels=[asdict(px) for px in self.selected_pixels],
            config_path=self.config_path,
            list_path=self.list_path,
            mcal_path=self.mcal_path,
            mcal_hsl_path=self.mcal_hsl_path,
            current_mcal_key=self.current_mcal_key,
        )
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
        self.status.showMessage(f'Proyecto guardado: {fn}')

    def _load_project(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Cargar proyecto', '', 'JSON (*.json)')
        if not fn:
            return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Rehidratar rutas si existen
            config_path = data.get('config_path')
            list_path = data.get('list_path')
            mcal_path = data.get('mcal_path')
            mcal_hsl_path = data.get('mcal_hsl_path')

            if config_path and os.path.exists(config_path):
                reload_config(config_path)
                self.config_path = config_path
            if list_path and os.path.exists(list_path):
                self.df_list = read_csv_file(list_path, '#')
                self.list_path = list_path
                self._validate_list_soft()
                self._refresh_dates_list()
            if mcal_path and os.path.exists(mcal_path):
                self.df_mcal = read_csv_file(mcal_path, '#')
                self.mcal_path = mcal_path
            if mcal_hsl_path and os.path.exists(mcal_hsl_path):
                self.df_mcal_hsl = read_csv_file(mcal_hsl_path, '#')
                self.mcal_hsl_path = mcal_hsl_path

            self._update_labels()

            current_mcal_key = data.get('current_mcal_key', 'Mcal_py.csv')
            self._set_mcal_source(current_mcal_key)

            self.show_mcal = bool(data.get('show_mcal', True))
            self.chk_mcal.setChecked(self.show_mcal)
            self.mcal_filter_by_date = bool(data.get('mcal_filter_by_date', True))
            self.chk_mcal_filter.setChecked(self.mcal_filter_by_date)

            preset = data.get('preset', 'TrueColor')
            if preset in PRESETS:
                self.cmb_preset.setCurrentText(preset)

            self.selected_pixels = [
                PixelSel(p['pid'], int(p['i']), int(p['j']))
                for p in data.get('selected_pixels', [])
            ]

            sig_mode = data.get('xaxis_mode', 'lambda')
            self.sig_win.chk_lambda.setChecked(sig_mode == 'lambda')
            avg_mode = data.get('avg_xaxis_mode', 'lambda')
            self.avg_sig_win.chk_lambda.setChecked(avg_mode == 'lambda')

            self.active_date = data.get('active_date')
            if self.active_date:
                items = self.list_dates.findItems(self.active_date, Qt.MatchExactly)
                if items:
                    self.list_dates.setCurrentItem(items[0])
                    self._on_date_changed(self.active_date)

            self._update_signatures_view()
            self.status.showMessage(f'Proyecto cargado: {fn}')
        except Exception as exc:
            self._warn(f'Error al cargar proyecto: {exc}\n\n{traceback.format_exc()}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
