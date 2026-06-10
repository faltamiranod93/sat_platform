"""Explorador de firmas espectrales (Streamlit) — reemplaza los g_Sat_SpectralSignature_v*.

Reutiliza el pipeline: re-extrae las firmas de las escenas reales (GeoJSON v7 +
extract_at_utm_points, georef corregida al vuelo), calcula bandas+HSL+índices con
FeatureService, y ofrece 3 vistas: firmas de clases, separabilidad y visor de escena.

Lanzar:
    streamlit run tools/spectral_explorer/app.py -- \
        --root <project> --geojson <McalHSL_..._utm.geojson>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from satplatform.adapters.mahalanobis_classifier import DEFAULT_BAND_FILTER
from satplatform.composition import di
from satplatform.services.multiband_loader import load_multiband_bandset
from satplatform.services.scene_view_service import SceneViewService
from satplatform.services.spectral_signature_service import (
    S2_WAVELENGTHS_NM,
    SpectralSignatureService,
)
from satplatform.services.training_set_builder import is_scene_file, scene_index_from_uris

ALL_INDICES = ["NDVI", "NDWI", "MNDWI", "NDBI", "BSI"]


def _cli_defaults():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="")
    p.add_argument("--geojson", default="")
    p.add_argument("--scenes-glob", default="01-Raw/*/S2*MSIL2A*")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


# ---------------------------------------------------------------------------
# Carga (cacheada)
# ---------------------------------------------------------------------------

def _resolve_glob(root: str, glob_pat: str) -> str:
    return glob_pat if Path(glob_pat).is_absolute() else str(Path(root) / glob_pat)


@st.cache_data(show_spinner="Re-extrayendo firmas de las escenas…")
def load_training(root: str, geojson: str, glob_pat: str):
    ts = di.build_training_set(Path(geojson), _resolve_glob(root, glob_pat))
    return ts.df, dict(ts.used_by_date), dict(ts.omitted_by_date)


@st.cache_data(show_spinner=False)
def load_scene_index(root: str, glob_pat: str):
    uris = [u for u in sorted(glob.glob(_resolve_glob(root, glob_pat))) if is_scene_file(u)]
    return scene_index_from_uris(uris)


@st.cache_data(show_spinner=False)
def load_classes(root: str):
    s = di.build_settings(Path(root))
    out = {}
    for c in di.resolve_classes(s):
        out[int(c.id)] = (c.name, "#%02x%02x%02x" % c.color.as_tuple())
    return out


@st.cache_resource(show_spinner=False)
def get_reader():
    return di.build_raster_reader(fix_georef=True)


@st.cache_data(show_spinner="Cargando escena…")
def load_scene_rgb(uri: str, preset: str):
    bandset = load_multiband_bandset(get_reader(), uri)
    rgb = SceneViewService().rgb_composite(bandset, preset)
    arrays = {n: r.data for n, r in bandset.bands.items()}
    profile = next(iter(bandset.bands.values())).profile
    return rgb, arrays, profile


def _settings(root: str):
    # build_settings deja project_root="." (del yaml); lo fijamos al root real.
    return di.build_settings(Path(root)).model_copy(update={"project_root": Path(root).resolve()})


@st.cache_data(show_spinner=False)
def load_classmap_colored(root: str, date_yyyymmdd: str, clf: str, _palette: dict):
    """Lee 03-Products/CLASSMAP/{date}/classmap_{clf}.tif y lo colorea. None si no existe."""
    p = Path(_settings(root).out_path("classmap", date=date_yyyymmdd, classifier=clf))
    if not p.exists():
        return None
    r = get_reader().read(str(p))
    return SceneViewService().colorize_labels(r.data, _palette)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

ARGS = _cli_defaults()
st.set_page_config(layout="wide", page_title="Explorador de firmas espectrales")
st.title("🛰️ Explorador de firmas espectrales — Laguna Seca")

sb = st.sidebar
sb.header("Fuente de datos")
root = sb.text_input("Project root", ARGS.root)
geojson = sb.text_input("GeoJSON v7 (UTM)", ARGS.geojson)
glob_pat = sb.text_input("Scenes glob", ARGS.scenes_glob)
sb.header("Features")
include_hsl = sb.checkbox("Incluir HSL", value=True)
indices = tuple(sb.multiselect("Índices espectrales", ALL_INDICES, default=["NDVI", "NDWI"]))

if not root or not geojson:
    st.info("Indica **Project root** y **GeoJSON v7** en la barra lateral para empezar.")
    st.stop()

try:
    df, used, omitted = load_training(root, geojson, glob_pat)
    classes = load_classes(root)
except Exception as e:  # noqa: BLE001
    st.error(f"Error cargando datos: {e}")
    st.stop()

if df.empty:
    st.warning("No se extrajeron muestras (ninguna fecha del GeoJSON tiene escena disponible).")
    st.stop()

cname = lambda cid: classes.get(int(cid), (str(cid), "#888888"))[0]  # noqa: E731
ccolor = lambda cid: classes.get(int(cid), (str(cid), "#888888"))[1]  # noqa: E731

sb.caption(f"Muestras: {len(df)} | usadas por fecha: {used} | omitidas: {sum(omitted.values())}")

sig_svc = SpectralSignatureService()
scene_svc = SceneViewService()
BAND_FILTER = DEFAULT_BAND_FILTER

tab_sig, tab_sep, tab_scene = st.tabs(["📈 Firmas de clases", "🔀 Separabilidad", "🗺️ Visor de escena"])

# ---- Tab 1: firmas de clases ----
with tab_sig:
    sigs = sig_svc.signatures_by_class(df, BAND_FILTER, include_hsl, indices)
    all_ids = sorted(sigs["class_id"].unique())
    sel = st.multiselect("Clases", all_ids, default=all_ids, format_func=cname)

    # Separar features por escala: bandas (λ) | HSL | índices — cada una en su gráfico.
    band_feats = [b for b in BAND_FILTER if b in set(sigs["feature"])]
    hsl_feats = [f for f in ("H", "S", "L") if f in set(sigs["feature"])]
    index_feats = [f for f in indices if f in set(sigs["feature"])]

    def _signature_fig(feats, x_is_wavelength, y_title, title):
        fig = go.Figure()
        if x_is_wavelength:
            xs = [S2_WAVELENGTHS_NM[b] for b in feats]
        else:
            xs = list(range(len(feats)))
        for cid in sel:
            sub = sigs[sigs.class_id == cid].set_index("feature")
            ys = [sub.loc[f, "mean"] for f in feats]
            es = [sub.loc[f, "std"] for f in feats]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, error_y=dict(type="data", array=es, visible=True),
                mode="lines+markers", name=cname(cid), line=dict(color=ccolor(cid)),
            ))
        if not x_is_wavelength:
            fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=feats)
        fig.update_layout(height=330, title=title,
                          xaxis_title="λ (nm)" if x_is_wavelength else "feature",
                          yaxis_title=y_title, legend_title="clase", margin=dict(t=40, b=10))
        return fig

    # 1) Firma espectral (bandas vs longitud de onda)
    if band_feats:
        st.plotly_chart(_signature_fig(band_feats, True, "reflectancia (DN)",
                                       "Firma espectral — bandas vs longitud de onda"),
                        use_container_width=True)
    # 2) HSL (escala aparte)
    if hsl_feats:
        st.plotly_chart(_signature_fig(hsl_feats, False, "valor HSL (H:0-360, S/L:0-100)",
                                       "HSL por clase"),
                        use_container_width=True)
    # 3) Índices espectrales (escala -1..1)
    if index_feats:
        st.plotly_chart(_signature_fig(index_feats, False, "índice (-1..1)",
                                       "Índices espectrales por clase"),
                        use_container_width=True)
    elif not hsl_feats:
        st.caption("Activa HSL o índices en la barra lateral para ver sus gráficos.")

    with st.expander("Tabla de firmas (media±std)"):
        st.dataframe(sigs, use_container_width=True)

# ---- Tab 2: separabilidad ----
with tab_sep:
    metric = st.radio("Métrica de distancia entre clases", ["euclidean", "mahalanobis"], horizontal=True)
    M = sig_svc.separability_matrix(df, BAND_FILTER, include_hsl, indices, metric=metric)
    labels = [cname(c) for c in M.index]
    fig_h = px.imshow(M.values, x=labels, y=labels, color_continuous_scale="Viridis",
                      labels=dict(color=f"dist ({metric})"), title="Matriz de separabilidad (mayor = más separable)")
    fig_h.update_layout(height=520)
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("Valores bajos fuera de la diagonal = clases que se confunden (poca separabilidad).")

    pca = sig_svc.pca_2d(df, BAND_FILTER, include_hsl, indices)
    pca["clase"] = pca["class_id"].map(cname)
    fig_p = px.scatter(pca, x="pc1", y="pc2", color="clase",
                       color_discrete_map={cname(c): ccolor(c) for c in classes},
                       title="PCA 2D de las muestras", opacity=0.6)
    fig_p.update_layout(height=520)
    st.plotly_chart(fig_p, use_container_width=True)

# ---- Tab 3: visor de escena (real vs clasificaciones) ----
with tab_scene:
    idx = load_scene_index(root, glob_pat)
    if not idx:
        st.warning("No se encontraron escenas con el glob indicado.")
    else:
        col_ctrl, col_real = st.columns([1, 2])
        with col_ctrl:
            # lista de fechas en un recuadro deslizable (scroll), no un radio largo
            st.markdown("**Fecha (escena)**")
            with st.container(height=280, border=True):
                date_sel = st.radio("Fecha", sorted(idx), index=0, label_visibility="collapsed")
            preset = st.selectbox("Composición RGB", scene_svc.presets())
            show_pts = st.checkbox("Puntos GeoJSON", value=True)
        rgb, arrays, profile = load_scene_rgb(idx[date_sel], preset)
        # id -> (r,g,b) desde el hex de class_labels, para colorear los classmaps
        palette = {cid: tuple(bytes.fromhex(hexc[1:])) for cid, (_, hexc) in classes.items()}

        # overlay de puntos sobre la RGB
        img = rgb.copy()
        pts_here = df[df["Fecha"] == date_sel]
        if show_pts and len(pts_here):
            for _, p in scene_svc.points_to_pixels(pts_here, profile).iterrows():
                ci, ri = int(round(p["col"])), int(round(p["row"]))
                if 0 <= ri < img.shape[0] and 0 <= ci < img.shape[1]:
                    img[max(0, ri-1):ri+2, max(0, ci-1):ci+2] = (255, 255, 0)

        # --- fila superior: imagen real (pequeña) con clic en píxel ---
        clicked = None
        with col_real:
            st.markdown(f"**Imagen real — {date_sel} ({preset})**")
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                clicked = streamlit_image_coordinates(img, width=380, key="scene")
                st.caption("Clic en un píxel para ver su firma.")
            except ModuleNotFoundError:
                st.image(img, width=380)
            if show_pts and not len(pts_here):
                st.caption(f"(sin ground truth para {date_sel}; solo 2024-01-23 lo tiene)")

        # --- fila inferior: mosaico de los 3 classmaps ---
        st.markdown("**Clasificaciones** (mismo recorte, paleta de clases)")
        date_ymd = date_sel.replace("-", "")
        cols = st.columns(3)
        for col, key in zip(cols, ("maha", "cos", "euc")):
            cm = load_classmap_colored(root, date_ymd, key, palette)
            with col:
                if cm is not None:
                    st.image(cm, caption=f"classmap_{key}", use_container_width=True)
                else:
                    st.caption(f"classmap_{key}: (no generado para {date_ymd})")

        # leyenda de clases
        with st.expander("Leyenda de clases"):
            for cid in sorted(classes):
                name, hexc = classes[cid]
                st.markdown(f"<span style='color:{hexc}'>■</span> {cid} — {name}",
                            unsafe_allow_html=True)

        # firma del píxel clicado
        if clicked:
            ci, ri = int(round(clicked["x"])), int(round(clicked["y"]))
            vals = {b: float(a[ri, ci]) for b, a in arrays.items()
                    if 0 <= ri < a.shape[0] and 0 <= ci < a.shape[1]}
            if vals:
                st.markdown(f"**Firma del píxel (col={ci}, row={ri})**")
                order = [b for b in DEFAULT_BAND_FILTER if b in vals]
                fig_px = go.Figure(go.Scatter(
                    x=[S2_WAVELENGTHS_NM[b] for b in order],
                    y=[vals[b] for b in order], mode="lines+markers"))
                fig_px.update_layout(height=300, xaxis_title="λ (nm)", yaxis_title="reflectancia (DN)")
                st.plotly_chart(fig_px, use_container_width=True)
