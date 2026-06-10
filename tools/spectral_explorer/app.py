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
from satplatform.services.spectral_signature_service import SpectralSignatureService
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
    feat_order = list(dict.fromkeys(sigs["feature"]))  # orden canónico
    fig = go.Figure()
    for cid in sel:
        sub = sigs[sigs.class_id == cid].set_index("feature").reindex(feat_order).reset_index()
        fig.add_trace(go.Scatter(
            x=sub["feature"], y=sub["mean"],
            error_y=dict(type="data", array=sub["std"], visible=True),
            mode="lines+markers", name=cname(cid),
            line=dict(color=ccolor(cid)),
        ))
    fig.update_layout(height=520, xaxis_title="feature", yaxis_title="valor (DN / índice·escala)",
                      legend_title="clase")
    st.plotly_chart(fig, use_container_width=True)
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

# ---- Tab 3: visor de escena ----
with tab_scene:
    idx = load_scene_index(root, glob_pat)
    if not idx:
        st.warning("No se encontraron escenas con el glob indicado.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            date_sel = st.selectbox("Fecha (escena)", sorted(idx))
            preset = st.selectbox("Composición RGB", scene_svc.presets())
            show_pts = st.checkbox("Mostrar puntos del GeoJSON", value=True)
        rgb, arrays, profile = load_scene_rgb(idx[date_sel], preset)

        # firma de píxel por clic (si está disponible el componente)
        clicked = None
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            img = rgb.copy()
            pts_here = df[df["Fecha"] == date_sel]
            if show_pts and len(pts_here):
                pts = scene_svc.points_to_pixels(pts_here, profile)
                for _, p in pts.iterrows():
                    ci, ri = int(round(p["col"])), int(round(p["row"]))
                    if 0 <= ri < img.shape[0] and 0 <= ci < img.shape[1]:
                        img[max(0, ri-1):ri+2, max(0, ci-1):ci+2] = (255, 255, 0)
            with c2:
                if show_pts and not len(pts_here):
                    st.caption(f"(sin puntos de ground truth para {date_sel}; solo 2024-01-23 los tiene)")
                clicked = streamlit_image_coordinates(img, key="scene")
        except ModuleNotFoundError:
            with c2:
                st.image(rgb, caption="Instala 'streamlit-image-coordinates' para clic en píxel.")

        if clicked:
            ci, ri = int(round(clicked["x"])), int(round(clicked["y"]))
            vals = {b: float(a[ri, ci]) for b, a in arrays.items()
                    if 0 <= ri < a.shape[0] and 0 <= ci < a.shape[1]}
            if vals:
                st.markdown(f"**Firma del píxel (col={ci}, row={ri})**")
                order = [b for b in DEFAULT_BAND_FILTER if b in vals]
                st.plotly_chart(
                    go.Figure(go.Scatter(x=order, y=[vals[b] for b in order], mode="lines+markers")),
                    use_container_width=True,
                )
