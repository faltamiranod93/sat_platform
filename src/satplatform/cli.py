# src/satplatform/cli.py
from __future__ import annotations

"""
CLI para la plataforma Sentinel-2 (contracts-first, minimal).

Comandos principales:
  - classify: clasifica por píxel a partir de bandas S2 y genera classmap.
  - stack: crea un stack multibanda (para inspección/depuración).
  - hist-norm: normaliza una banda o un stack simple.

Ejemplos rápidos:
  python -m satplatform.cli classify \
      --date 20250721 \
      -b B03=./B03.tif -b B04=./B04.tif -b B08=./B08.tif -b B11=./B11.tif \
      --roi ./data/roi/roi.geojson --png

  python -m satplatform.cli stack --date 20250721 \
      -b B02=./B02.tif -b B03=./B03.tif -b B04=./B04.tif
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

# Contracts / Config
from .config import Settings, get_settings
from .contracts.core import ClassLabel, MacroClass, S2BandName, SceneId, Stage
from .contracts.geo import GeoRaster, GeoProfile, CRSRef, validate_profile_compat
from .contracts.products import BandSet

# Adapters
from .adapters.gdal_raster_reader import GdalRasterReader
from .adapters.gdal_raster_writer import GdalRasterWriter
from .adapters.gdalwarp_cli import GdalWarpClipper
from .adapters.legacy_histnorm_adapter import LegacyHistNormAdapter
from .adapters.legacy_pixelclass_adapter import LegacyPixelClassifier
from .adapters.legacy_classmap_adapter import LegacyClassMapAdapter

# ----------------------
# Utilidades locales
# ----------------------

def _coerce_resolution_from_profile(p: GeoProfile) -> int:
    px, py = p.pixel_size()
    gsd = min(abs(px), abs(py))
    # Cerca de 10, 20 o 60 m
    cand = min((10, 20, 60), key=lambda r: abs(gsd - r))
    return int(cand)


def _parse_band_args(items: Iterable[str]) -> Dict[S2BandName, str]:
    out: Dict[S2BandName, str] = {}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
        else:
            raise ValueError("Formato de banda inválido. Usa -b B03=path.tif")
        if not k.startswith("B") or len(k) not in (3, 4):
            raise ValueError(f"Nombre de banda inválido: {k}")
        out[k] = v
    if not out:
        raise ValueError("Debes especificar al menos una banda con -b BXX=path.tif")
    return out  # type: ignore[return-value]


def _load_roi_geojson(path: str | Path) -> Mapping:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Acepta Feature/FeatureCollection/Geometry, devuelve Geometry
    t = obj.get("type")
    if t == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            raise ValueError("GeoJSON vacío")
        return feats[0]["geometry"]
    if t == "Feature":
        return obj["geometry"]
    if "coordinates" in obj:
        return obj
    raise ValueError("Formato GeoJSON no reconocido para ROI")


def _ensure_default_classes(s: Settings) -> tuple[ClassLabel, ...]:
    if s.classes and len(s.classes):
        return s.classes
    # Fallback sensato
    return (
        ClassLabel(id=1, name="Agua",   macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    )


def _save_png_quicklook(labels: GeoRaster, classes: tuple[ClassLabel, ...], out_path: Path) -> Path:
    # Mapea ids -> RGB
    h, w = labels.data.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    palette = {int(c.id): (c.color.r, c.color.g, c.color.b) for c in classes}
    ids = np.unique(labels.data)
    for i in ids:
        rgb[labels.data == i] = palette.get(int(i), (0, 0, 0))
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(rgb, mode="RGB").save(out_path)
    except Exception:
        # Fallback a matplotlib si PIL no está
        import matplotlib.image as mpimg  # type: ignore
        mpimg.imsave(out_path, rgb)
    return out_path


# ----------------------
# Comandos
# ----------------------

def cmd_stack(args: argparse.Namespace) -> int:
    s = get_settings()
    if args.root:
        s = s.model_copy(update={"project_root": Path(args.root)})
    reader = GdalRasterReader()
    writer = GdalRasterWriter()

    # Carga bandas
    band_map = _parse_band_args(args.band)
    rasters: Dict[S2BandName, GeoRaster] = {}
    ref_p: GeoProfile | None = None

    for bname, uri in band_map.items():
        r = reader.read(uri)
        if ref_p is None:
            ref_p = r.profile
        else:
            validate_profile_compat(ref_p, r.profile)
        rasters[bname] = r

    assert ref_p is not None
    res = _coerce_resolution_from_profile(ref_p)
    bs = BandSet(resolution_m=res, bands=rasters)

    order = tuple(b for b in args.order) if args.order else tuple(band_map.keys())
    stacked = bs.stack(order)  # GeoRaster multibanda

    out_path = s.out_path("stack", date=args.date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(str(out_path), stacked)
    print(str(out_path))
    return 0


def cmd_hist_norm(args: argparse.Namespace) -> int:
    s = get_settings()
    if args.root:
        s = s.model_copy(update={"project_root": Path(args.root)})
    reader = GdalRasterReader()
    writer = GdalRasterWriter()
    pre = LegacyHistNormAdapter()

    # Si viene un solo raster, normaliza y escribe
    if args.input and len(args.input) == 1:
        r = reader.read(args.input[0])
        rn = pre.normalize(r)
        out_path = Path(args.out if args.out else s.out_path("hist_norm", date=args.date))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer.write(str(out_path), rn)
        print(str(out_path))
        return 0

    # Si son varias bandas, normaliza cada una y genera stack
    band_map = _parse_band_args(args.band)
    rasters: Dict[S2BandName, GeoRaster] = {}
    ref_p: GeoProfile | None = None

    for bname, uri in band_map.items():
        r = reader.read(uri)
        if ref_p is None:
            ref_p = r.profile
        else:
            validate_profile_compat(ref_p, r.profile)
        rasters[bname] = pre.normalize(r)

    assert ref_p is not None
    res = _coerce_resolution_from_profile(ref_p)
    bs = BandSet(resolution_m=res, bands=rasters)
    order = tuple(b for b in args.order) if args.order else tuple(band_map.keys())
    stacked = bs.stack(order)

    out_path = Path(args.out if args.out else s.out_path("hist_norm", date=args.date))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(str(out_path), stacked)
    print(str(out_path))
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    # Settings y adapters
    s = get_settings()
    upd: dict = {}
    if args.root:
        upd["project_root"] = Path(args.root)
    if args.gdalwarp:
        upd["gdalwarp_exe"] = Path(args.gdalwarp)
    if upd:
        s = s.model_copy(update=upd)

    reader = GdalRasterReader()
    writer = GdalRasterWriter()
    clipper = GdalWarpClipper(raster_reader=reader, raster_writer=writer, gdalwarp_exe=str(s.gdalwarp_exe) if s.gdalwarp_exe else None)
    pre = LegacyHistNormAdapter()
    classes = _ensure_default_classes(s)
    clf = LegacyPixelClassifier(classes_def=classes)
    cmapper = LegacyClassMapAdapter()

    # Carga bandas
    band_map = _parse_band_args(args.band)

    rasters: Dict[S2BandName, GeoRaster] = {}
    ref_p: GeoProfile | None = None

    # ROI opcional
    roi_geom = None
    if args.roi:
        roi_geom = _load_roi_geojson(args.roi)
        roi_crs = s.crs_out  # asumimos ROI en CRS de salida (ajusta si necesitas reproyección)
    
    for bname, uri in band_map.items():
        r = reader.read(uri)
        if roi_geom is not None:
            r = clipper.clip_raster(r, roi_geom, roi_crs)
        if ref_p is None:
            ref_p = r.profile
        else:
            validate_profile_compat(ref_p, r.profile)
        # Normalización previa si se pide
        rasters[bname] = pre.normalize(r) if args.normalize else r

    assert ref_p is not None
    res = _coerce_resolution_from_profile(ref_p)
    bs = BandSet(resolution_m=res, bands=rasters)

    # Clasifica
    labels = clf.predict(bs)

    # Escribe GeoTIFF classmap (usa patrón de Settings)
    out_tif = Path(args.out if args.out else s.out_path("classmap", date=args.date))
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    writer.write(str(out_tif), labels)

    # Quicklook PNG opcional
    if args.png:
        out_png = out_tif.with_suffix(".png")
        _save_png_quicklook(labels, classes, out_png)
        print(str(out_png))

    print(str(out_tif))
    return 0


# ----------------------
# Parser
# ----------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="satplatform", description="CLI Sentinel-2 minimal (contracts-first)")
    p.add_argument("--root", help="project_root (sobre-escribe Settings.project_root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # stack
    ps = sub.add_parser("stack", help="genera stack multibanda")
    ps.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    ps.add_argument("-b", "--band", action="append", default=[], help="Par banda=path (p.ej., B04=./B04.tif)")
    ps.add_argument("--order", nargs="*", default=None, help="Orden explícito de bandas en el stack")
    ps.set_defaults(func=cmd_stack)

    # hist-norm
    ph = sub.add_parser("hist-norm", help="normaliza banda(s) y opcionalmente stackea")
    ph.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    ph.add_argument("--out", help="ruta de salida explícita (opcional)")
    ph.add_argument("-i", "--input", nargs="*", help="Una sola imagen para normalizar (alternativa a -b)")
    ph.add_argument("-b", "--band", action="append", default=[], help="Par banda=path para stack normalizado")
    ph.add_argument("--order", nargs="*", default=None, help="Orden de bandas si se stackea")
    ph.set_defaults(func=cmd_hist_norm)

    # classify
    pc = sub.add_parser("classify", help="clasifica por píxel y genera classmap")
    pc.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    pc.add_argument("-b", "--band", action="append", default=[], help="Par banda=path (p.ej., B03=./B03.tif)")
    pc.add_argument("--roi", help="GeoJSON de ROI (Feature/FeatureCollection/Geometry)")
    pc.add_argument("--normalize", action="store_true", help="aplica normalización previa a cada banda")
    pc.add_argument("--out", help="ruta de salida TIFF (si no, usa Settings.output_patterns['classmap']")
    pc.add_argument("--png", action="store_true", help="exporta quicklook PNG junto al TIFF")
    pc.add_argument("--gdalwarp", help="ruta a gdalwarp (si no está en PATH)")
    pc.set_defaults(func=cmd_classify)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(bool(args.func(args)))  # 0 si todo bien
    except KeyboardInterrupt:
        return 130
    except Exception as ex:
        print(f"[ERROR] {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
