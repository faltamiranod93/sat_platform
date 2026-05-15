"""CLI para la plataforma Sentinel-2 (contracts-first).

Esta capa solo:
  - parsea argumentos
  - resuelve Settings
  - delega a services cableados en `composition/di.py`

Subcomandos:
  - classify   pipeline completo a classmap
  - stack      apila bandas en GeoTIFF multibanda
  - hist-norm  normaliza por histograma (1 raster o múltiples bandas)

Ejemplos:
  satplatform classify --date 20250721 \\
      -b B03=./B03.tif -b B04=./B04.tif -b B08=./B08.tif -b B11=./B11.tif \\
      --roi ./roi.geojson --png

  satplatform stack --date 20250721 -b B02=./B02.tif -b B03=./B03.tif -b B04=./B04.tif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

from .composition import di
from .config import Settings, get_settings
from .contracts.core import S2BandName


# ---------------------------------------------------------------------------
# Helpers locales (parseo y I/O auxiliar)
# ---------------------------------------------------------------------------

def _parse_band_args(items: Iterable[str]) -> Dict[S2BandName, str]:
    out: Dict[S2BandName, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError("Formato de banda inválido. Usa -b B03=path.tif")
        k, v = it.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if not k.startswith("B") or len(k) not in (3, 4):
            raise ValueError(f"Nombre de banda inválido: {k}")
        out[k] = v  # type: ignore[index]
    if not out:
        raise ValueError("Debes especificar al menos una banda con -b BXX=path.tif")
    return out


def _load_roi_geojson(path: str | Path) -> Mapping:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
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


def _settings_for(args: argparse.Namespace) -> Settings:
    s = get_settings()
    upd: dict = {}
    if getattr(args, "root", None):
        upd["project_root"] = Path(args.root)
    if getattr(args, "gdalwarp", None):
        upd["gdalwarp_exe"] = Path(args.gdalwarp)
    if upd:
        s = s.model_copy(update=upd)
    return s


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_stack(args: argparse.Namespace) -> int:
    s = _settings_for(args)
    reader = di.build_raster_reader()
    writer = di.build_raster_writer()

    from .contracts.geo import validate_profile_compat, GeoRaster, GeoProfile
    from .contracts.products import BandSet

    band_map = _parse_band_args(args.band)
    rasters: dict = {}
    ref_p: GeoProfile | None = None
    for bname, uri in band_map.items():
        r = reader.read(uri)
        if ref_p is None:
            ref_p = r.profile
        else:
            validate_profile_compat(ref_p, r.profile)
        rasters[bname] = r
    assert ref_p is not None

    res = _resolution_from_profile(ref_p)
    bs = BandSet(resolution_m=res, bands=rasters)
    order = tuple(args.order) if args.order else tuple(band_map.keys())
    stacked = bs.stack(order)

    out_path = Path(args.out) if args.out else s.out_path("stack", date=args.date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(str(out_path), stacked)
    print(str(out_path))
    return 0


def cmd_hist_norm(args: argparse.Namespace) -> int:
    s = _settings_for(args)
    reader = di.build_raster_reader()
    writer = di.build_raster_writer()
    pre = di.build_preprocessing_adapter()

    if args.input and len(args.input) == 1:
        r = reader.read(args.input[0])
        rn = pre.normalize(r)
        out_path = Path(args.out) if args.out else s.out_path("hist_norm", date=args.date)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer.write(str(out_path), rn)
        print(str(out_path))
        return 0

    from .contracts.geo import validate_profile_compat, GeoProfile
    from .contracts.products import BandSet

    band_map = _parse_band_args(args.band)
    rasters: dict = {}
    ref_p: GeoProfile | None = None
    for bname, uri in band_map.items():
        r = reader.read(uri)
        if ref_p is None:
            ref_p = r.profile
        else:
            validate_profile_compat(ref_p, r.profile)
        rasters[bname] = pre.normalize(r)
    assert ref_p is not None

    res = _resolution_from_profile(ref_p)
    bs = BandSet(resolution_m=res, bands=rasters)
    order = tuple(args.order) if args.order else tuple(band_map.keys())
    stacked = bs.stack(order)

    out_path = Path(args.out) if args.out else s.out_path("hist_norm", date=args.date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(str(out_path), stacked)
    print(str(out_path))
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    s = _settings_for(args)
    svc = di.build_classmap_service(s)

    from .services.classmap_service import ClassMapInputs, ClassMapSpec
    from .contracts.geo import CRSRef

    band_map = _parse_band_args(args.band)
    roi_geom = _load_roi_geojson(args.roi) if args.roi else None
    roi_crs: CRSRef | None = s.crs_out_ref() if roi_geom else None

    out_tif = Path(args.out) if args.out else s.out_path("classmap", date=args.date)
    out_png = out_tif.with_suffix(".png") if args.png else None

    inputs = ClassMapInputs(band_uris=band_map, classes=di.resolve_classes(s))
    spec = ClassMapSpec(
        date=args.date,
        normalize=bool(args.normalize),
        roi_geojson=roi_geom,
        roi_crs=roi_crs,
        out_tif=out_tif,
        out_png=out_png,
    )

    result = svc.run(inputs, spec)

    if result.labels_tif:
        print(str(result.labels_tif))
    if result.quicklook_png:
        print(str(result.quicklook_png))

    total = sum(result.counts.values()) or 1
    print("Distribución de clases:")
    for cls in di.resolve_classes(s):
        n = result.counts.get(int(cls.id), 0)
        pct = (n / total) * 100.0
        print(f"  {cls.name:<10} (id={cls.id}): {pct:5.1f}%  ({n:,} px)")
    return 0


def _resolution_from_profile(p) -> int:
    px, py = p.pixel_size()
    gsd = min(abs(px), abs(py))
    return int(min((10, 20, 60), key=lambda r: abs(gsd - r)))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="satplatform",
        description="CLI Sentinel-2 (contracts-first)",
    )
    p.add_argument("--root", help="project_root (sobre-escribe Settings.project_root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stack", help="genera stack multibanda")
    ps.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    ps.add_argument("-b", "--band", action="append", default=[],
                    help="Par banda=path (p.ej., B04=./B04.tif)")
    ps.add_argument("--order", nargs="*", default=None, help="Orden explícito de bandas")
    ps.add_argument("--out", help="ruta de salida explícita (opcional)")
    ps.set_defaults(func=cmd_stack)

    ph = sub.add_parser("hist-norm", help="normaliza banda(s) y opcionalmente stackea")
    ph.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    ph.add_argument("--out", help="ruta de salida explícita (opcional)")
    ph.add_argument("-i", "--input", nargs="*", help="Una sola imagen para normalizar")
    ph.add_argument("-b", "--band", action="append", default=[],
                    help="Par banda=path para stack normalizado")
    ph.add_argument("--order", nargs="*", default=None, help="Orden de bandas si se stackea")
    ph.set_defaults(func=cmd_hist_norm)

    pc = sub.add_parser("classify", help="clasifica por píxel y genera classmap")
    pc.add_argument("--date", required=True, help="YYYYMMDD para rutas de salida")
    pc.add_argument("-b", "--band", action="append", default=[],
                    help="Par banda=path (p.ej., B03=./B03.tif)")
    pc.add_argument("--roi", help="GeoJSON de ROI (Feature/FeatureCollection/Geometry)")
    pc.add_argument("--normalize", action="store_true",
                    help="aplica normalización previa a cada banda")
    pc.add_argument("--out", help="ruta de salida TIFF (si no, usa Settings)")
    pc.add_argument("--png", action="store_true", help="exporta quicklook PNG junto al TIFF")
    pc.add_argument("--gdalwarp", help="ruta a gdalwarp (si no está en PATH)")
    pc.set_defaults(func=cmd_classify)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as ex:
        print(f"[ERROR] {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
