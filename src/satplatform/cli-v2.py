#!/usr/bin/env python3
"""
Minimal CLI to:
1) Read a project structure (01-Raw/s2, 02-Processed, 03-Reports)
2) Scan Sentinel-2 files under 01-Raw/s2 and build a catalog CSV
3) Create RGB quicklooks (B04-B03-B02) when possible and save PNGs

Portable: if moved to another computer, only update --project-root or settings.yaml

Optional settings.yaml (put it in the project root):
---------------------------------------------------
project_root: /absolute/path/to/project
raw_data: 01-Raw/s2
processed: 02-Processed
reports: 03-Reports

Dependencies (add to requirements.txt):
--------------------------------------
rasterio
numpy
pandas
PyYAML
Pillow

Usage:
------
python sat_cli.py --project-root /path/to/Project \
    --settings settings.yaml  # optional

Outputs:
--------
- 02-Processed/catalog.csv
- 02-Processed/quicklooks/<scene_id>_RGB.png
"""
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml
import rasterio
from rasterio.enums import Resampling


# ---------------------------
# Settings & Paths
# ---------------------------
@dataclass
class Settings:
    project_root: Path
    raw_data: Path
    processed: Path
    reports: Path

    @staticmethod
    def from_args(project_root: Path, settings_file: Optional[Path]) -> "Settings":
        # Defaults relative to project root
        defaults = {
            "raw_data": "01-Raw/s2",
            "processed": "02-Processed",
            "reports": "03-Reports",
        }
        if settings_file and settings_file.exists():
            data = yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {}
            raw = data.get("raw_data", defaults["raw_data"])
            proc = data.get("processed", defaults["processed"])
            reps = data.get("reports", defaults["reports"])
        else:
            raw, proc, reps = defaults["raw_data"], defaults["processed"], defaults["reports"]

        root = project_root.resolve()
        return Settings(
            project_root=root,
            raw_data=(root / raw).resolve(),
            processed=(root / proc).resolve(),
            reports=(root / reps).resolve(),
        )


# ---------------------------
# Catalog builder
# ---------------------------
BAND_REGEX = re.compile(r"(B(?:0[2-8]|8A|11|12))", re.IGNORECASE)
DATE_REGEX = re.compile(r"(20\d{6}|20\d{2}-\d{2}-\d{2})")
EXTENSIONS = {".tif", ".tiff", ".jp2"}


def _find_band_token(name: str) -> Optional[str]:
    m = BAND_REGEX.search(name)
    return m.group(1).upper() if m else None


def _find_date_token(name: str) -> Optional[str]:
    m = DATE_REGEX.search(name)
    if not m:
        return None
    token = m.group(1)
    if len(token) == 8:
        return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"
    return token


def _scene_id_from_name(stem: str) -> str:
    # Remove band token if present to group triplets
    band = _find_band_token(stem)
    if band:
        stem = stem.replace(band, "").replace("__", "_")
    # Normalize separators
    stem = re.sub(r"[-_.]+", "_", stem)
    return stem.strip("_")


def scan_catalog(raw_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            try:
                with rasterio.open(p) as ds:
                    res_x = abs(ds.transform.a)
                    res_y = abs(ds.transform.e)
                    band = _find_band_token(p.name) or ""
                    date = _find_date_token(p.name)
                    rows.append(
                        {
                            "path": str(p.resolve()),
                            "file": p.name,
                            "stem": p.stem,
                            "scene_id": _scene_id_from_name(p.stem),
                            "band": band,
                            "date": date,
                            "driver": ds.driver,
                            "crs": str(ds.crs) if ds.crs else None,
                            "width": ds.width,
                            "height": ds.height,
                            "count": ds.count,
                            "dtype": ds.dtypes[0],
                            "res_x": res_x,
                            "res_y": res_y,
                        }
                    )
            except Exception as e:
                print(f"[WARN] Could not read {p.name}: {e}")
                continue
    if not rows:
        raise SystemExit(f"No raster files found under: {raw_dir}")

    df = pd.DataFrame(rows)
    # Helpful ordering
    cols = [
        "scene_id",
        "date",
        "band",
        "file",
        "path",
        "driver",
        "crs",
        "width",
        "height",
        "count",
        "dtype",
        "res_x",
        "res_y",
    ]
    df = df[cols].sort_values(["date", "scene_id", "band", "file"], na_position="last")
    return df


# ---------------------------
# Quicklook RGB
# ---------------------------
RGB_ORDER = ["B04", "B03", "B02"]  # R, G, B


def _percentile_stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(arr[np.isfinite(arr)], [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return out


def _read_band_as_float(path: Path) -> Tuple[np.ndarray, rasterio.Affine, str, int, int]:
    with rasterio.open(path) as ds:
        data = ds.read(1).astype("float32")
        return data, ds.transform, str(ds.crs) if ds.crs else None, ds.width, ds.height


def _same_grid(meta_list: List[Tuple[rasterio.Affine, str, int, int]]) -> bool:
    # Compare transform, crs, width, height
    t0, crs0, w0, h0 = meta_list[0]
    for t, crs, w, h in meta_list[1:]:
        if (crs != crs0) or (w != w0) or (h != h0):
            return False
        # For transforms, allow tiny numeric tolerance
        if not np.allclose([t.a, t.b, t.c, t.d, t.e, t.f], [t0.a, t0.b, t0.c, t0.d, t0.e, t0.f], atol=1e-6):
            return False
    return True


def build_quicklook_rgb(df: pd.DataFrame, out_dir: Path, max_per_scene: int = 1) -> List[Path]:
    out_paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("scene_id")
    for scene_id, g in grouped:
        # Find the triplet
        band_paths: Dict[str, Path] = {}
        for b in RGB_ORDER:
            row = g[g["band"].str.upper() == b].head(1)
            if len(row) == 1:
                band_paths[b] = Path(row.iloc[0]["path"])  # type: ignore
        if len(band_paths) != 3:
            print(f"[INFO] Skip scene '{scene_id}' — missing any of {RGB_ORDER}")
            continue

        # Read bands and check grid compatibility
        arrays = []
        metas = []
        try:
            for b in RGB_ORDER:
                arr, transform, crs, w, h = _read_band_as_float(band_paths[b])
                arrays.append(arr)
                metas.append((transform, crs, w, h))
        except Exception as e:
            print(f"[WARN] Failed reading bands for {scene_id}: {e}")
            continue

        if not _same_grid(metas):
            print(f"[INFO] Skip scene '{scene_id}' — band grids differ (no reprojection in minimal CLI)")
            continue

        # Stretch and stack
        stretched = [_percentile_stretch(a) for a in arrays]
        rgb = np.stack(stretched, axis=0)  # (3, H, W)
        rgb8 = (np.transpose(rgb, (1, 2, 0)) * 255).astype("uint8")  # (H, W, 3)

        out_png = (out_dir / f"{scene_id}_RGB.png").resolve()
        try:
            Image.fromarray(rgb8).save(out_png)
            out_paths.append(out_png)
            print(f"[OK] Quicklook -> {out_png}")
        except Exception as e:
            print(f"[WARN] Could not save quicklook for {scene_id}: {e}")

    if not out_paths:
        print("[INFO] No quicklooks generated (check band availability and grid compatibility)")
    return out_paths


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan Sentinel-2 and produce catalog + RGB quicklooks")
    parser.add_argument("--project-root", type=Path, required=True, help="Project root directory")
    parser.add_argument("--settings", type=Path, default=None, help="Optional settings.yaml (in project root)")
    args = parser.parse_args(argv)

    settings = Settings.from_args(args.project_root, args.settings)

    # Ensure folders
    settings.processed.mkdir(parents=True, exist_ok=True)
    (settings.processed / "quicklooks").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root : {settings.project_root}")
    print(f"[INFO] Raw data     : {settings.raw_data}")
    print(f"[INFO] Processed    : {settings.processed}")

    if not settings.raw_data.exists():
        raise SystemExit(f"Raw data folder not found: {settings.raw_data}")

    # 1) Catalog
    print("[STEP] Scanning catalog…")
    catalog = scan_catalog(settings.raw_data)
    out_csv = settings.processed / "catalog.csv"
    catalog.to_csv(out_csv, index=False)
    print(f"[OK] Catalog -> {out_csv} ({len(catalog)} files)")

    # 2) Quicklooks
    print("[STEP] Generating RGB quicklooks (B04-B03-B02)…")
    ql_dir = settings.processed / "quicklooks"
    build_quicklook_rgb(catalog, ql_dir)

    print("[DONE] All tasks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
