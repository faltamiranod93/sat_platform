
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal radiometric normalization for Sentinel-2 multitemporal stacks
using operationally stable classes (PIF-like classes) defined by Ng labels.

Design choices:
- Correct ONLY physical bands B01..B12.
- Recompute H, S, L AFTER correction.
- Fit per-date, per-band gain/offset against a reference date using class medians.
- Classes are operationally stable, not assumed physically invariant.
- Validation is sample-based, using labeled points from McalHSL_mod_v6-like CSV.

Required inputs:
- ROI list CSV with at least columns: Fecha, Ruta
- Config JSON with Nband_sort, Nband_filter, nameg
- Labeled points CSV with at least columns: Fecha, i, j, Ng
  (band columns may exist, but are ignored for correction; values are re-read from TIFFs)

Outputs:
- corrected GeoTIFF per date
- corrected sample CSV with B01..B12 + H,S,L
- transforms table
- drift and JM separability tables before/after
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_any(path: Path) -> pd.DataFrame:
    # Robust enough for your files. Falls back to standard comma CSV.
    return pd.read_csv(path)


def resolve_tif_paths(roi_list: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    if "Fecha" not in roi_list.columns or "Ruta" not in roi_list.columns:
        raise ValueError("ROI list must contain columns 'Fecha' and 'Ruta'.")

    out = roi_list.copy()

    resolved = []
    for _, row in out.iterrows():
        ruta = str(row["Ruta"])
        p = Path(ruta)

        candidates = []
        if p.exists():
            candidates.append(p)
        candidates.append(input_dir / p.name)
        if "Archivo" in out.columns and pd.notna(row.get("Archivo", np.nan)):
            candidates.append(input_dir / str(row["Archivo"]))

        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c.resolve()
                break

        if chosen is None:
            raise FileNotFoundError(
                f"Could not resolve TIFF for Fecha={row['Fecha']} from Ruta={ruta}"
            )
        resolved.append(str(chosen))

    out["RutaResolved"] = resolved
    return out


def load_tif(path: Path) -> Tuple[np.ndarray, dict]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Could not open TIFF: {path}")

    x = ds.RasterXSize
    y = ds.RasterYSize
    n = ds.RasterCount

    arr = np.zeros((y, x, n), dtype=np.float32)
    band_names = []

    for k in range(n):
        b = ds.GetRasterBand(k + 1)
        arr[:, :, k] = b.ReadAsArray().astype(np.float32)
        desc = b.GetDescription()
        band_names.append(desc if desc else f"B{k+1:02d}")

    meta = {
        "transform": ds.GetGeoTransform(),
        "projection": ds.GetProjection(),
        "width": x,
        "height": y,
        "count": n,
        "band_names": band_names,
    }
    ds = None
    return arr, meta


def save_tif(path: Path, arr: np.ndarray, meta: dict, band_names: List[str], extra_meta: Dict[str, str]) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        meta["width"],
        meta["height"],
        arr.shape[2],
        gdal.GDT_Float32,
    )
    ds.SetGeoTransform(meta["transform"])
    ds.SetProjection(meta["projection"])

    for k in range(arr.shape[2]):
        b = ds.GetRasterBand(k + 1)
        b.WriteArray(arr[:, :, k].astype(np.float32))
        b.SetDescription(band_names[k])

    ds.SetMetadata({str(k): str(v) for k, v in extra_meta.items()})
    ds.FlushCache()
    ds = None


def clip_reflectance(arr: np.ndarray, max_val: float = 10000.0) -> np.ndarray:
    out = arr.copy()
    out = np.clip(out, 0.0, max_val)
    return out


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    # rgb in [0,1], shape (n,3)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin

    l = 0.5 * (cmax + cmin)

    s = np.zeros_like(l)
    mask = delta > 0
    s[mask] = delta[mask] / (1 - np.abs(2 * l[mask] - 1) + 1e-12)

    h = np.zeros_like(l)
    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g)
    mask_b = mask & (cmax == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-12)) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-12)) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-12)) + 4

    h = h * 60.0
    s = s * 100.0
    l = l * 100.0
    return np.column_stack([h, s, l])


def extract_samples_from_image(
    arr: np.ndarray,
    meta: dict,
    samples_df: pd.DataFrame,
    band_names: List[str],
    fecha: str,
) -> pd.DataFrame:
    df = samples_df.loc[samples_df["Fecha"] == fecha, ["Fecha", "i", "j", "Ng"]].copy()
    if df.empty:
        return df

    h, w, n = arr.shape
    ii = df["i"].astype(int).to_numpy()
    jj = df["j"].astype(int).to_numpy()

    valid = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    df = df.loc[valid].copy()
    ii = df["i"].astype(int).to_numpy()
    jj = df["j"].astype(int).to_numpy()

    vals = arr[ii, jj, :]
    for k, name in enumerate(band_names):
        df[name] = vals[:, k]

    return df


def build_hsl_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rgb = out[["B04", "B03", "B02"]].to_numpy(dtype=float) / 10000.0
    hsl = rgb_to_hsl(np.clip(rgb, 0.0, 1.0))
    out["H"] = hsl[:, 0]
    out["S"] = hsl[:, 1]
    out["L"] = hsl[:, 2]
    return out


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    # y ≈ a*x + b
    X = np.column_stack([x, np.ones_like(x)])
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * np.sqrt(w), rcond=None)
    return float(beta[0]), float(beta[1])


def robust_transform_from_class_medians(
    src_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    band: str,
    stable_classes: List[int],
) -> Tuple[float, float, dict]:
    rows = []
    for ng in stable_classes:
        src = src_df.loc[src_df["Ng"] == ng, band].dropna().to_numpy()
        ref = ref_df.loc[ref_df["Ng"] == ng, band].dropna().to_numpy()
        if len(src) == 0 or len(ref) == 0:
            continue
        rows.append({
            "Ng": ng,
            "src_med": float(np.median(src)),
            "ref_med": float(np.median(ref)),
            "w": float(min(len(src), len(ref))),
            "n_src": int(len(src)),
            "n_ref": int(len(ref)),
        })

    info = {"mode": None, "classes_used": None, "detail": rows}

    if len(rows) >= 2:
        x = np.array([r["src_med"] for r in rows], dtype=float)
        y = np.array([r["ref_med"] for r in rows], dtype=float)
        w = np.array([max(r["w"], 1.0) for r in rows], dtype=float)

        gain, offset = weighted_linear_fit(x, y, w)
        info["mode"] = "gain_offset"
        info["classes_used"] = [int(r["Ng"]) for r in rows]
        return gain, offset, info

    if len(rows) == 1:
        # Offset-only fallback
        offset = rows[0]["ref_med"] - rows[0]["src_med"]
        info["mode"] = "offset_only"
        info["classes_used"] = [int(rows[0]["Ng"])]
        return 1.0, float(offset), info

    info["mode"] = "identity"
    info["classes_used"] = []
    return 1.0, 0.0, info


def apply_linear_transform(arr: np.ndarray, gain: float, offset: float) -> np.ndarray:
    return arr * gain + offset


def mahalanobis_distance(mu1: np.ndarray, mu2: np.ndarray, cov: np.ndarray) -> float:
    inv_cov = np.linalg.pinv(cov)
    d = mu1 - mu2
    return float(np.sqrt(max(d.T @ inv_cov @ d, 0.0)))


def bhattacharyya_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    mu1 = np.mean(x1, axis=0)
    mu2 = np.mean(x2, axis=0)
    c1 = np.cov(x1, rowvar=False)
    c2 = np.cov(x2, rowvar=False)
    if np.ndim(c1) == 0:
        c1 = np.array([[float(c1)]])
    if np.ndim(c2) == 0:
        c2 = np.array([[float(c2)]])
    c = 0.5 * (c1 + c2)
    inv_c = np.linalg.pinv(c)
    dmu = mu1 - mu2
    term1 = 0.125 * float(dmu.T @ inv_c @ dmu)

    det_c = np.linalg.det(c)
    det_c1 = np.linalg.det(c1)
    det_c2 = np.linalg.det(c2)
    det_c = max(det_c, 1e-12)
    det_c1 = max(det_c1, 1e-12)
    det_c2 = max(det_c2, 1e-12)
    term2 = 0.5 * math.log(det_c / math.sqrt(det_c1 * det_c2))
    return term1 + term2


def jeffries_matusita(x1: np.ndarray, x2: np.ndarray) -> float:
    bd = bhattacharyya_distance(x1, x2)
    return float(2.0 * (1.0 - math.exp(-bd)))


def class_centroid_rmse(df: pd.DataFrame, features: List[str], reference_date: str) -> pd.DataFrame:
    rows = []
    ref = df.loc[df["Fecha"] == reference_date].copy()

    for ng in sorted(df["Ng"].dropna().unique()):
        ref_ng = ref.loc[ref["Ng"] == ng, features]
        if ref_ng.empty:
            continue
        mu_ref = ref_ng.median(axis=0).to_numpy(dtype=float)

        for fecha, g in df.groupby("Fecha"):
            cur_ng = g.loc[g["Ng"] == ng, features]
            if cur_ng.empty:
                continue
            mu_cur = cur_ng.median(axis=0).to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((mu_cur - mu_ref) ** 2)))
            rows.append({
                "Fecha": fecha,
                "Ng": int(ng),
                "rmse_to_reference": rmse,
                "n_samples": int(len(cur_ng)),
            })

    return pd.DataFrame(rows)


def jm_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for fecha, g in df.groupby("Fecha"):
        classes = sorted(g["Ng"].dropna().unique())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1 = int(classes[i])
                c2 = int(classes[j])
                x1 = g.loc[g["Ng"] == c1, features].to_numpy(dtype=float)
                x2 = g.loc[g["Ng"] == c2, features].to_numpy(dtype=float)
                if len(x1) < 2 or len(x2) < 2:
                    continue
                jm = jeffries_matusita(x1, x2)
                rows.append({
                    "Fecha": fecha,
                    "Ng_1": c1,
                    "Ng_2": c2,
                    "JM": jm,
                    "n1": int(len(x1)),
                    "n2": int(len(x2)),
                })
    return pd.DataFrame(rows)


def summarize_validation(drift_before: pd.DataFrame, drift_after: pd.DataFrame,
                         jm_before: pd.DataFrame, jm_after: pd.DataFrame) -> pd.DataFrame:
    out = []

    if not drift_before.empty and not drift_after.empty:
        a = drift_before["rmse_to_reference"].mean()
        b = drift_after["rmse_to_reference"].mean()
        out.append({
            "metric": "mean_rmse_to_reference",
            "before": float(a),
            "after": float(b),
            "delta_after_minus_before": float(b - a),
            "direction_desired": "lower",
        })

    if not jm_before.empty and not jm_after.empty:
        a = jm_before["JM"].mean()
        b = jm_after["JM"].mean()
        out.append({
            "metric": "mean_JM_separability",
            "before": float(a),
            "after": float(b),
            "delta_after_minus_before": float(b - a),
            "direction_desired": "higher",
        })

    return pd.DataFrame(out)


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--roi-list", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--reference-date", required=True, type=str)
    parser.add_argument("--pif-csv", required=True, type=Path,
                        help="CSV with at least Fecha, i, j, Ng")
    parser.add_argument("--stable-classes", nargs="+", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    out_tif_dir = args.output_dir / "corrected_tif"
    ensure_dir(out_tif_dir)

    config = load_json(args.config_json)
    band_names = list(config["Nband_sort"])
    feature_bands = list(config["Nband_filter"])
    class_names = list(config["nameg"])

    roi_list = read_csv_any(args.roi_list)
    roi_list = resolve_tif_paths(roi_list, args.input_dir)

    samples = read_csv_any(args.pif_csv)
    need_cols = {"Fecha", "i", "j", "Ng"}
    if not need_cols.issubset(samples.columns):
        raise ValueError(f"PIF CSV must contain columns: {sorted(need_cols)}")

    samples["Fecha"] = samples["Fecha"].astype(str)
    roi_list["Fecha"] = roi_list["Fecha"].astype(str)

    if args.reference_date not in set(roi_list["Fecha"]):
        raise ValueError(f"Reference date {args.reference_date} not found in ROI list")

    # Read all original images first
    images = {}
    metas = {}
    extracted_original = []

    for _, row in roi_list.iterrows():
        fecha = row["Fecha"]
        tif_path = Path(row["RutaResolved"])
        arr, meta = load_tif(tif_path)
        arr = clip_reflectance(arr[:, :, :len(band_names)])
        images[fecha] = arr
        metas[fecha] = meta

        ext = extract_samples_from_image(arr, meta, samples, band_names, fecha)
        extracted_original.append(ext)

    samples_original = pd.concat(extracted_original, ignore_index=True)
    samples_original = build_hsl_columns(samples_original)

    ref_samples = samples_original.loc[samples_original["Fecha"] == args.reference_date].copy()
    if ref_samples.empty:
        raise ValueError("No labeled points found on reference date.")

    # Fit transforms per date and band using stable classes
    transforms = []
    corrected_images = {}

    for fecha in roi_list["Fecha"]:
        arr = images[fecha].copy()
        arr_corr = arr.copy()

        if fecha == args.reference_date:
            corrected_images[fecha] = arr_corr
            for band in band_names:
                transforms.append({
                    "Fecha": fecha,
                    "Banda": band,
                    "Gain": 1.0,
                    "Offset": 0.0,
                    "Mode": "reference_identity",
                    "StableClassesUsed": ",".join(map(str, args.stable_classes)),
                })
            continue

        src_samples = samples_original.loc[samples_original["Fecha"] == fecha].copy()

        for k, band in enumerate(band_names):
            gain, offset, info = robust_transform_from_class_medians(
                src_samples, ref_samples, band, args.stable_classes
            )
            arr_corr[:, :, k] = apply_linear_transform(arr[:, :, k], gain, offset)

            transforms.append({
                "Fecha": fecha,
                "Banda": band,
                "Gain": gain,
                "Offset": offset,
                "Mode": info["mode"],
                "StableClassesUsed": ",".join(map(str, info["classes_used"])) if info["classes_used"] else "",
            })

        arr_corr = clip_reflectance(arr_corr)
        corrected_images[fecha] = arr_corr

    transforms_df = pd.DataFrame(transforms)
    transforms_df.to_csv(args.output_dir / "radiometric_transforms.csv", index=False)

    # Save corrected TIFFs
    for _, row in roi_list.iterrows():
        fecha = row["Fecha"]
        arr_corr = corrected_images[fecha]
        tif_out = out_tif_dir / f"{fecha}_temporal_norm.tif"
        save_tif(
            tif_out,
            arr_corr,
            metas[fecha],
            band_names,
            {
                "ReferenceDate": args.reference_date,
                "StableClasses": ",".join(map(str, args.stable_classes)),
                "Method": "class_median_weighted_linear_fit",
            },
        )

    # Extract corrected samples
    extracted_corrected = []
    for fecha in roi_list["Fecha"]:
        ext = extract_samples_from_image(corrected_images[fecha], metas[fecha], samples, band_names, fecha)
        extracted_corrected.append(ext)

    samples_corrected = pd.concat(extracted_corrected, ignore_index=True)
    samples_corrected = build_hsl_columns(samples_corrected)

    # Save sample tables
    samples_original.to_csv(args.output_dir / "eval_samples_original.csv", index=False)
    samples_corrected.to_csv(args.output_dir / "eval_samples_corrected.csv", index=False)
    samples_corrected.to_csv(args.output_dir / "McalHSL_mod_temporal.csv", index=False)

    # Validation
    drift_before = class_centroid_rmse(samples_original, feature_bands, args.reference_date)
    drift_after = class_centroid_rmse(samples_corrected, feature_bands, args.reference_date)
    jm_before = jm_table(samples_original, feature_bands)
    jm_after = jm_table(samples_corrected, feature_bands)

    drift_before.to_csv(args.output_dir / "drift_before.csv", index=False)
    drift_after.to_csv(args.output_dir / "drift_after.csv", index=False)
    jm_before.to_csv(args.output_dir / "jm_before.csv", index=False)
    jm_after.to_csv(args.output_dir / "jm_after.csv", index=False)

    summary = summarize_validation(drift_before, drift_after, jm_before, jm_after)
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)

    # Extra diagnostic: class support by date
    support = (
        samples.groupby(["Fecha", "Ng"])
        .size()
        .reset_index(name="n_points")
        .sort_values(["Fecha", "Ng"])
    )
    support.to_csv(args.output_dir / "class_support_by_date.csv", index=False)

import sys
sys.argv = [
    "s_Sat_temporal_rrn_pif_v3.py",
    "--input-dir", r"C:\Users\felip\Desktop\Msc-UTFSM\Laguna-Seca\02-Space-Facilities\ROI-LIST",
    "--roi-list", r"C:\Users\felip\Desktop\Msc-UTFSM\Laguna-Seca\02-Space-Facilities\03-ROI-LIST.csv",
    "--config-json", r"C:\Users\felip\Desktop\Msc-UTFSM\00_Codigos\Python\config_bandas_v3.json",
    "--reference-date", "2024-01-23",
    "--pif-csv", r"C:\Users\felip\Desktop\Msc-UTFSM\Laguna-Seca\McalHSL_mod_v6_py.csv",
    "--stable-classes", "3", "8", "10",
    "--output-dir", r"C:\Users\felip\Desktop\Msc-UTFSM\Laguna-Seca\02-Space-Facilities\ROI-MOD-v2",
]

if __name__ == "__main__":
    main()
