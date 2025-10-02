# =============================
# FILE: src/satplatform/adapters/csv_catalog.py
# =============================
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from satplatform.ports.catalog import CatalogItem, CatalogPort, MosaicItem, ROIItem


# Column maps tolerantes a distintas nomenclaturas
ROI_COLMAP: Dict[str, Tuple[str, ...]] = {
    "roi_id": ("roi_id", "ROI_ID", "roi", "ROI", "id"),
    "name": ("roi_name", "ROI_NAME", "name", "NAME", "label"),
    "geom_path": (
        "roi_geom_path",
        "geom_path",
        "ROI_GEOM",
        "geometry",
        "geom",
        "path_geom",
        "roi_path",
        "roi_shp",
        "roi_geojson",
    ),
    "epsg": ("epsg", "roi_epsg", "EPSG", "srid", "CRS_EPSG"),
}

MOSAIC_COLMAP: Dict[str, Tuple[str, ...]] = {
    "mosaic_id": ("mosaic_id", "MOSAIC_ID", "scene_id", "SCENE_ID", "mosaic"),
    "roi_id": ("roi_id", "ROI_ID", "roi", "ROI"),
    "acq_date": (
        "acq_date",
        "date",
        "DATE",
        "acquisition_date",
        "sensing_date",
        "datetime",
        "time",
    ),
    "product_path": ("product_path", "PRODUCT_PATH", "path", "filepath", "asset_path"),
    "crs": ("crs", "CRS", "epsg", "EPSG", "srid"),
    "res_m": ("res_m", "resolution", "res", "gsd"),
    "sensor": ("sensor", "SENSOR", "platform", "PLATFORM"),
    "cloud_pct": ("cloud_pct", "clouds", "CLOUD_PCT", "cloud_coverage"),
}

ASSET_COLMAP: Dict[str, Tuple[str, ...]] = {
    # Opcional: si el CSV ya trae activos/bandas explícitas
    "asset_path": ("asset_path", "path", "filepath", "file"),
    "band": ("band", "BAND", "s2_band", "S2_BAND"),
}


def _first_present(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _standardize(df: pd.DataFrame, spec: Dict[str, Tuple[str, ...]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    keep_extra: List[str] = []
    for std, cands in spec.items():
        col = _first_present(df, cands)
        if col is None:
            out[std] = pd.Series([None] * len(df))
        else:
            out[std] = df[col]
    # Extras
    for c in df.columns:
        if all(c not in cands for cands in spec.values()):
            keep_extra.append(c)
    if keep_extra:
        out["_extras"] = df[keep_extra].to_dict(orient="records")
    else:
        out["_extras"] = [{} for _ in range(len(df))]
    return out


def _parse_date(val: Any) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # fall-back: pandas to_datetime
    try:
        return pd.to_datetime(s).date()  # type: ignore
    except Exception:
        return None


class CsvCatalog(CatalogPort):
    """Adapter de catálogo que **lee CSVs** pero expone un **CatalogPort**.

    - 03-ROI-LIST-MOSAICS.csv: define mosaicos por ROI (mínimo `mosaic_id`, `roi_id`, `acq_date`, `product_path`).
    - 04-ROI-MOD.csv: define/actualiza la tabla de ROIs (mínimo `roi_id`, `name`, `geom_path`).

    Es tolerante a variaciones de columnas y mantiene extras.
    """

    def __init__(
        self,
        project_root: Path,
        roi_list_path: Optional[Path] = None,
        roi_mod_path: Optional[Path] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.root = project_root.resolve()
        self.encoding = encoding
        # resolución de rutas con heurística de nombre
        self.roi_list_csv = roi_list_path or self._find_csv("03-ROI-LIST-MOSAICS.csv")
        self.roi_mod_csv = roi_mod_path or self._find_csv("04-ROI-MOD.csv")

        if not self.roi_list_csv or not self.roi_list_csv.exists():
            raise FileNotFoundError("No se encontró 03-ROI-LIST-MOSAICS.csv en el proyecto")
        if not self.roi_mod_csv or not self.roi_mod_csv.exists():
            raise FileNotFoundError("No se encontró 04-ROI-MOD.csv en el proyecto")

        self._rois: List[ROIItem] = []
        self._mosaics: List[MosaicItem] = []
        self._assets: List[CatalogItem] = []

        self._load()

    # -------------
    # Infra
    # -------------
    def _find_csv(self, filename: str) -> Optional[Path]:
        # Busca por nombre exacto en el árbol del proyecto
        candidates = list(self.root.rglob(filename))
        return candidates[0] if candidates else None

    def _abspath(self, p: Optional[str | Path]) -> Optional[Path]:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return None
        path = Path(str(p))
        return (self.root / path).resolve() if not path.is_absolute() else path.resolve()

    # -------------
    # Load & normalize
    # -------------
    def _load(self) -> None:
        # ROIs (mod)
        df_roimod_raw = pd.read_csv(self.roi_mod_csv, encoding=self.encoding)
        df_roimod = _standardize(df_roimod_raw, ROI_COLMAP)
        rois: Dict[str, ROIItem] = {}
        for _, r in df_roimod.iterrows():
            roi_id = str(r["roi_id"]).strip() if r["roi_id"] is not None else None
            if not roi_id:
                continue
            roi = ROIItem(
                roi_id=roi_id,
                name=(None if pd.isna(r.get("name")) else str(r.get("name"))),
                geom_path=self._abspath(r.get("geom_path")),
                epsg=int(r["epsg"]) if r.get("epsg") not in (None, "", float("nan")) else None,
                extras=r.get("_extras", {}) or {},
            )
            rois[roi_id] = roi

        # Mosaics (list)
        df_list_raw = pd.read_csv(self.roi_list_csv, encoding=self.encoding)
        df_list = _standardize(df_list_raw, MOSAIC_COLMAP)
        mosaics: List[MosaicItem] = []
        for _, r in df_list.iterrows():
            mosaic_id = r.get("mosaic_id")
            if mosaic_id is None or (isinstance(mosaic_id, float) and pd.isna(mosaic_id)):
                continue
            roi_id = r.get("roi_id")
            acq_date = _parse_date(r.get("acq_date"))
            mosaic = MosaicItem(
                mosaic_id=str(mosaic_id),
                roi_id=None if pd.isna(roi_id) else str(roi_id),
                acq_date=acq_date,
                product_path=self._abspath(r.get("product_path")),
                crs=None if pd.isna(r.get("crs")) else str(r.get("crs")),
                res_m=None if pd.isna(r.get("res_m")) else int(r.get("res_m")),
                sensor=None if pd.isna(r.get("sensor")) else str(r.get("sensor")),
                cloud_pct=None if pd.isna(r.get("cloud_pct")) else float(r.get("cloud_pct")),
                extras=r.get("_extras", {}) or {},
            )
            mosaics.append(mosaic)

        # Activos (opcional) si el CSV lista archivos/bandas
        df_assets = _standardize(df_list_raw, {**MOSAIC_COLMAP, **ASSET_COLMAP})
        assets: List[CatalogItem] = []
        for _, r in df_assets.iterrows():
            asset_path = self._abspath(r.get("asset_path"))
            if not asset_path:
                continue
            roi_id = r.get("roi_id")
            acq_date = _parse_date(r.get("acq_date"))
            mosaic_id = r.get("mosaic_id")
            base = CatalogItem(
                roi=rois.get(str(roi_id)) if roi_id is not None and not pd.isna(roi_id) else None,
                mosaic=MosaicItem(
                    mosaic_id=str(mosaic_id) if mosaic_id is not None and not pd.isna(mosaic_id) else "",
                    roi_id=None if pd.isna(roi_id) else str(roi_id),
                    acq_date=acq_date,
                    product_path=self._abspath(r.get("product_path")),
                    crs=None if pd.isna(r.get("crs")) else str(r.get("crs")),
                    res_m=None if pd.isna(r.get("res_m")) else int(r.get("res_m")),
                ),
                asset_path=asset_path,
                band=None if pd.isna(r.get("band")) else str(r.get("band")).upper(),
                date=acq_date,
                crs=None if pd.isna(r.get("crs")) else str(r.get("crs")),
                res_m=None if pd.isna(r.get("res_m")) else int(r.get("res_m")),
                extras=r.get("_extras", {}) or {},
            )
            assets.append(base)

        self._rois = list(rois.values())
        self._mosaics = mosaics
        self._assets = assets

    # -------------
    # API CatalogPort
    # -------------
    def list_rois(self) -> Sequence[ROIItem]:
        return self._rois

    def list_mosaics(self, roi_id: Optional[str] = None) -> Sequence[MosaicItem]:
        if roi_id is None:
            return self._mosaics
        rid = str(roi_id)
        return [m for m in self._mosaics if (m.roi_id or "") == rid]

    def iter_assets(
        self,
        roi_id: Optional[str] = None,
        mosaic_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> Iterable[CatalogItem]:
        def _in_range(d: Optional[date]) -> bool:
            if d is None:
                return True
            if date_from and d < date_from:
                return False
            if date_to and d > date_to:
                return False
            return True

        for item in self._assets or []:
            if roi_id and (item.mosaic and item.mosaic.roi_id) != str(roi_id):
                continue
            if mosaic_id and (item.mosaic and item.mosaic.mosaic_id) != str(mosaic_id):
                continue
            if not _in_range(item.date):
                continue
            yield item

        # Si no hay activos explícitos, exponemos mosaicos como "activos de alto nivel"
        if not self._assets:
            for m in self.list_mosaics(roi_id=roi_id):
                if not _in_range(m.acq_date):
                    continue
                yield CatalogItem(
                    roi=next((r for r in self._rois if r.roi_id == (m.roi_id or "")), None),
                    mosaic=m,
                    asset_path=m.product_path,
                    date=m.acq_date,
                    crs=m.crs,
                    res_m=m.res_m,
                )
