"""Módulo de evaluación de clasificación (Milestone 1).

Orquesta 4 protocolos sobre el DataFrame de puntos Mcal `[Fecha, UTM_E, UTM_N, Ng, <bandas>]`:
  - P0  CV aleatoria (baseline, exhibe el sesgo del split aleatorio).
  - P1  CV espacial por bloques (generalización espacial, sin fuga).
  - P2  TFC leave-one-date-out (transferencia temporal).
  - P3  TFC ancla (entrena en la fecha más antigua, clasifica las demás).

Genérico por clasificador: recibe `EvalConfig(make_classifier=(train_df, classes)->PixelClassifierPort)`.
Política de clases: por fold se excluyen las clases con soporte < 2 en el train (evita la
covarianza-identidad de fallback del clasificador); se reportan como `excluded_classes`.
Hook `norm` (M1 = None) transforma columnas de banda por fecha antes de clasificar → seam del RRN (M2).

Puro dominio: no importa de `composition/`. Depende solo de `PixelClassifierPort` y de los
módulos puros `evaluation_folds` / `evaluation_metrics`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..contracts.core import ClassLabel
from ..ports.pixel_class import PixelClassifierPort
from . import evaluation_folds as folds
from . import evaluation_metrics as metrics

# make_classifier(train_df, classes) -> clasificador entrenado
ClassifierFactory = Callable[[pd.DataFrame, Sequence[ClassLabel]], PixelClassifierPort]
# norm(df_subset, fecha) -> df_subset con columnas de banda transformadas
NormFn = Callable[[pd.DataFrame, str], pd.DataFrame]

_META_COLS = ("Fecha", "UTM_E", "UTM_N", "Ng")


@dataclass(frozen=True)
class EvalConfig:
    name: str
    make_classifier: ClassifierFactory


@dataclass
class FoldResult:
    config: str
    protocol: str
    fold_id: int
    held_date: Optional[str]
    class_ids: Tuple[int, ...]
    confusion: np.ndarray
    metrics: Dict[str, float]
    per_class: pd.DataFrame
    excluded_classes: Tuple[int, ...]
    n_train: int
    n_test: int


@dataclass
class EvalResult:
    per_fold: List[FoldResult]
    summary: pd.DataFrame
    per_class: pd.DataFrame
    block_size_m: float
    autocorr: Optional[dict] = None
    skipped: List[str] = field(default_factory=list)


def _band_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _extent(coords: np.ndarray) -> float:
    spans = coords.max(axis=0) - coords.min(axis=0)
    return float(spans.max()) if spans.size else 0.0


def _n_cells(coords: np.ndarray, bs: float) -> int:
    cells = np.floor(coords / float(bs)).astype(np.int64)
    return int(np.unique(cells, axis=0).shape[0])


def _choose_block_size(
    df: pd.DataFrame, coords: np.ndarray, requested: Optional[float], n_folds: int
) -> Tuple[float, Optional[dict]]:
    """Devuelve (block_size_m, autocorr_info). Si `requested` es None/<=0, estima con
    Moran's I sobre la 1ª PC de las bandas y garantiza >= n_folds bloques."""
    autocorr: Optional[dict] = None
    if requested and requested > 0:
        bs = float(requested)
    else:
        bands = _band_columns(df)
        if bands:
            X = df[bands].to_numpy(dtype=float)
            pc1 = metrics.first_principal_component(X)
            autocorr = metrics.estimate_autocorr_range(coords, pc1)
            bs = float(autocorr.get("range_m") or 0.0)
        else:
            bs = 0.0
        if bs <= 0:
            ext = _extent(coords)
            bs = ext / (2 * n_folds) if ext > 0 else 1.0
    # garantiza suficientes celdas para armar n_folds
    for _ in range(25):
        if _n_cells(coords, bs) >= n_folds or bs <= 1.0:
            break
        bs *= 0.6
    return max(bs, 1.0), autocorr


def _apply_norm(df: pd.DataFrame, norm: Optional[NormFn]) -> pd.DataFrame:
    if norm is None:
        return df
    parts = []
    for fecha, group in df.groupby("Fecha", sort=False):
        parts.append(norm(group.copy(), str(fecha)))
    return pd.concat(parts, ignore_index=False).loc[df.index]


@dataclass
class EvaluationService:
    seed: int = 42

    def evaluate(
        self,
        df: pd.DataFrame,
        configs: Sequence[EvalConfig],
        *,
        catalog: Sequence[ClassLabel],
        protocols: Sequence[str] = ("p0", "p1", "p2", "p3"),
        block_size_m: Optional[float] = None,
        n_folds: int = 5,
        norm: Optional[NormFn] = None,
        out_dir: Optional[Path] = None,
    ) -> EvalResult:
        if "Ng" not in df.columns:
            raise ValueError("df debe tener columna 'Ng'")
        df = _apply_norm(df.reset_index(drop=True), norm)
        name_by_id = {int(c.id): c.name for c in catalog}
        coords = df[["UTM_E", "UTM_N"]].to_numpy(dtype=float)

        bs, autocorr = _choose_block_size(df, coords, block_size_m, n_folds)

        per_fold: List[FoldResult] = []
        skipped: List[str] = []

        # construye la lista de folds por protocolo (índices posicionales)
        def _fold_specs(protocol: str):
            if protocol == "p0":
                return [(tr, te, None) for tr, te in folds.random_folds(len(df), n_folds, self.seed)]
            if protocol == "p1":
                return [(tr, te, None) for tr, te in folds.spatial_block_folds(coords, bs, n_folds, self.seed)]
            if protocol == "p2":
                return folds.leave_one_date_out(df["Fecha"].tolist())
            if protocol == "p3":
                return folds.anchor_date_folds(df["Fecha"].tolist())
            raise ValueError(f"protocolo desconocido: {protocol}")

        for protocol in protocols:
            specs = _fold_specs(protocol)
            if not specs:
                skipped.append(f"{protocol} (sin folds: se requieren >=2 fechas)")
                continue
            for cfg in configs:
                for fold_id, (train_idx, test_idx, held_date) in enumerate(specs):
                    fr = self._eval_fold(
                        df, np.asarray(train_idx), np.asarray(test_idx), cfg, catalog,
                        name_by_id, protocol, fold_id, held_date,
                    )
                    if fr is not None:
                        per_fold.append(fr)

        summary = self._build_summary(per_fold)
        per_class = self._build_per_class(per_fold)
        result = EvalResult(
            per_fold=per_fold, summary=summary, per_class=per_class,
            block_size_m=bs, autocorr=autocorr, skipped=skipped,
        )
        if out_dir is not None:
            self._write_outputs(result, Path(out_dir), catalog)
        return result

    def _eval_fold(
        self, df, train_idx, test_idx, cfg, catalog, name_by_id, protocol, fold_id, held_date,
    ) -> Optional[FoldResult]:
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        counts = train_df["Ng"].value_counts().to_dict()
        usable = tuple(c for c in catalog if int(counts.get(int(c.id), 0)) >= 2)
        usable_ids = [int(c.id) for c in usable]
        excluded = tuple(int(c.id) for c in catalog if int(c.id) not in usable_ids)
        if not usable_ids:
            return None
        test_used = test_df[test_df["Ng"].isin(usable_ids)]
        if len(test_used) == 0:
            return None

        clf = cfg.make_classifier(train_df, usable)
        y_true = test_used["Ng"].to_numpy()
        y_pred = clf.predict_points(test_used)

        cm = metrics.confusion_matrix(y_true, y_pred, usable_ids)
        oa = metrics.overall_accuracy(cm)
        kp = metrics.kappa(cm)
        f1 = metrics.f1_macro(cm)
        q, a = metrics.quantity_allocation_disagreement(cm)
        pa, ua = metrics.producers_users(cm)
        f1c = metrics.f1_per_class(cm)
        support = cm.sum(axis=0)  # por columna = verdad

        per_class = pd.DataFrame({
            "class_id": usable_ids,
            "class_name": [name_by_id.get(c, str(c)) for c in usable_ids],
            "support": support.astype(int),
            "PA": pa,
            "UA": ua,
            "F1": f1c,
        })
        return FoldResult(
            config=cfg.name, protocol=protocol, fold_id=fold_id, held_date=held_date,
            class_ids=tuple(usable_ids), confusion=cm,
            metrics={"OA": oa, "kappa": kp, "F1_macro": f1, "Q": q, "A": a},
            per_class=per_class, excluded_classes=excluded,
            n_train=int(len(train_df)), n_test=int(len(test_used)),
        )

    @staticmethod
    def _build_summary(per_fold: List[FoldResult]) -> pd.DataFrame:
        rows = []
        for fr in per_fold:
            rows.append({
                "config": fr.config, "protocol": fr.protocol, "fold": fr.fold_id,
                "held_date": fr.held_date or "", "n_train": fr.n_train, "n_test": fr.n_test,
                "OA": fr.metrics["OA"], "kappa": fr.metrics["kappa"],
                "F1_macro": fr.metrics["F1_macro"], "Q": fr.metrics["Q"], "A": fr.metrics["A"],
                "excluded_classes": ",".join(str(c) for c in fr.excluded_classes),
            })
        cols = ["config", "protocol", "fold", "held_date", "n_train", "n_test",
                "OA", "kappa", "F1_macro", "Q", "A", "excluded_classes"]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _build_per_class(per_fold: List[FoldResult]) -> pd.DataFrame:
        frames = []
        for fr in per_fold:
            pc = fr.per_class.copy()
            pc.insert(0, "held_date", fr.held_date or "")
            pc.insert(0, "fold", fr.fold_id)
            pc.insert(0, "protocol", fr.protocol)
            pc.insert(0, "config", fr.config)
            frames.append(pc)
        if not frames:
            return pd.DataFrame(
                columns=["config", "protocol", "fold", "held_date",
                         "class_id", "class_name", "support", "PA", "UA", "F1"]
            )
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _write_outputs(result: EvalResult, out_dir: Path, catalog: Sequence[ClassLabel]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(out_dir / "eval_summary.csv", index=False)
        result.per_class.to_csv(out_dir / "eval_per_class.csv", index=False)
        for fr in result.per_fold:
            hd = f"_{fr.held_date}" if fr.held_date else ""
            fname = f"confusion_{fr.config}_{fr.protocol}_{fr.fold_id}{hd}.csv"
            cm_df = pd.DataFrame(
                fr.confusion, index=[f"pred_{c}" for c in fr.class_ids],
                columns=[f"true_{c}" for c in fr.class_ids],
            )
            cm_df.to_csv(out_dir / fname)
        # correlograma / rango de autocorrelación
        rows = [{"block_size_m_used": result.block_size_m}]
        if result.autocorr:
            rows[0]["e_i"] = result.autocorr.get("e_i")
            rows[0]["range_m_estimado"] = result.autocorr.get("range_m")
        ac = pd.DataFrame(rows)
        if result.autocorr and result.autocorr.get("correlogram"):
            corr = pd.DataFrame(
                result.autocorr["correlogram"], columns=["dist_m", "morans_I", "n_pares"]
            )
            corr.to_csv(out_dir / "autocorr_range.csv", index=False)
        else:
            ac.to_csv(out_dir / "autocorr_range.csv", index=False)


__all__ = ["EvalConfig", "FoldResult", "EvalResult", "EvaluationService", "ClassifierFactory", "NormFn"]
