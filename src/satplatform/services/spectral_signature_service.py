"""Análisis de firmas espectrales por clase (dominio puro, sin I/O ni UI).

Opera sobre un DataFrame de muestras (el `train_df` del TrainingSetBuilder:
columnas Ng + bandas, y opcionalmente Fecha), construyendo el espacio de features
con FeatureService (bandas + HSL + índices). Produce DataFrames listos para
graficar: firmas media±std por clase, matriz de separabilidad, proyección PCA y
evolución temporal de una clase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .feature_service import FeatureService

# Longitudes de onda centrales Sentinel-2 (nm) — de config_bandas_v3.json.
S2_WAVELENGTHS_NM: dict[str, float] = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B8A": 864.7, "B09": 945.1, "B11": 1613.7, "B12": 2202.4,
}


@dataclass
class SpectralSignatureService:
    features: FeatureService = field(default_factory=FeatureService)

    def _matrix(self, df, band_filter, include_hsl, indices):
        X, names = self.features.from_dataframe(df, band_filter, include_hsl, indices)
        return X, names

    def signatures_by_class(
        self,
        df: pd.DataFrame,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Media±std de cada feature por clase. Columnas: class_id, feature, mean, std, n."""
        X, names = self._matrix(df, band_filter, include_hsl, indices)
        ng = df["Ng"].to_numpy()
        rows = []
        for cid in sorted(np.unique(ng)):
            Xc = X[ng == cid]
            if len(Xc) == 0:
                continue
            for j, fname in enumerate(names):
                rows.append({
                    "class_id": int(cid), "feature": fname,
                    "mean": float(np.nanmean(Xc[:, j])),
                    "std": float(np.nanstd(Xc[:, j])),
                    "n": int(len(Xc)),
                })
        return pd.DataFrame(rows, columns=["class_id", "feature", "mean", "std", "n"])

    def separability_matrix(
        self,
        df: pd.DataFrame,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
        metric: str = "euclidean",
    ) -> pd.DataFrame:
        """Distancia entre centroides de clase (matriz NxN, simétrica, diagonal 0).

        metric: 'euclidean' (en el espacio de features crudo) o 'mahalanobis'
        (usando la inversa de la covarianza global → escala-invariante).
        """
        X, _ = self._matrix(df, band_filter, include_hsl, indices)
        ng = df["Ng"].to_numpy()
        classes = sorted(int(c) for c in np.unique(ng))
        centroids = {c: np.nanmean(X[ng == c], axis=0) for c in classes}

        if metric == "mahalanobis":
            cov = np.cov(X, rowvar=False)
            vi = np.linalg.pinv(np.atleast_2d(cov))
        elif metric != "euclidean":
            raise ValueError(f"metric no soportada: {metric}")

        n = len(classes)
        M = np.zeros((n, n), dtype=np.float64)
        for a, ca in enumerate(classes):
            for b, cb in enumerate(classes):
                d = centroids[ca] - centroids[cb]
                if metric == "euclidean":
                    M[a, b] = float(np.sqrt(np.sum(d * d)))
                else:
                    M[a, b] = float(np.sqrt(max(d @ vi @ d, 0.0)))
        return pd.DataFrame(M, index=classes, columns=classes)

    def pca_2d(
        self,
        df: pd.DataFrame,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Proyección PCA a 2 componentes (sobre features estandarizadas).
        Columnas: class_id, pc1, pc2."""
        from sklearn.decomposition import PCA

        X, _ = self._matrix(df, band_filter, include_hsl, indices)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        pcs = PCA(n_components=2).fit_transform(Xs)
        return pd.DataFrame({
            "class_id": df["Ng"].to_numpy().astype(int),
            "pc1": pcs[:, 0], "pc2": pcs[:, 1],
        })

    def temporal_means(
        self,
        df: pd.DataFrame,
        class_id: int,
        band_filter: Sequence[str],
        include_hsl: bool = False,
        indices: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Media de cada feature por Fecha para una clase. Columnas: Fecha, feature, mean."""
        sub = df[df["Ng"] == class_id]
        if len(sub) == 0 or "Fecha" not in sub.columns:
            return pd.DataFrame(columns=["Fecha", "feature", "mean"])
        X, names = self._matrix(sub, band_filter, include_hsl, indices)
        fechas = sub["Fecha"].to_numpy()
        rows = []
        for f in sorted(set(fechas)):
            Xf = X[fechas == f]
            for j, fname in enumerate(names):
                rows.append({"Fecha": str(f), "feature": fname, "mean": float(np.nanmean(Xf[:, j]))})
        return pd.DataFrame(rows, columns=["Fecha", "feature", "mean"])


__all__ = ["SpectralSignatureService"]
