"""Unit tests para EvaluationService (protocolos, skip, exclusión, CSV)."""
import numpy as np
import pandas as pd
import pytest

from satplatform.contracts.core import ClassLabel, MacroClass
from satplatform.adapters.euclidean_classifier import EuclideanClassifierAdapter
from satplatform.services.evaluation_service import EvalConfig, EvaluationService

BANDS = ["B02", "B03", "B04", "B08"]


def _catalog():
    return [
        ClassLabel(id=1, name="Agua", macro=MacroClass.AGUA),
        ClassLabel(id=2, name="Relave", macro=MacroClass.RELAVE),
        ClassLabel(id=3, name="Terreno", macro=MacroClass.TERRENO),
    ]


def _configs():
    def make(train_df, classes):
        return EuclideanClassifierAdapter.fit(train_df, classes, band_filter=BANDS, include_hsl=False)
    return [EvalConfig("prod", make)]


def _synth_df(dates=("2020-04-03", "2021-04-28", "2022-04-28"), seed=0, class3_only=None, n=14):
    rng = np.random.default_rng(seed)
    centers = {1: 500.0, 2: 5000.0, 3: 9000.0}
    rows = []
    for d in dates:
        for ng, c in centers.items():
            if class3_only is not None and ng == 3 and d != class3_only:
                continue
            for k in range(n):
                rows.append({
                    "Fecha": d,
                    "UTM_E": 480000 + ng * 2000 + rng.normal(0, 60) + k * 3,
                    "UTM_N": 7300000 + ng * 2000 + rng.normal(0, 60) + k * 3,
                    "Ng": ng,
                    **{b: float(rng.normal(c, 30.0)) for b in BANDS},
                })
    return pd.DataFrame(rows)


class TestProtocols:
    def test_runs_all_protocols_multidate(self):
        svc = EvaluationService()
        res = svc.evaluate(_synth_df(), _configs(), catalog=_catalog(), n_folds=4)
        assert not res.summary.empty
        protos = set(res.summary["protocol"])
        assert {"p0", "p1", "p2", "p3"} <= protos
        assert res.block_size_m > 0

    def test_high_accuracy_on_separable_data(self):
        # centroides muy separados → OA alta en todos los protocolos
        svc = EvaluationService()
        res = svc.evaluate(_synth_df(), _configs(), catalog=_catalog(), n_folds=4)
        assert res.summary["OA"].mean() > 0.9

    def test_p2_p3_skipped_single_date(self):
        svc = EvaluationService()
        res = svc.evaluate(_synth_df(dates=("2024-01-23",)), _configs(),
                           catalog=_catalog(), n_folds=4)
        assert any("p2" in s for s in res.skipped)
        assert any("p3" in s for s in res.skipped)
        assert set(res.summary["protocol"]) == {"p0", "p1"}


class TestClassExclusion:
    def test_class_with_low_support_excluded_in_fold(self):
        # clase 3 solo existe en 2022 → en el fold LODO held=2022, train no la tiene
        df = _synth_df(class3_only="2022-04-28")
        svc = EvaluationService()
        res = svc.evaluate(df, _configs(), catalog=_catalog(), protocols=("p2",))
        held22 = [fr for fr in res.per_fold if fr.held_date == "2022-04-28"]
        assert held22, "debe existir el fold held=2022"
        assert all(3 in fr.excluded_classes for fr in held22)
        # y la clase 3 no aparece en las clases evaluadas de ese fold
        assert all(3 not in fr.class_ids for fr in held22)


class TestInvariantAndOutput:
    def test_q_plus_a_equals_1_minus_oa_per_fold(self):
        svc = EvaluationService()
        res = svc.evaluate(_synth_df(), _configs(), catalog=_catalog(), protocols=("p0",), n_folds=4)
        for fr in res.per_fold:
            assert fr.metrics["Q"] + fr.metrics["A"] == pytest.approx(1.0 - fr.metrics["OA"], abs=1e-9)

    def test_writes_csv(self, tmp_path):
        svc = EvaluationService()
        svc.evaluate(_synth_df(), _configs(), catalog=_catalog(), n_folds=4, out_dir=tmp_path)
        assert (tmp_path / "eval_summary.csv").exists()
        assert (tmp_path / "eval_per_class.csv").exists()
        assert (tmp_path / "autocorr_range.csv").exists()
        # al menos una matriz de confusión
        assert list(tmp_path.glob("confusion_*.csv"))

    def test_explicit_block_size_used(self):
        svc = EvaluationService()
        res = svc.evaluate(_synth_df(), _configs(), catalog=_catalog(),
                           protocols=("p1",), block_size_m=1500.0, n_folds=3)
        assert res.block_size_m == pytest.approx(1500.0)
