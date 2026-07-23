"""Unit tests para evaluation_metrics (valores conocidos + invariante Pontius)."""
import numpy as np
import pytest

from satplatform.services import evaluation_metrics as m


def _known_cm():
    """CM[pred,true] con métricas calculadas a mano.
    [[20, 5],
     [10, 15]]  (clases 1,2)  N=50, OA=0.7, kappa=0.4
    """
    return np.array([[20, 5], [10, 15]], dtype=np.int64)


class TestConfusionMatrix:
    def test_counts_orientation_pred_true(self):
        # 2 pred=1/true=2 y 1 pred=2/true=1
        y_true = [1, 2, 2, 1]
        y_pred = [1, 1, 1, 2]
        cm = m.confusion_matrix(y_true, y_pred, [1, 2])
        # pred=1,true=1 →1 ; pred=1,true=2 →2 ; pred=2,true=1 →1 ; pred=2,true=2 →0
        assert cm.tolist() == [[1, 2], [1, 0]]

    def test_ignores_unknown_classes(self):
        cm = m.confusion_matrix([1, 9], [1, 9], [1, 2])
        assert cm.sum() == 1  # el par (9,9) se ignora


class TestScalarMetrics:
    def test_overall_accuracy(self):
        assert m.overall_accuracy(_known_cm()) == pytest.approx(0.7)

    def test_kappa(self):
        assert m.kappa(_known_cm()) == pytest.approx(0.4)

    def test_producers_users(self):
        pa, ua = m.producers_users(_known_cm())
        assert pa == pytest.approx([20 / 30, 15 / 20])  # recall por columna (verdad)
        assert ua == pytest.approx([20 / 25, 15 / 25])  # precision por fila (predicho)

    def test_f1_macro(self):
        assert m.f1_macro(_known_cm()) == pytest.approx(0.69697, abs=1e-4)

    def test_perfect_classification(self):
        cm = np.array([[10, 0], [0, 10]], dtype=np.int64)
        assert m.overall_accuracy(cm) == pytest.approx(1.0)
        assert m.kappa(cm) == pytest.approx(1.0)
        assert m.f1_macro(cm) == pytest.approx(1.0)


class TestPontius:
    def test_quantity_allocation_values(self):
        q, a = m.quantity_allocation_disagreement(_known_cm())
        assert q == pytest.approx(0.1)
        assert a == pytest.approx(0.2)

    def test_invariant_q_plus_a_equals_1_minus_oa(self):
        cm = _known_cm()
        q, a = m.quantity_allocation_disagreement(cm)
        assert q + a == pytest.approx(1.0 - m.overall_accuracy(cm))

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_invariant_random_cm(self, seed):
        rng = np.random.default_rng(seed)
        cm = rng.integers(0, 20, size=(4, 4)).astype(np.int64)
        q, a = m.quantity_allocation_disagreement(cm)
        assert q + a == pytest.approx(1.0 - m.overall_accuracy(cm), abs=1e-9)


class TestAutocorr:
    def test_range_shrinks_with_random_field(self):
        # campo espacial ALEATORIO (sin autocorrelación) → rango pequeño
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 1000, size=(80, 2))
        values = rng.normal(0, 1, 80)
        out = m.estimate_autocorr_range(coords, values, n_bins=10)
        assert "range_m" in out and out["range_m"] >= 0.0
        assert out["e_i"] == pytest.approx(-1.0 / 79)

    def test_degenerate_small_n(self):
        out = m.estimate_autocorr_range(np.zeros((2, 2)), [1.0, 2.0])
        assert out["range_m"] == 0.0
