"""Unit tests para evaluation_folds (partición correcta, sin fuga espacial)."""
import numpy as np
import pytest

from satplatform.services import evaluation_folds as f


class TestRandomFolds:
    def test_partition_covers_all_and_disjoint(self):
        folds = f.random_folds(50, n_folds=5, seed=42)
        assert len(folds) == 5
        test_all = np.concatenate([te for _, te in folds])
        assert sorted(test_all.tolist()) == list(range(50))  # cubre todo, sin repetir
        for tr, te in folds:
            assert set(tr.tolist()).isdisjoint(set(te.tolist()))

    def test_requires_min_folds(self):
        with pytest.raises(ValueError):
            f.random_folds(10, n_folds=1)


class TestSpatialBlockFolds:
    def _clustered_coords(self):
        # 4 clusters bien separados (bloques) de 10 puntos cada uno
        rng = np.random.default_rng(0)
        centers = [(0, 0), (10000, 0), (0, 10000), (10000, 10000)]
        pts = []
        for cx, cy in centers:
            pts.append(np.column_stack([
                rng.uniform(cx, cx + 100, 10),
                rng.uniform(cy, cy + 100, 10),
            ]))
        return np.vstack(pts)

    def test_no_block_shared_between_train_and_test(self):
        coords = self._clustered_coords()
        bs = 1000.0  # cada cluster (spread 100 m) cae en su propia celda de 1 km
        folds = f.spatial_block_folds(coords, block_size_m=bs, n_folds=4, seed=1)
        assert len(folds) >= 2
        cells = np.floor(coords / bs).astype(np.int64)
        keys = cells[:, 0] * 100000 + cells[:, 1]
        for tr, te in folds:
            train_blocks = set(keys[tr].tolist())
            test_blocks = set(keys[te].tolist())
            assert train_blocks.isdisjoint(test_blocks)  # sin fuga espacial

    def test_rejects_bad_block_size(self):
        with pytest.raises(ValueError):
            f.spatial_block_folds(np.zeros((5, 2)), block_size_m=0.0, n_folds=3)

    def test_rejects_bad_coords_shape(self):
        with pytest.raises(ValueError):
            f.spatial_block_folds(np.zeros((5, 3)), block_size_m=10.0, n_folds=3)


class TestTemporalFolds:
    def test_leave_one_date_out(self):
        dates = ["2020", "2020", "2021", "2022"]
        folds = f.leave_one_date_out(dates)
        assert len(folds) == 3
        for tr, te, held in folds:
            # test = solo la fecha held; train = ninguna de esa fecha
            assert all(dates[i] == held for i in te)
            assert all(dates[i] != held for i in tr)

    def test_leave_one_date_out_single_date_empty(self):
        assert f.leave_one_date_out(["2024", "2024"]) == []

    def test_anchor_date_folds_trains_on_earliest(self):
        dates = ["2020", "2021", "2022", "2020"]
        folds = f.anchor_date_folds(dates)  # ancla = 2020 (más antigua)
        assert {held for _, _, held in folds} == {"2021", "2022"}
        for tr, te, held in folds:
            assert all(dates[i] == "2020" for i in tr)
            assert all(dates[i] == held for i in te)

    def test_anchor_single_date_empty(self):
        assert f.anchor_date_folds(["2024"]) == []
