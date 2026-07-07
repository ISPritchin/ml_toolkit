"""Тесты градиентной корректности ml_toolkit/losses/_ghm.py (GHMLoss)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import GHMLoss
from tests.losses.conftest import numeric_der1_binary


class TestGHMLoss:
    def test_der2_matches_numeric_derivative_of_der1(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=100) * 2
        y = (rng.random(100) < 0.3).astype(np.float64)
        loss = GHMLoss(bins=10, momentum=0.0)
        der1, der2 = zip(*loss.calc_ders_range(f, y, None))
        der1, der2 = np.array(der1), np.array(der2)
        for i in rng.choice(100, size=15, replace=False):
            numeric = numeric_der1_binary(GHMLoss(bins=10, momentum=0.0), f, y, i)
            assert abs(numeric - der2[i]) < 1e-3

    def test_momentum_smooths_across_calls(self):
        loss = GHMLoss(bins=5, momentum=0.9)
        f1 = np.array([0.0, 0.0, 5.0, 5.0])
        y1 = np.array([1.0, 1.0, 0.0, 0.0])
        loss.calc_ders_range(f1, y1, None)
        acc_after_first = loss._acc_counts.copy()

        # Second call has a different gradient-norm distribution (all near-zero,
        # i.e. all in the lowest bin) — the EMA should move toward it, not jump to it.
        f2 = np.array([0.0, 0.0, 0.0, 0.0])
        y2 = np.array([1.0, 1.0, 1.0, 1.0])
        loss.calc_ders_range(f2, y2, None)
        assert not np.allclose(acc_after_first, loss._acc_counts)

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            GHMLoss(bins=0)
        with pytest.raises(ValueError):
            GHMLoss(momentum=1.0)
