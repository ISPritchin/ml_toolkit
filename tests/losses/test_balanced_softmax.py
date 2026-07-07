"""Тесты градиентной корректности ml_toolkit/losses/_balanced_softmax.py (BalancedSoftmaxLoss)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import BalancedSoftmaxLoss
from tests.losses.conftest import softmax


class TestBalancedSoftmaxLoss:
    def test_matches_manual_loss_reconstruction(self):
        counts = np.array([500.0, 100.0, 30.0, 10.0])
        tau = 1.0
        log_prior = tau * np.log(counts / counts.sum())

        def loss_fn(z, y):
            p = softmax(z + log_prior)
            return -np.log(p[y] + 1e-12)

        loss = BalancedSoftmaxLoss(counts, tau=tau)
        rng = np.random.default_rng(5)
        for _ in range(20):
            z = rng.normal(size=4) * 1.5
            y = int(rng.integers(0, 4))
            der1, der2 = loss.calc_ders_multi(z.copy(), float(y), 1.0)
            der1 = np.array(der1)
            assert np.array(der2).shape == (4, 4)
            for k in range(4):
                zp, zm = z.copy(), z.copy()
                zp[k] += 1e-6
                zm[k] -= 1e-6
                numeric = -(loss_fn(zp, y) - loss_fn(zm, y)) / 2e-6
                assert abs(numeric - der1[k]) < 1e-3

    def test_rejects_invalid_counts(self):
        with pytest.raises(ValueError):
            BalancedSoftmaxLoss([0, 10, 10])
