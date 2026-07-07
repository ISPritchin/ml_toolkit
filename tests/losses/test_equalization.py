"""Тесты градиентной корректности ml_toolkit/losses/_equalization.py (EqualizationLoss)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import EqualizationLoss
from tests.losses.conftest import softmax


class TestEqualizationLoss:
    def test_matches_manual_loss_reconstruction_with_frozen_state(self):
        counts = np.array([500.0, 100.0, 30.0, 10.0])
        loss = EqualizationLoss(counts, lambda_=0.9, seesaw_p=0.8, seesaw_q=2.0)
        z = np.array([0.3, -0.2, 0.5, -0.1])
        y = 2

        avg_p = np.full(4, 0.25)
        ratio_n = counts[y] / counts
        mitigation = np.where(counts > counts[y], ratio_n ** 0.8, 1.0)
        ratio_p = avg_p / max(avg_p[y], 1e-7)
        compensation = np.where(avg_p > avg_p[y], ratio_p ** 2.0, 1.0)
        s = mitigation * compensation
        s[y] = 1.0

        def loss_fn(z):
            p = softmax(z + np.log(s + 1e-7))
            return -np.log(p[y] + 1e-12)

        der1, der2 = loss.calc_ders_multi(z.copy(), float(y), 1.0)
        der1 = np.array(der1)
        assert np.array(der2).shape == (4, 4)
        for k in range(4):
            zp, zm = z.copy(), z.copy()
            zp[k] += 1e-6
            zm[k] -= 1e-6
            numeric = -(loss_fn(zp) - loss_fn(zm)) / 2e-6
            assert abs(numeric - der1[k]) < 1e-3

    def test_ema_updates_between_calls(self):
        loss = EqualizationLoss(np.array([100.0, 50.0, 10.0]), lambda_=0.5)
        z = np.array([1.0, 0.0, -1.0])
        loss.calc_ders_multi(z, 0.0, 1.0)
        avg_after_first = loss._avg_p.copy()
        loss.calc_ders_multi(z, 0.0, 1.0)
        assert not np.allclose(avg_after_first, loss._avg_p)

    def test_rejects_invalid_counts(self):
        with pytest.raises(ValueError):
            EqualizationLoss([5])
