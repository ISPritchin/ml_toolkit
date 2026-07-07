"""Тесты градиентной корректности ml_toolkit/losses/_logitnorm.py (LogitNormLoss)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import LogitNormLoss
from tests.losses.conftest import softmax


class TestLogitNormLoss:
    def test_matches_manual_loss_reconstruction(self):
        t = 1.0

        def loss_fn(z, y):
            norm = max(np.sqrt(np.sum(z * z)), 1e-7)
            p = softmax(z / (t * norm))
            return -np.log(p[y] + 1e-12)

        loss = LogitNormLoss(temperature=t)
        rng = np.random.default_rng(6)
        for _ in range(20):
            z = rng.normal(size=4) * 0.8
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

    def test_rejects_nonpositive_temperature(self):
        with pytest.raises(ValueError):
            LogitNormLoss(temperature=0.0)

    def test_respects_sample_weight(self):
        """Регресс: ранняя версия принимала weight в сигнатуре, но не

        домножала на него der1/der2 (в отличие от сестринских
        EqualizationLoss/BalancedSoftmaxLoss, где weight используется).
        """
        loss = LogitNormLoss(temperature=0.5)
        z = np.array([0.5, -0.3, 0.1, 0.2])
        der1_w1, der2_w1 = loss.calc_ders_multi(z.copy(), 1.0, 1.0)
        der1_w2, der2_w2 = loss.calc_ders_multi(z.copy(), 1.0, 2.0)
        assert np.allclose(np.array(der1_w2), 2.0 * np.array(der1_w1))
        assert np.allclose(np.array(der2_w2), 2.0 * np.array(der2_w1))
