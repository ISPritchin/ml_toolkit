"""Тесты градиентной корректности ml_toolkit/losses/_asl.py (AsymmetricLoss)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.losses import AsymmetricLoss
from tests.losses.conftest import sigmoid


class TestAsymmetricLossRegression:
    """Регресс на баг с перевёрнутым знаком der1 в негативной ветке (y=0, p > prob_margin)."""

    def test_der1_sign_matches_loss_direction(self):
        eps = 1e-7
        m, gn = 0.05, 3.0

        def loss_neg(f):
            p = sigmoid(f)
            p_s = max(p - m, 0.0)
            return (p_s ** gn) * np.log(1 - p_s + eps) if p_s > 0 else 0.0

        loss = AsymmetricLoss(gamma_pos=0.0, gamma_neg=gn, prob_margin=m)
        rng = np.random.default_rng(4)
        for _ in range(30):
            f0 = rng.normal() * 3
            h = 1e-6
            numeric = -(loss_neg(f0 + h) - loss_neg(f0 - h)) / (2 * h)
            der1_code, _ = loss.calc_ders_range(np.array([f0]), np.array([0.0]), None)[0]
            assert abs(numeric - der1_code) < 1e-3
