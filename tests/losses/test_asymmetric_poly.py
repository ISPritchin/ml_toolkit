"""Тесты градиентной корректности ml_toolkit/losses/_asymmetric_poly.py (AsymmetricPolyLoss)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.losses import AsymmetricLoss, AsymmetricPolyLoss
from tests.losses.conftest import sigmoid


class TestAsymmetricPolyLoss:
    def test_matches_manual_loss_reconstruction(self):
        eps = 1e-7
        gp, gn, m, eps1 = 0.5, 3.0, 0.05, 2.0

        def loss_fn(f, y):
            p = sigmoid(f)
            if y == 1:
                base = -((1 - p) ** gp) * np.log(p + eps)
                p_t = p
            else:
                p_s = max(p - m, 0.0)
                base = (p_s ** gn) * np.log(1 - p_s + eps) if p_s > 0 else 0.0
                p_t = 1 - p
            return base + eps1 * (1 - p_t)

        loss = AsymmetricPolyLoss(gamma_pos=gp, gamma_neg=gn, prob_margin=m, eps1=eps1)
        rng = np.random.default_rng(2)
        for _ in range(30):
            f0 = rng.normal() * 3
            y0 = float(rng.random() < 0.4)
            h = 1e-6
            numeric = -(loss_fn(f0 + h, y0) - loss_fn(f0 - h, y0)) / (2 * h)
            der1_code, _ = loss.calc_ders_range(np.array([f0]), np.array([y0]), None)[0]
            assert abs(numeric - der1_code) < 1e-3

    def test_reduces_to_asymmetric_loss_when_eps1_zero(self):
        rng = np.random.default_rng(3)
        f = rng.normal(size=30)
        y = (rng.random(30) < 0.3).astype(np.float64)
        asl = AsymmetricLoss(gamma_pos=0.5, gamma_neg=3.0, prob_margin=0.05)
        asl_poly = AsymmetricPolyLoss(gamma_pos=0.5, gamma_neg=3.0, prob_margin=0.05, eps1=0.0)
        der1_asl = [d[0] for d in asl.calc_ders_range(f, y, None)]
        der1_poly = [d[0] for d in asl_poly.calc_ders_range(f, y, None)]
        assert np.allclose(der1_asl, der1_poly)
