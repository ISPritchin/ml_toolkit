"""Тесты градиентной корректности ml_toolkit/losses/_ib.py (InfluenceBalancedLoss)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import InfluenceBalancedLoss
from tests.losses.conftest import sigmoid


class TestInfluenceBalancedLoss:
    def test_der1_pushes_toward_target(self):
        loss = InfluenceBalancedLoss(n_pos=20, n_neg=80)
        f = np.array([2.0, -2.0])
        y = np.array([1.0, 0.0])
        der1, der2 = zip(*loss.calc_ders_range(f, y, None))
        # y=1, p>0.5 already right direction but der1 should still point toward increasing f (correcting residual)
        assert der1[0] > 0  # p<1, correct answer is to push f up further toward y=1
        assert der1[1] < 0  # p>0 for y=0, push f down
        assert der2[0] < 0 and der2[1] < 0

    def test_high_influence_examples_downweighted(self):
        # A very wrong prediction (large |p-y|) should get a *smaller* weight magnitude
        # relative to class-balanced-only baseline once alpha is large.
        loss_low_alpha = InfluenceBalancedLoss(n_pos=50, n_neg=50, alpha=0.0)
        loss_high_alpha = InfluenceBalancedLoss(n_pos=50, n_neg=50, alpha=1000.0)
        f = np.array([-5.0])  # very wrong for y=1 (p close to 0)
        y = np.array([1.0])
        der1_low, _ = loss_low_alpha.calc_ders_range(f, y, None)[0]
        der1_high, _ = loss_high_alpha.calc_ders_range(f, y, None)[0]
        assert abs(der1_high) < abs(der1_low)

    def test_rejects_invalid_counts(self):
        with pytest.raises(ValueError):
            InfluenceBalancedLoss(n_pos=0, n_neg=10)

    def test_der1_matches_full_derivative_including_weight_term(self):
        """Регресс: ранняя версия трактовала ib_w(p) как константу (der1 =

        ib_w*(y-p)), пропуская d(ib_w)/df — расхождение с истинным градиентом
        было устойчивым (~1e-3) и не исчезало при уменьшении шага, то есть не
        было артефактом конечных разностей.
        """
        n_pos, n_neg, alpha, beta = 30, 50, 500.0, 0.999
        loss = InfluenceBalancedLoss(n_pos=n_pos, n_neg=n_neg, alpha=alpha, beta=beta)
        eps = 1e-7

        def full_loss(f_val, y_val):
            p = sigmoid(np.array([f_val]))[0]
            cw = loss.w_pos if y_val == 1 else loss.w_neg
            w = cw / (1.0 + alpha * abs(p - y_val))
            ce = -(y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps))
            return w * ce

        rng = np.random.default_rng(7)
        for _ in range(30):
            f0 = rng.normal() * 3
            y0 = float(rng.random() < 0.4)
            h = 1e-6
            numeric = -(full_loss(f0 + h, y0) - full_loss(f0 - h, y0)) / (2 * h)
            der1_code, _ = loss.calc_ders_range(np.array([f0]), np.array([y0]), None)[0]
            assert abs(numeric - der1_code) < 1e-3
