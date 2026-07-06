"""Тесты градиентной корректности для ml_toolkit/losses/.

Для calc_ders_range-лоссов der1 = -dL/df, der2 = -d²L/df² — проверяется
численным дифференцированием der1 по f (совпадает с der2 с точностью
до документированных приближений, см. TverskyLoss/DiceLoss/AsymmetricPoly).
Для calc_ders_multi-лоссов der1 сверяется с численным градиентом
самостоятельно реконструированного лосса (тот же приём).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import (
    AsymmetricLoss,
    AsymmetricPolyLoss,
    BalancedSoftmaxLoss,
    DiceLoss,
    EqualizationLoss,
    FocalLoss,
    GHMLoss,
    InfluenceBalancedLoss,
    LabelSmoothingLoss,
    LDAMLoss,
    LogitNormLoss,
    NNPULoss,
    PolyLoss,
    TverskyLoss,
)


def _sigmoid(f):
    return 1.0 / (1.0 + np.exp(-f))


def _numeric_der1_binary(loss_obj, f, y, idx, h=1e-6):
    f_plus = f.copy()
    f_plus[idx] += h
    f_minus = f.copy()
    f_minus[idx] -= h
    der1_plus = np.array([d[0] for d in loss_obj.calc_ders_range(f_plus, y, None)])
    der1_minus = np.array([d[0] for d in loss_obj.calc_ders_range(f_minus, y, None)])
    return (der1_plus[idx] - der1_minus[idx]) / (2 * h)


class TestGHMLoss:
    def test_der2_matches_numeric_derivative_of_der1(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=100) * 2
        y = (rng.random(100) < 0.3).astype(np.float64)
        loss = GHMLoss(bins=10, momentum=0.0)
        der1, der2 = zip(*loss.calc_ders_range(f, y, None))
        der1, der2 = np.array(der1), np.array(der2)
        for i in rng.choice(100, size=15, replace=False):
            numeric = _numeric_der1_binary(GHMLoss(bins=10, momentum=0.0), f, y, i)
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
            p = _sigmoid(np.array([f_val]))[0]
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


class TestDiceLoss:
    def test_equivalent_to_tversky_half_half(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=50)
        y = (rng.random(50) < 0.4).astype(np.float64)
        dice = DiceLoss(smooth=1.0)
        tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)
        der1_d, der2_d = zip(*dice.calc_ders_range(f, y, None))
        der1_t, der2_t = zip(*tversky.calc_ders_range(f, y, None))
        assert np.allclose(der1_d, der1_t)
        assert np.allclose(der2_d, der2_t)


class TestAsymmetricPolyLoss:
    def test_matches_manual_loss_reconstruction(self):
        eps = 1e-7
        gp, gn, m, eps1 = 0.5, 3.0, 0.05, 2.0

        def loss_fn(f, y):
            p = _sigmoid(f)
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


class TestAsymmetricLossRegression:
    """Регресс на баг с перевёрнутым знаком der1 в негативной ветке (y=0, p > prob_margin)."""

    def test_der1_sign_matches_loss_direction(self):
        eps = 1e-7
        m, gn = 0.05, 3.0

        def loss_neg(f):
            p = _sigmoid(f)
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


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class TestBalancedSoftmaxLoss:
    def test_matches_manual_loss_reconstruction(self):
        counts = np.array([500.0, 100.0, 30.0, 10.0])
        tau = 1.0
        log_prior = tau * np.log(counts / counts.sum())

        def loss_fn(z, y):
            p = _softmax(z + log_prior)
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


class TestLogitNormLoss:
    def test_matches_manual_loss_reconstruction(self):
        t = 1.0

        def loss_fn(z, y):
            norm = max(np.sqrt(np.sum(z * z)), 1e-7)
            p = _softmax(z / (t * norm))
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
            p = _softmax(z + np.log(s + 1e-7))
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


class TestSampleWeights:
    """Регресс: calc_ders_range принимал weights в сигнатуре, но ни один

    бинарный лосс его не использовал (CatBoost не применяет sample weight
    к результату кастомного лосса сам — это обязанность лосса, проверено
    эмпирически). Для простых (per-row) лоссов weighted der = w*unweighted
    der точно; для batch-уровневых (Tversky/Dice/GHM/NNPU) — TP/FP/FN и
    risk-статистики становятся взвешенными суммами/средними, проверяется
    против независимо реконструированного взвешенного лосса.
    """

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        n = 40
        y = (rng.random(n) < 0.35).astype(np.float64)
        f = rng.normal(size=n) * 2
        w = rng.uniform(0.1, 5.0, size=n)
        return f, y, w

    @pytest.mark.parametrize('loss_factory', [
        lambda: FocalLoss(gamma=2.0, alpha=0.3),
        lambda: AsymmetricLoss(gamma_pos=0.4, gamma_neg=3.5, prob_margin=0.05),
        lambda: LabelSmoothingLoss(eps=0.1),
        lambda: PolyLoss(eps1=1.5),
        lambda: InfluenceBalancedLoss(n_pos=14, n_neg=26, alpha=500.0),
        lambda: AsymmetricPolyLoss(gamma_pos=0.4, gamma_neg=3.5, prob_margin=0.05, eps1=1.5),
    ])
    def test_simple_losses_scale_by_weight(self, data, loss_factory):
        f, y, w = data
        loss = loss_factory()
        der1_u, der2_u = zip(*loss.calc_ders_range(f, y, None))
        der1_w, der2_w = zip(*loss.calc_ders_range(f, y, w))
        assert np.allclose(der1_w, w * np.array(der1_u), atol=1e-8)
        assert np.allclose(der2_w, w * np.array(der2_u), atol=1e-8)

    def test_ldam_scales_by_weight(self, data):
        f, y, w = data
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        loss = LDAMLoss(n_pos=n_pos, n_neg=n_neg, max_margin=0.5,
                        reweight_epoch_frac=0.9, n_total_iterations=10)
        der1_u, der2_u = zip(*loss.calc_ders_range(f, y, None))
        loss2 = LDAMLoss(n_pos=n_pos, n_neg=n_neg, max_margin=0.5,
                         reweight_epoch_frac=0.9, n_total_iterations=10)
        der1_w, der2_w = zip(*loss2.calc_ders_range(f, y, w))
        assert np.allclose(der1_w, w * np.array(der1_u), atol=1e-8)
        assert np.allclose(der2_w, w * np.array(der2_u), atol=1e-8)

    def test_tversky_and_dice_weighted_tp_fp_fn(self, data):
        f, y, w = data
        eps = 1e-7

        def tversky_total(f_, y_, w_, alpha, beta, smooth):
            p = _sigmoid(f_)
            tp = np.sum(w_ * p * y_)
            fp = np.sum(w_ * p * (1 - y_))
            fn = np.sum(w_ * (1 - p) * y_)
            D = tp + alpha * fp + beta * fn + smooth
            N = tp + smooth
            return 1 - N / D

        alpha, beta, smooth = 0.35, 0.65, 1.0
        loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        der1, _ = zip(*loss.calc_ders_range(f, y, w))
        der1 = np.array(der1)
        for i in range(len(f)):
            h = 1e-6
            fp_, fm_ = f.copy(), f.copy()
            fp_[i] += h
            fm_[i] -= h
            numeric = -(tversky_total(fp_, y, w, alpha, beta, smooth)
                        - tversky_total(fm_, y, w, alpha, beta, smooth)) / (2 * h)
            assert abs(numeric - der1[i]) < 1e-3

        dice = DiceLoss(smooth=smooth)
        tversky_half = TverskyLoss(alpha=0.5, beta=0.5, smooth=smooth)
        d1, d2 = zip(*dice.calc_ders_range(f, y, w))
        t1, t2 = zip(*tversky_half.calc_ders_range(f, y, w))
        assert np.allclose(d1, t1) and np.allclose(d2, t2)

    def test_ghm_weighted_histogram(self, data):
        f, y, w = data
        loss_w = GHMLoss(bins=10, momentum=0.0)
        loss_w.calc_ders_range(f, y, w)
        loss_u = GHMLoss(bins=10, momentum=0.0)
        loss_u.calc_ders_range(f, y, np.ones(len(f)))
        assert not np.allclose(loss_w._acc_counts, loss_u._acc_counts)
        assert np.isclose(loss_w._acc_counts.sum(), w.sum())

    def test_nnpu_weighted_risk(self, data):
        f, y, w = data
        eps = 1e-7

        def nnpu_total(f_, y_, w_, pi, beta, gamma):
            p = np.clip(_sigmoid(f_), eps, 1 - eps)
            pos = y_ == 1
            w_p = max(float(w_[pos].sum()), eps)
            w_u = max(float(w_[~pos].sum()), eps)
            r_p_plus = float(np.sum(w_[pos] * -np.log(p[pos] + eps)) / w_p) if pos.any() else 0.0
            r_p_minus = float(np.sum(w_[pos] * -np.log(1 - p[pos] + eps)) / w_p) if pos.any() else 0.0
            r_u_minus = float(np.sum(w_[~pos] * -np.log(1 - p[~pos] + eps)) / w_u) if (~pos).any() else 0.0
            neg_risk = r_u_minus - pi * r_p_minus
            if neg_risk >= -beta:
                return pi * r_p_plus + neg_risk
            return pi * r_p_plus - gamma * neg_risk

        loss = NNPULoss(class_prior=0.3, beta=0.0, gamma=1.0)
        der1, _ = zip(*loss.calc_ders_range(f, y, w))
        der1 = np.array(der1)
        for i in range(len(f)):
            h = 1e-6
            fp_, fm_ = f.copy(), f.copy()
            fp_[i] += h
            fm_[i] -= h
            numeric = -(nnpu_total(fp_, y, w, 0.3, 0.0, 1.0)
                        - nnpu_total(fm_, y, w, 0.3, 0.0, 1.0)) / (2 * h)
            assert abs(numeric - der1[i]) < 1e-3
