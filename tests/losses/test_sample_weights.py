"""Сквозной регресс: поддержка sample weight во всех calc_ders_range-лоссах ml_toolkit/losses/.

Не привязан к одному файлу-источнику — проверяет общий контракт (weighted der =
w * unweighted der для per-row лоссов; взвешенные TP/FP/FN и т.п. статистики
для batch-уровневых) сразу по всем лоссам, где это раньше было не так:
calc_ders_range принимал weights в сигнатуре, но ни один бинарный лосс его не
использовал (CatBoost не применяет sample weight к результату кастомного лосса
сам — это обязанность лосса, проверено эмпирически).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import (
    AsymmetricLoss,
    AsymmetricPolyLoss,
    DiceLoss,
    FocalLoss,
    GHMLoss,
    InfluenceBalancedLoss,
    LabelSmoothingLoss,
    LDAMLoss,
    NNPULoss,
    PolyLoss,
    TverskyLoss,
)
from tests.losses.conftest import sigmoid


@pytest.fixture
def data():
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
def test_simple_losses_scale_by_weight(data, loss_factory):
    f, y, w = data
    loss = loss_factory()
    der1_u, der2_u = zip(*loss.calc_ders_range(f, y, None), strict=False)
    der1_w, der2_w = zip(*loss.calc_ders_range(f, y, w), strict=False)
    assert np.allclose(der1_w, w * np.array(der1_u), atol=1e-8)
    assert np.allclose(der2_w, w * np.array(der2_u), atol=1e-8)


def test_ldam_scales_by_weight(data):
    f, y, w = data
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    loss = LDAMLoss(n_pos=n_pos, n_neg=n_neg, max_margin=0.5,
                    reweight_epoch_frac=0.9, n_total_iterations=10)
    der1_u, der2_u = zip(*loss.calc_ders_range(f, y, None), strict=False)
    loss2 = LDAMLoss(n_pos=n_pos, n_neg=n_neg, max_margin=0.5,
                     reweight_epoch_frac=0.9, n_total_iterations=10)
    der1_w, der2_w = zip(*loss2.calc_ders_range(f, y, w), strict=False)
    assert np.allclose(der1_w, w * np.array(der1_u), atol=1e-8)
    assert np.allclose(der2_w, w * np.array(der2_u), atol=1e-8)


def test_tversky_and_dice_weighted_tp_fp_fn(data):
    f, y, w = data
    eps = 1e-7

    def tversky_total(f_, y_, w_, alpha, beta, smooth):
        p = sigmoid(f_)
        tp = np.sum(w_ * p * y_)
        fp = np.sum(w_ * p * (1 - y_))
        fn = np.sum(w_ * (1 - p) * y_)
        D = tp + alpha * fp + beta * fn + smooth
        N = tp + smooth
        return 1 - N / D

    alpha, beta, smooth = 0.35, 0.65, 1.0
    loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    der1, _ = zip(*loss.calc_ders_range(f, y, w), strict=False)
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
    d1, d2 = zip(*dice.calc_ders_range(f, y, w), strict=False)
    t1, t2 = zip(*tversky_half.calc_ders_range(f, y, w), strict=False)
    assert np.allclose(d1, t1) and np.allclose(d2, t2)


def test_ghm_weighted_histogram(data):
    f, y, w = data
    loss_w = GHMLoss(bins=10, momentum=0.0)
    loss_w.calc_ders_range(f, y, w)
    loss_u = GHMLoss(bins=10, momentum=0.0)
    loss_u.calc_ders_range(f, y, np.ones(len(f)))
    assert not np.allclose(loss_w._acc_counts, loss_u._acc_counts)
    assert np.isclose(loss_w._acc_counts.sum(), w.sum())


def test_nnpu_weighted_risk(data):
    f, y, w = data
    eps = 1e-7

    def nnpu_total(f_, y_, w_, pi, beta, gamma):
        p = np.clip(sigmoid(f_), eps, 1 - eps)
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
    der1, _ = zip(*loss.calc_ders_range(f, y, w), strict=False)
    der1 = np.array(der1)
    for i in range(len(f)):
        h = 1e-6
        fp_, fm_ = f.copy(), f.copy()
        fp_[i] += h
        fm_[i] -= h
        numeric = -(nnpu_total(fp_, y, w, 0.3, 0.0, 1.0)
                    - nnpu_total(fm_, y, w, 0.3, 0.0, 1.0)) / (2 * h)
        assert abs(numeric - der1[i]) < 1e-3
