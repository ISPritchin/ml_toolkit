"""Тесты QuantileHuberRegressor (ml_toolkit/presets/regression/quantile_huber.py)

и корректности градиента QuantileHuberLoss (ml_toolkit/presets/regression/_losses.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models._utils import quantile_loss
from ml_toolkit.presets.regression import QuantileHuberRegressor
from ml_toolkit.presets.regression._losses import QuantileHuberLoss
from tests.presets.regression.conftest import BASE_PARAMS


def _huber(r: np.ndarray, kappa: float) -> np.ndarray:
    return np.where(np.abs(r) <= kappa, 0.5 * r * r, kappa * (np.abs(r) - 0.5 * kappa))


def _loss_value(f: np.ndarray, y: np.ndarray, quantile: float, kappa: float) -> np.ndarray:
    r = y - f
    w = np.where(r >= 0, quantile, 1.0 - quantile)
    return w * _huber(r, kappa)


# ── 1. Численная проверка градиента ─────────────────────────────────────────

@pytest.mark.parametrize('quantile', [0.1, 0.5, 0.9])
@pytest.mark.parametrize('kappa', [0.3, 1.5])
def test_gradient_matches_numeric_finite_difference(quantile, kappa):
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(scale=3.0, size=n)
    f = y + rng.normal(scale=2.0, size=n)

    loss = QuantileHuberLoss(quantile=quantile, kappa=kappa)
    der1, _der2 = zip(*loss.calc_ders_range(f, y, None))
    der1 = np.array(der1)

    eps = 1e-4
    l_minus = _loss_value(f - eps, y, quantile, kappa)
    l_plus = _loss_value(f + eps, y, quantile, kappa)
    numeric_der1 = (l_minus - l_plus) / (2 * eps)

    r = y - f
    mask = np.abs(r) > 1e-2  # excl. точки на изломе huber (|r|=kappa) и на r=0
    mask &= np.abs(np.abs(r) - kappa) > 1e-2
    assert np.allclose(der1[mask], numeric_der1[mask], atol=1e-3), (quantile, kappa)


def test_der2_negative_and_zero_kappa_bounds_rejected():
    with pytest.raises(ValueError):
        QuantileHuberLoss(quantile=0.5, kappa=0.0)
    with pytest.raises(ValueError):
        QuantileHuberLoss(quantile=1.5, kappa=1.0)

    loss = QuantileHuberLoss(quantile=0.5, kappa=1.0)
    rng = np.random.default_rng(0)
    f = rng.normal(size=50)
    y = rng.normal(size=50)
    _, der2 = zip(*loss.calc_ders_range(f, y, None))
    assert np.all(np.array(der2) <= 0)


# ── 2. Смоук fit/predict для разных квантилей ───────────────────────────────

@pytest.mark.parametrize('quantile', [0.1, 0.5, 0.9])
def test_fit_predict_each_quantile(regression_data, quantile):
    X_train, y_train, X_valid, y_valid = regression_data
    model = QuantileHuberRegressor(quantile=quantile, kappa=1.0, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)


def test_constructor_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        QuantileHuberRegressor(quantile=0.0)
    with pytest.raises(ValueError):
        QuantileHuberRegressor(quantile=1.0)


# ── 3. Квантильное поведение: p90 предсказывает выше медианы p50 ───────────

def test_higher_quantile_predicts_higher_values(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    m10 = QuantileHuberRegressor(quantile=0.1, kappa=1.0, base_params=BASE_PARAMS)
    m10.fit(X_train, y_train, X_valid, y_valid)
    m90 = QuantileHuberRegressor(quantile=0.9, kappa=1.0, base_params=BASE_PARAMS)
    m90.fit(X_train, y_train, X_valid, y_valid)

    p10 = m10.predict(X_valid)
    p90 = m90.predict(X_valid)
    # На подавляющем большинстве строк верхний квантиль должен быть выше нижнего
    assert np.mean(p90 >= p10) > 0.9


# ── 4. Optuna: тюнит только kappa (quantile фиксирован), score = pinball ───

@pytest.mark.slow
def test_optuna_tunes_kappa_and_architecture(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = QuantileHuberRegressor(quantile=0.7, n_optuna_trials=4, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.quantile == 0.7  # не тюнится
    assert 0.01 <= model.best_params_['kappa'] <= 5.0
    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.01, 0.2), 'l2_leaf_reg': (1e-3, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])


@pytest.mark.slow
def test_optuna_selection_uses_pinball_not_mae(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = QuantileHuberRegressor(quantile=0.2, n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    expected = quantile_loss(y_valid.values, model.valid_pred_, q=0.2)
    assert model._trial_score(y_valid.values, model.valid_pred_) == pytest.approx(expected)
