"""Тесты LogCoshRegressor (ml_toolkit/presets/regression/log_cosh.py)

и корректности градиента LogCoshLoss (ml_toolkit/presets/regression/_losses.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import LogCoshRegressor
from ml_toolkit.presets.regression._losses import LogCoshLoss
from tests.presets.regression.conftest import BASE_PARAMS


# ── 1. Численная проверка градиента ─────────────────────────────────────────

def test_gradient_matches_numeric_finite_difference():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(scale=3.0, size=n)
    f = y + rng.normal(scale=2.0, size=n)

    loss = LogCoshLoss()
    der1, der2 = zip(*loss.calc_ders_range(f, y, None))
    der1, der2 = np.array(der1), np.array(der2)

    eps = 1e-4
    l_minus = np.log(np.cosh(f - eps - y))
    l_plus = np.log(np.cosh(f + eps - y))
    numeric_der1 = (l_minus - l_plus) / (2 * eps)
    assert np.allclose(der1, numeric_der1, atol=1e-3)

    numeric_der2 = (l_plus - 2 * np.log(np.cosh(f - y)) + l_minus) / eps ** 2
    assert np.allclose(der2, -numeric_der2, atol=1e-2)
    assert np.all(der2 <= 0)


def test_weights_scale_derivatives():
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    f = y + rng.normal(scale=1.0, size=50)
    loss = LogCoshLoss()

    unweighted = np.array([d1 for d1, _ in loss.calc_ders_range(f, y, None)])
    weighted = np.array([d1 for d1, _ in loss.calc_ders_range(f, y, np.full(50, 3.0))])
    assert np.allclose(weighted, unweighted * 3.0)


# ── 2. Смоук fit/predict ─────────────────────────────────────────────────────

def test_fit_predict(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = LogCoshRegressor(base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert model.train_pred_.shape == (len(X_train),)
    assert np.allclose(model.valid_pred_, pred)
    assert model.best_params_['iterations'] == 50


# ── 3. Optuna: тюнит только архитектуру ─────────────────────────────────────

def test_optuna_tunes_architecture_only(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = LogCoshRegressor(n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    expected_keys = {
        'iterations', 'max_depth', 'learning_rate', 'l2_leaf_reg',
        'subsample', 'min_data_in_leaf', 'early_stopping_rounds', 'random_seed', 'verbose',
    }
    assert set(model.best_params_) == expected_keys
    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.01, 0.2), 'l2_leaf_reg': (1e-3, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])


# ── 4. Устойчивость к выбросам: LogCosh ближе к RMSE-модели по MAE, чем сама RMSE ──

def test_log_cosh_more_robust_to_outliers_than_rmse(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    y_train_out = y_train.copy()
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(len(y_train_out), size=10, replace=False)
    y_train_out.iloc[outlier_idx] += rng.choice([-1, 1], size=10) * 50.0

    params = {'iterations': 300, 'verbose': 0, 'random_seed': 42}

    from catboost import CatBoostRegressor as _RawCB, Pool
    rmse_model = _RawCB(loss_function='RMSE', **params)
    rmse_model.fit(Pool(X_train, y_train_out), eval_set=Pool(X_valid, y_valid), verbose=False)
    rmse_mae = mean_absolute_error(y_valid, rmse_model.predict(X_valid))

    logcosh_model = LogCoshRegressor(base_params=params)
    logcosh_model.fit(X_train, y_train_out, X_valid, y_valid)
    logcosh_mae = mean_absolute_error(y_valid, logcosh_model.predict(X_valid))

    assert logcosh_mae <= rmse_mae * 1.1
