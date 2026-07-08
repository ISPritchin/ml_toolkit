"""Тесты TweedieOptunaRegressor (ml_toolkit/presets/regression/tweedie_optuna.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.regression import TweedieOptunaRegressor
from tests.presets.regression.conftest import BASE_PARAMS


@pytest.fixture
def zero_inflated_data():
    """Compound Poisson-Gamma-подобный таргет: часть строк — точный 0."""
    rng = np.random.default_rng(2)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]

    def make(n):
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        occurs = rng.random(n) < (0.3 + 0.4 / (1 + np.exp(-X['f0'])))
        severity = rng.gamma(shape=2.0, scale=np.exp(0.3 * X['f1']))
        y = pd.Series(np.where(occurs, severity, 0.0))
        return X, y

    X_train, y_train = make(n_train)
    X_valid, y_valid = make(n_valid)
    return X_train, y_train, X_valid, y_valid


# ── 1. Валидация конструктора и входных данных ──────────────────────────────

def test_constructor_rejects_power_out_of_range():
    with pytest.raises(ValueError):
        TweedieOptunaRegressor(power=1.0)
    with pytest.raises(ValueError):
        TweedieOptunaRegressor(power=2.0)


def test_fit_rejects_negative_target(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    assert (y_train < 0).any()
    model = TweedieOptunaRegressor(base_params=BASE_PARAMS)
    with pytest.raises(ValueError, match='неотрицательный'):
        model.fit(X_train, y_train, X_valid, y_valid)


# ── 2. Смоук fit/predict ─────────────────────────────────────────────────────

def test_fit_predict_positive_target(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TweedieOptunaRegressor(power=1.5, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)  # Tweedie использует log-link — прогноз структурно неотрицателен
    assert np.allclose(model.valid_pred_, pred)


def test_fit_predict_zero_inflated(zero_inflated_data):
    X_train, y_train, X_valid, y_valid = zero_inflated_data
    model = TweedieOptunaRegressor(power=1.5, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)


def test_zero_target_allowed(positive_regression_data):
    """y == 0 (не только y < 0) должен быть допустим — Tweedie(1,2) поддерживает массу в нуле."""
    X_train, y_train, X_valid, y_valid = positive_regression_data
    y_train = y_train.copy()
    y_train.iloc[:5] = 0.0
    model = TweedieOptunaRegressor(power=1.5, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)  # не должно поднять исключение
    assert model._model is not None


# ── 3. Optuna: power + архитектура ──────────────────────────────────────────

def test_optuna_tunes_power_and_architecture(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TweedieOptunaRegressor(n_optuna_trials=4, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert 1.01 <= model.best_params_['power'] <= 1.99
    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.01, 0.2), 'l2_leaf_reg': (1e-3, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)


def test_optuna_overrides_power_via_param_space(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    narrow = (1.2, 1.3)

    def param_space(trial):
        return {'power': trial.suggest_float('power', *narrow)}

    model = TweedieOptunaRegressor(n_optuna_trials=5, param_space=param_space, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)
    assert narrow[0] <= model.best_params_['power'] <= narrow[1]
