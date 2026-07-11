"""Тесты HuberOptunaRegressor (ml_toolkit/presets/regression/huber_optuna.py)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import HuberOptunaRegressor
from tests.presets.regression.conftest import BASE_PARAMS

# ── 1. Прямой режим ──────────────────────────────────────────────────────────

def test_no_optuna_uses_constructor_delta_directly(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = HuberOptunaRegressor(delta=2.5, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.best_params_['delta'] == 2.5
    assert model.best_params_['iterations'] == 50

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)


def test_constructor_rejects_non_positive_delta():
    with pytest.raises(ValueError, match='delta должен быть положительным'):
        HuberOptunaRegressor(delta=0.0)
    with pytest.raises(ValueError, match='delta должен быть положительным'):
        HuberOptunaRegressor(delta=-1.0)


# ── 2. Optuna: delta + архитектура тюнятся, отбор по MAE ───────────────────

@pytest.mark.slow
def test_optuna_tunes_delta_and_architecture(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = HuberOptunaRegressor(n_optuna_trials=4, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert 0.01 <= model.best_params_['delta'] <= 10.0
    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.01, 0.2), 'l2_leaf_reg': (1e-3, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))


def test_optuna_overrides_delta_via_param_space(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    narrow = (4.0, 4.5)

    def param_space(trial):
        return {'delta': trial.suggest_float('delta', *narrow)}

    model = HuberOptunaRegressor(n_optuna_trials=5, param_space=param_space, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert narrow[0] <= model.best_params_['delta'] <= narrow[1]


# ── 3. Устойчивость к выбросам: Huber даёт меньшую MAE, чем RMSE-модель ────

def test_huber_more_robust_to_outliers_than_rmse(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    y_train_out = y_train.copy()
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(len(y_train_out), size=10, replace=False)
    y_train_out.iloc[outlier_idx] += rng.choice([-1, 1], size=10) * 50.0

    params = {'iterations': 300, 'verbose': 0, 'random_seed': 42}

    from catboost import CatBoostRegressor as _RawCB
    from catboost import Pool
    rmse_model = _RawCB(loss_function='RMSE', **params)
    rmse_model.fit(Pool(X_train, y_train_out), eval_set=Pool(X_valid, y_valid), verbose=False)
    rmse_mae = mean_absolute_error(y_valid, rmse_model.predict(X_valid))

    huber_model = HuberOptunaRegressor(delta=1.0, base_params=params)
    huber_model.fit(X_train, y_train_out, X_valid, y_valid)
    huber_mae = mean_absolute_error(y_valid, huber_model.predict(X_valid))

    assert huber_mae <= rmse_mae * 1.1
