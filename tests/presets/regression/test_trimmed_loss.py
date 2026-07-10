"""Тесты TrimmedLossRegressor (ml_toolkit/presets/regression/trimmed_loss.py)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import TrimmedLossRegressor
from tests.presets.regression.conftest import BASE_PARAMS


@pytest.fixture
def outlier_regression_data(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    y_train = y_train.copy()
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(len(y_train), size=15, replace=False)
    y_train.iloc[outlier_idx] += rng.choice([-1, 1], size=15) * 50.0
    return X_train, y_train, X_valid, y_valid


# ── 1. Валидация конструктора ───────────────────────────────────────────────

def test_constructor_rejects_invalid_trim_frac():
    with pytest.raises(ValueError):
        TrimmedLossRegressor(trim_frac=0.0)
    with pytest.raises(ValueError):
        TrimmedLossRegressor(trim_frac=0.5)


def test_constructor_rejects_invalid_n_rounds():
    with pytest.raises(ValueError):
        TrimmedLossRegressor(n_rounds=0)


# ── 2. Смоук fit/predict, динамика раундов ──────────────────────────────────

def test_fit_predict_and_round_bookkeeping(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = TrimmedLossRegressor(trim_frac=0.05, n_rounds=3, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)

    assert len(model.mae_per_round_) == 3
    assert len(model.active_frac_per_round_) == 3
    assert len(model.models_) == 3
    assert model.active_frac_per_round_[0] == pytest.approx(1.0)
    # После раунда 0 активное множество урезано ровно на trim_frac
    assert model.active_frac_per_round_[1] == pytest.approx(0.95, abs=0.02)

    best_round = int(np.argmin(model.mae_per_round_))
    assert model._model is model.models_[best_round]


# ── 3. Трimming реально повышает робастность к выбросам в таргете ──────────

def test_trimming_beats_single_shot_on_outliers(outlier_regression_data):
    X_train, y_train, X_valid, y_valid = outlier_regression_data
    params = {'iterations': 100, 'verbose': 0, 'random_seed': 42}

    from catboost import CatBoostRegressor as _RawCB, Pool
    baseline = _RawCB(loss_function='RMSE', **params)
    baseline.fit(Pool(X_train, y_train), eval_set=Pool(X_valid, y_valid), verbose=False)
    baseline_mae = mean_absolute_error(y_valid, baseline.predict(X_valid))

    model = TrimmedLossRegressor(trim_frac=0.05, n_rounds=3, base_params=params)
    model.fit(X_train, y_train, X_valid, y_valid)
    model_mae = mean_absolute_error(y_valid, model.predict(X_valid))

    assert model_mae < baseline_mae * 0.8
    # Раунд 0 (нетронутые данные) должен совпадать с однократной моделью,
    # последующие раунды — заметно лучше него.
    assert model.mae_per_round_[0] == pytest.approx(baseline_mae, rel=1e-6)
    assert min(model.mae_per_round_[1:]) < model.mae_per_round_[0]


# ── 4. n_rounds=1 — эквивалент однократного обучения без trimming ──────────

def test_single_round_matches_plain_catboost(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = TrimmedLossRegressor(trim_frac=0.1, n_rounds=1, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    from catboost import CatBoostRegressor as _RawCB, Pool
    baseline = _RawCB(loss_function='RMSE', **{**BASE_PARAMS, 'eval_metric': 'RMSE'})
    baseline.fit(Pool(X_train, y_train), eval_set=Pool(X_valid, y_valid), verbose=False)

    assert np.allclose(model.predict(X_valid), baseline.predict(X_valid), atol=1e-6)


# ── 5. Optuna в раунде 0 тюнит архитектуру ──────────────────────────────────

@pytest.mark.slow
def test_optuna_round0_tunes_architecture(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = TrimmedLossRegressor(trim_frac=0.05, n_rounds=2, n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.001, 0.3), 'l2_leaf_reg': (1e-5, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
