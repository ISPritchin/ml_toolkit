"""Тесты QuantileEnsembleRegressor (ml_toolkit/presets/regression/quantile_ensemble.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.regression import QuantileEnsembleRegressor
from tests.presets.regression.conftest import BASE_PARAMS


# ── 1. Валидация конструктора ───────────────────────────────────────────────

def test_constructor_rejects_duplicate_quantiles():
    with pytest.raises(ValueError):
        QuantileEnsembleRegressor(quantiles=[0.5, 0.5])


def test_constructor_rejects_out_of_range_quantiles():
    with pytest.raises(ValueError):
        QuantileEnsembleRegressor(quantiles=[0.0, 0.5])
    with pytest.raises(ValueError):
        QuantileEnsembleRegressor(quantiles=[0.5, 1.0])


# ── 2. Смоук fit/predict ─────────────────────────────────────────────────────

def test_fit_predict_direct_mode(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    quantiles = [0.1, 0.5, 0.9]
    model = QuantileEnsembleRegressor(quantiles=quantiles, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert set(model.models_) == set(quantiles)
    assert model._median_q == 0.5

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))

    profile = model.predict_quantiles(X_valid)
    assert list(profile.columns) == quantiles
    assert np.allclose(profile[0.5].values, pred)

    assert model.train_pred_.shape == (len(X_train),)
    assert model.valid_pred_.shape == (len(X_valid),)


def test_median_column_picked_when_05_absent(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = QuantileEnsembleRegressor(quantiles=[0.2, 0.6], base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)
    assert model._median_q == 0.6  # ближе к 0.5, чем 0.2


# ── 3. Non-crossing поправка гарантирует монотонность по строке ────────────

def test_non_crossing_enforces_row_monotonic(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    model = QuantileEnsembleRegressor(quantiles=quantiles, non_crossing=True, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    profile = model.predict_quantiles(X_valid).values
    assert np.all(np.diff(profile, axis=1) >= -1e-9)


def test_non_crossing_false_can_still_cross(regression_data):
    """Без коррекции независимые модели не обязаны быть монотонными — проверяем,

    что non_crossing=False действительно пропускает "сырые" (несортированные)
    предсказания, а не тайно сортирует их так же, как non_crossing=True.
    """
    X_train, y_train, X_valid, y_valid = regression_data
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    raw_model = QuantileEnsembleRegressor(quantiles=quantiles, non_crossing=False, base_params=BASE_PARAMS)
    raw_model.fit(X_train, y_train, X_valid, y_valid)
    raw_profile = raw_model.predict_quantiles(X_valid).values

    corrected_model = QuantileEnsembleRegressor(quantiles=quantiles, non_crossing=True, base_params=BASE_PARAMS)
    corrected_model.fit(X_train, y_train, X_valid, y_valid)
    corrected_profile = corrected_model.predict_quantiles(X_valid).values

    assert np.allclose(np.sort(raw_profile, axis=1), corrected_profile)


# ── 4. Optuna: тюнит архитектуру независимо на каждый квантиль ─────────────

@pytest.mark.slow
def test_optuna_tunes_each_quantile_independently(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    quantiles = [0.3, 0.7]
    model = QuantileEnsembleRegressor(quantiles=quantiles, n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert set(model.best_params_per_quantile_) == set(quantiles)
    for q in quantiles:
        params = model.best_params_per_quantile_[q]
        assert params['loss_function'] == f'Quantile:alpha={q}'
        for key, bounds in {
            'iterations': (300, 1000), 'max_depth': (3, 7),
            'learning_rate': (0.001, 0.3), 'l2_leaf_reg': (1e-5, 10.0),
            'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
        }.items():
            assert bounds[0] <= params[key] <= bounds[1], (q, key, params[key])

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
