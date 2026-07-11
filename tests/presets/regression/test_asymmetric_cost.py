"""Тесты AsymmetricCostRegressor (ml_toolkit/presets/regression/asymmetric_cost.py).

и корректности градиента AsymmetricMSELoss (ml_toolkit/presets/regression/_losses.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.regression import AsymmetricCostRegressor
from ml_toolkit.presets.regression._losses import AsymmetricMSELoss
from ml_toolkit.presets.regression.asymmetric_cost import _asymmetric_cost
from tests.presets.regression.conftest import BASE_PARAMS

# ── 1. Численная проверка градиента ─────────────────────────────────────────

def test_gradient_matches_numeric_finite_difference():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(scale=3.0, size=n)
    f = y + rng.normal(scale=2.0, size=n)
    over_cost, under_cost = 1.0, 3.0

    loss = AsymmetricMSELoss(over_cost=over_cost, under_cost=under_cost)
    der1, der2 = zip(*loss.calc_ders_range(f, y, None), strict=False)
    der1, der2 = np.array(der1), np.array(der2)

    def L(fv):
        r = fv - y
        cost = np.where(r > 0, over_cost, under_cost)
        return cost * r * r

    eps = 1e-4
    numeric_der1 = (L(f - eps) - L(f + eps)) / (2 * eps)
    mask = np.abs(f - y) > 1e-2
    assert np.allclose(der1[mask], numeric_der1[mask], atol=1e-2)
    assert np.all(der2 < 0)


def test_asymmetry_scales_gradient_correctly():
    y = np.array([0.0, 0.0])
    f = np.array([1.0, -1.0])  # первая точка — over-forecast, вторая — under-forecast
    loss = AsymmetricMSELoss(over_cost=1.0, under_cost=4.0)
    der1, _ = zip(*loss.calc_ders_range(f, y, None), strict=False)
    # der1 = -2*cost*r; |der1| над over-точкой должен быть меньше, чем над under
    assert abs(der1[1]) > abs(der1[0])


def test_constructor_rejects_non_positive_costs():
    with pytest.raises(ValueError, match='должны быть положительными'):
        AsymmetricMSELoss(over_cost=0.0, under_cost=1.0)


# ── 2. Валидация конструктора пресета ───────────────────────────────────────

def test_preset_rejects_invalid_loss_name():
    with pytest.raises(ValueError, match='loss должен быть'):
        AsymmetricCostRegressor(loss='bogus')


def test_preset_rejects_non_positive_costs():
    with pytest.raises(ValueError, match='должны быть положительными'):
        AsymmetricCostRegressor(over_cost=-1.0)


# ── 3. Смоук fit/predict для обоих режимов ──────────────────────────────────

@pytest.mark.parametrize('loss', ['pinball', 'asym_mse'])
def test_fit_predict_both_modes(regression_data, loss):
    X_train, y_train, X_valid, y_valid = regression_data
    model = AsymmetricCostRegressor(loss=loss, over_cost=1.0, under_cost=3.0, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)


# ── 4. Асимметрия действительно смещает прогноз в дорогую сторону ──────────

@pytest.mark.parametrize('loss', ['pinball', 'asym_mse'])
def test_high_under_cost_biases_predictions_upward(regression_data, loss):
    """under_cost >> over_cost: недопрогноз намного дороже — модель должна.

    систематически прогнозировать выше символичной "нейтральной" MAE-модели,
    чтобы реже промахиваться в дорогую сторону.
    """
    X_train, y_train, X_valid, y_valid = regression_data
    params = {'iterations': 300, 'verbose': 0, 'random_seed': 42}

    from catboost import CatBoostRegressor as _RawCB
    from catboost import Pool
    neutral = _RawCB(loss_function='MAE', **params)
    neutral.fit(Pool(X_train, y_train), eval_set=Pool(X_valid, y_valid), verbose=False)
    neutral_pred = neutral.predict(X_valid)

    model = AsymmetricCostRegressor(loss=loss, over_cost=1.0, under_cost=8.0, base_params=params)
    model.fit(X_train, y_train, X_valid, y_valid)
    asym_pred = model.predict(X_valid)

    assert np.mean(asym_pred) > np.mean(neutral_pred)


# ── 5. Optuna: тюнит только архитектуру, score = линейная asymmetric cost ──

def test_optuna_tunes_architecture_score_is_asymmetric_cost(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = AsymmetricCostRegressor(
        loss='pinball', over_cost=1.0, under_cost=3.0, n_optuna_trials=3, random_seed=42,
    )
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

    expected_score = _asymmetric_cost(y_valid.values, model.valid_pred_, 1.0, 3.0)
    assert model._trial_score(y_valid.values, model.valid_pred_) == pytest.approx(expected_score)
