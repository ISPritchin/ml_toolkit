"""Тесты RelativeErrorRegressor (ml_toolkit/presets/regression/relative_error.py)

и корректности градиента RelativeErrorLoss (ml_toolkit/presets/regression/_losses.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import RelativeErrorRegressor
from ml_toolkit.presets.regression._losses import RelativeErrorLoss
from ml_toolkit.presets.regression.relative_error import _relative_score
from tests.presets.regression.conftest import BASE_PARAMS


def _loss_value(metric: str, f: np.ndarray, y: np.ndarray, floor: float, global_denom: float | None) -> np.ndarray:
    e = np.abs(f - y)
    if metric == 'mape':
        return e / np.maximum(np.abs(y), floor)
    if metric == 'wape':
        return e / global_denom
    return 2.0 * e / (np.abs(y) + np.abs(f) + floor)


# ── 1. Численная проверка градиента (finite differences) ───────────────────

@pytest.mark.parametrize('metric', ['mape', 'smape', 'wape'])
def test_gradient_matches_numeric_finite_difference(metric):
    rng = np.random.default_rng(0)
    n = 200
    y = rng.uniform(1.0, 20.0, size=n)
    f = y + rng.normal(scale=2.0, size=n)
    floor = 1.0

    loss = RelativeErrorLoss(metric=metric, denom_floor=floor)
    if metric == 'wape':
        loss.global_denom = max(float(np.mean(np.abs(y))), floor)
    der1, der2 = zip(*loss.calc_ders_range(f, y, None))
    der1 = np.array(der1)

    eps = 1e-4
    gd = loss.global_denom if metric == 'wape' else None
    l_plus = _loss_value(metric, f + eps, y, floor, gd)
    l_minus = _loss_value(metric, f - eps, y, floor, gd)
    numeric_dL_df = (l_plus - l_minus) / (2 * eps)
    numeric_der1 = -numeric_dL_df

    # |y-f| не гладкая в нуле — исключаем точки с |residual| < eps из сравнения
    mask = np.abs(f - y) > 1e-2
    assert np.allclose(der1[mask], numeric_der1[mask], atol=1e-3), metric


def test_weights_scale_derivatives():
    rng = np.random.default_rng(1)
    y = rng.uniform(1.0, 10.0, size=50)
    f = y + rng.normal(scale=1.0, size=50)
    loss = RelativeErrorLoss(metric='mape', denom_floor=1.0)

    unweighted = np.array([d1 for d1, _ in loss.calc_ders_range(f, y, None)])
    weights = np.full(50, 2.0)
    weighted = np.array([d1 for d1, _ in loss.calc_ders_range(f, y, weights)])
    assert np.allclose(weighted, unweighted * 2.0)


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        RelativeErrorLoss(metric='bogus')


# ── 2. Смоук fit/predict для каждой метрики ─────────────────────────────────

@pytest.mark.parametrize('metric', ['mape', 'smape', 'wape'])
def test_fit_predict_each_metric(regression_data, metric):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RelativeErrorRegressor(metric=metric, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert model.train_pred_.shape == (len(X_train),)
    assert model.valid_pred_.shape == (len(X_valid),)
    assert np.allclose(model.valid_pred_, pred)


def test_constructor_rejects_invalid_metric():
    with pytest.raises(ValueError):
        RelativeErrorRegressor(metric='bogus')


# ── 3. Обучение на WAPE даёт сравнимый с MAE-бейзлайном WAPE ────────────────

def test_wape_training_converges_close_to_mae_baseline_on_wape(regression_data):
    """Кастомный Python-лосс использует leaf_estimation_method='Newton' (дефолт

    CatBoost для произвольных лоссов), тогда как встроенный 'MAE' — быстро
    сходящийся 'Exact' (медианная оценка листа) — при равном числе итераций
    Newton сходится медленнее. Поэтому сравниваем на достаточном числе
    итераций (не на BASE_PARAMS с iterations=50, где разрыв — это просто
    недообучение Newton-версии, а не признак сломанного лосса).
    """
    X_train, y_train, X_valid, y_valid = regression_data
    from catboost import CatBoostRegressor as _RawCB, Pool

    params = {'iterations': 300, 'verbose': 0, 'random_seed': 42}
    baseline = _RawCB(loss_function='MAE', **params)
    baseline.fit(Pool(X_train, y_train), eval_set=Pool(X_valid, y_valid), verbose=False)
    baseline_wape = _relative_score(y_valid.values, baseline.predict(X_valid), 'wape', 1.0)

    model = RelativeErrorRegressor(metric='wape', denom_floor=1.0, base_params=params)
    model.fit(X_train, y_train, X_valid, y_valid)
    model_wape = _relative_score(y_valid.values, model.predict(X_valid), 'wape', 1.0)

    assert model_wape <= baseline_wape * 1.25


# ── 4. Optuna: тюнит только архитектуру (у лосса нет param_bounds) ─────────

@pytest.mark.slow
def test_optuna_tunes_architecture_only(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RelativeErrorRegressor(metric='mape', n_optuna_trials=3, random_seed=42)
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


def test_wape_global_denom_computed_from_train(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RelativeErrorRegressor(metric='wape', denom_floor=1.0, base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    expected = max(float(np.mean(np.abs(y_train.values))), 1.0)
    built = model._build_loss({}, tr_pool=type('P', (), {'get_label': lambda self: y_train.values})())
    assert built.global_denom == pytest.approx(expected)
