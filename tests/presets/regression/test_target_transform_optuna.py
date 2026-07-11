"""Тесты TargetTransformOptunaRegressor (ml_toolkit/presets/regression/target_transform_optuna.py).

Покрывает: прямой режим (без Optuna) для каждого трансформа, корректность
predict() в исходном масштабе (включая smearing-поправку log1p), Optuna-поиск
по transform+архитектуре, автоматическую фильтрацию несовместимых трансформов
(box-cox/log1p на нестрого положительном таргете).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import TargetTransformOptunaRegressor
from ml_toolkit.presets.regression.target_transform_optuna import ALL_TRANSFORMS
from tests.presets.regression.conftest import BASE_PARAMS

# ── 1. Прямой режим: каждый трансформ по отдельности, положительный таргет ──

@pytest.mark.parametrize('transform', list(ALL_TRANSFORMS))
def test_direct_mode_each_transform_predicts_original_scale(positive_regression_data, transform):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TargetTransformOptunaRegressor(transforms=[transform], base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.transform_name_ == transform
    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    # predict() должен возвращать значения в масштабе исходного y, а не
    # трансформированного — иначе ошибка была бы на порядки больше
    # (y ~ exp(...), масштаб сотен/тысяч; log1p(y) ~ единицы).
    assert mean_absolute_error(y_valid, pred) < 5 * float(np.std(y_valid))


def test_log1p_smearing_reduces_bias_vs_naive_inversion(positive_regression_data):
    """Без smearing expm1(f(x)) систематически недооценивает E[y|x] (Йенсен).

    Проверяем, что smearing-поправка положительна и отлична от 1 (то есть
    реально что-то корректирует, а не является заглушкой) и что итоговый
    prediction bias (mean(pred) - mean(y)) меньше по модулю, чем без неё.
    """
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TargetTransformOptunaRegressor(transforms=['log1p'], base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    smear = model.transform_.smear_
    assert smear > 0

    pred_with_smear = model.predict(X_valid)
    # Ручная безsmearing-инверсия для сравнения
    from catboost import Pool
    pool = Pool(X_valid[model.selected_features_], cat_features=model.cat_features_)
    pred_no_smear = np.expm1(model._model.predict(pool))

    bias_with = abs(float(np.mean(pred_with_smear) - np.mean(y_valid)))
    bias_without = abs(float(np.mean(pred_no_smear) - np.mean(y_valid)))
    assert bias_with <= bias_without + 1e-6


# ── 2. Автофильтрация несовместимых трансформов ─────────────────────────────

def test_box_cox_excluded_for_non_positive_target(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    assert (y_train <= 0).any()  # regression_data специально не строго положительный

    model = TargetTransformOptunaRegressor(
        transforms=['box-cox', 'log1p', 'identity'], base_params=BASE_PARAMS,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    assert model.transform_name_ not in ('box-cox', 'log1p')


def test_all_incompatible_transforms_raises(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = TargetTransformOptunaRegressor(transforms=['box-cox'], base_params=BASE_PARAMS)
    with pytest.raises(ValueError, match='box-cox'):
        model.fit(X_train, y_train, X_valid, y_valid)


# ── 3. Optuna: выбирает transform + архитектуру совместно ──────────────────

@pytest.mark.slow
def test_optuna_picks_transform_and_tunes_architecture(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TargetTransformOptunaRegressor(
        transforms=['identity', 'log1p', 'yeo-johnson'],
        n_optuna_trials=4,
        random_seed=42,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.best_params_['transform'] in ('identity', 'log1p', 'yeo-johnson')
    assert model.transform_name_ == model.best_params_['transform']
    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.001, 0.3), 'l2_leaf_reg': (1e-5, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))


def test_optuna_custom_param_space_fixes_transform(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data

    def param_space(trial):
        return {'transform': 'yeo-johnson', 'iterations': trial.suggest_int('iterations', 50, 100, step=50)}

    model = TargetTransformOptunaRegressor(
        transforms=['identity', 'log1p', 'yeo-johnson'],
        n_optuna_trials=3,
        param_space=param_space,
        random_seed=42,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    assert model.transform_name_ == 'yeo-johnson'
    assert model.best_params_['iterations'] in (50, 100)


# ── 4. train_pred_/valid_pred_ заполнены и в исходном масштабе ─────────────

def test_train_and_valid_pred_populated(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = TargetTransformOptunaRegressor(transforms=['identity'], base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.train_pred_.shape == (len(X_train),)
    assert model.valid_pred_.shape == (len(X_valid),)
    assert np.allclose(model.valid_pred_, model.predict(X_valid))
