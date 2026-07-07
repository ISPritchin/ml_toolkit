"""Тесты param_space/Optuna для TverskyLossClassifier (движок _CustomLossClassifierBase).

Покрывает контракт param_space, описанный в docstring TverskyLossClassifier
и реализованный в _custom_loss_base.py: любой ключ (лосса или архитектуры
CatBoost), отсутствующий в словаре, который вернула param_space, тюнится
дефолтным способом — для лосса по self._loss_spec.param_bounds, для
архитектуры по фиксированным границам внутри _CustomLossClassifierBase._tune.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification.high_pr_auc import TverskyLossClassifier

_ALPHA_BOUNDS = (0.05, 0.95)
_BETA_BOUNDS = (0.05, 0.95)
_ARCH_BOUNDS = {
    'iterations': (300, 1000),
    'max_depth': (3, 7),
    'learning_rate': (0.01, 0.2),
    'l2_leaf_reg': (1e-3, 10.0),
    'subsample': (0.5, 1.0),
    'min_data_in_leaf': (1, 30),
}


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 100
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series((rng.random(n_train) < 0.3).astype(int))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series((rng.random(n_valid) < 0.3).astype(int))
    return X_train, y_train, X_valid, y_valid


def _in_bounds(value: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    return lo <= value <= hi


# ── 1. Optuna не запускается ────────────────────────────────────────────────

def test_no_optuna_uses_constructor_values_directly(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    model = TverskyLossClassifier(
        alpha=0.3, beta=0.7,
        base_params={'iterations': 50, 'verbose': 0, 'random_seed': 42},
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.best_params_['alpha'] == 0.3
    assert model.best_params_['beta'] == 0.7
    assert model.best_params_['iterations'] == 50


# ── 2. Optuna на дефолтном пространстве (лосс + архитектура) ───────────────

def test_optuna_default_space_tunes_loss_and_architecture(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    model = TverskyLossClassifier(n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert _in_bounds(model.best_params_['alpha'], _ALPHA_BOUNDS)
    assert _in_bounds(model.best_params_['beta'], _BETA_BOUNDS)
    for key, bounds in _ARCH_BOUNDS.items():
        assert _in_bounds(model.best_params_[key], bounds), (key, model.best_params_[key])


# ── 3. Optuna, переопределён один параметр лосса ────────────────────────────

def test_optuna_overrides_single_loss_param(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    narrow_alpha = (0.6, 0.65)

    def param_space(trial):
        return {'alpha': trial.suggest_float('alpha', *narrow_alpha)}

    model = TverskyLossClassifier(n_optuna_trials=5, param_space=param_space, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert _in_bounds(model.best_params_['alpha'], narrow_alpha)
    # beta не задан в param_space — должен тюниться дефолтным способом,
    # а не остаться равным конструкторскому 0.7 или выпасть из диапазона.
    assert _in_bounds(model.best_params_['beta'], _BETA_BOUNDS)


# ── 4. Кастомное пространство модели, один параметр лосса не задан ─────────

def test_optuna_model_space_leaves_one_loss_param_default(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    narrow_iterations = (50, 100)
    narrow_alpha = (0.6, 0.65)

    def param_space(trial):
        return {
            'iterations': trial.suggest_int('iterations', *narrow_iterations, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 3),
            'alpha': trial.suggest_float('alpha', *narrow_alpha),
            # beta намеренно не задан
        }

    model = TverskyLossClassifier(n_optuna_trials=5, param_space=param_space, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert _in_bounds(model.best_params_['iterations'], narrow_iterations)
    assert model.best_params_['max_depth'] in (2, 3)
    assert _in_bounds(model.best_params_['alpha'], narrow_alpha)
    # beta не задан в param_space — тюнится дефолтным способом, не фиксируется
    assert _in_bounds(model.best_params_['beta'], _BETA_BOUNDS)


# ── 5. Кастомное пространство модели из одного параметра ───────────────────

def test_optuna_partial_model_space_other_params_still_tuned(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    narrow_iterations = (50, 100)

    def param_space(trial):
        return {'iterations': trial.suggest_int('iterations', *narrow_iterations, step=50)}

    model = TverskyLossClassifier(n_optuna_trials=5, param_space=param_space, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert _in_bounds(model.best_params_['iterations'], narrow_iterations)
    # Остальные параметры архитектуры не заданы в param_space — Optuna
    # продолжает перебирать их по дефолтным границам (не фиксирует, не пропускает).
    for key in ('max_depth', 'learning_rate', 'l2_leaf_reg', 'subsample', 'min_data_in_leaf'):
        assert _in_bounds(model.best_params_[key], _ARCH_BOUNDS[key]), (key, model.best_params_[key])
    assert _in_bounds(model.best_params_['alpha'], _ALPHA_BOUNDS)
    assert _in_bounds(model.best_params_['beta'], _BETA_BOUNDS)
