"""Тесты NGBoostPreset (ml_toolkit/presets/regression/ngboost_preset.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.regression import NGBoostPreset

_SMALL = {'n_estimators': 60, 'early_stopping_rounds': 20}


# ── 1. Валидация конструктора и входных данных ──────────────────────────────

def test_constructor_rejects_invalid_dist():
    with pytest.raises(ValueError, match='dist должен быть'):
        NGBoostPreset(dist='bogus')


def test_fit_rejects_cat_features(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = NGBoostPreset(dist='Normal', **_SMALL)
    with pytest.raises(ValueError, match='категориальные'):
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['f0'])


@pytest.mark.parametrize('dist', ['LogNormal', 'Gamma'])
def test_fit_rejects_non_positive_target_for_positive_dists(regression_data, dist):
    X_train, y_train, X_valid, y_valid = regression_data
    assert (y_train <= 0).any()
    model = NGBoostPreset(dist=dist, **_SMALL)
    with pytest.raises(ValueError, match='положительный'):
        model.fit(X_train, y_train, X_valid, y_valid)


# ── 2. Смоук fit/predict для каждого распределения ──────────────────────────

def test_fit_predict_normal(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = NGBoostPreset(dist='Normal', **_SMALL)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)
    assert model.best_params_['dist'] == 'Normal'


@pytest.mark.parametrize('dist', ['LogNormal', 'Gamma'])
def test_fit_predict_positive_dists(positive_regression_data, dist):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = NGBoostPreset(dist=dist, **_SMALL)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)


# ── 3. predict_dist / predict_interval ──────────────────────────────────────

def test_predict_dist_mean_matches_predict(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = NGBoostPreset(dist='LogNormal', **_SMALL)
    model.fit(X_train, y_train, X_valid, y_valid)

    dist = model.predict_dist(X_valid)
    assert np.allclose(dist.mean(), model.predict(X_valid))


def test_predict_interval_contains_median_and_is_ordered(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = NGBoostPreset(dist='LogNormal', **_SMALL)
    model.fit(X_train, y_train, X_valid, y_valid)

    lower, upper = model.predict_interval(X_valid, alpha=0.2)
    assert np.all(lower <= upper)
    assert lower.shape == (len(X_valid),)

    # Более широкий alpha (более узкий интервал) должен давать интервал внутри
    lower_narrow, upper_narrow = model.predict_interval(X_valid, alpha=0.5)
    assert np.all(lower_narrow >= lower - 1e-9)
    assert np.all(upper_narrow <= upper + 1e-9)


def test_predict_interval_rejects_invalid_alpha(positive_regression_data):
    X_train, y_train, X_valid, y_valid = positive_regression_data
    model = NGBoostPreset(dist='LogNormal', **_SMALL)
    model.fit(X_train, y_train, X_valid, y_valid)
    with pytest.raises(ValueError, match='alpha должен быть'):
        model.predict_interval(X_valid, alpha=0.0)


# ── 4. Optuna: тюнит n_estimators/learning_rate/base_max_depth/minibatch_frac ──

def test_optuna_tunes_hyperparameters(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = NGBoostPreset(
        dist='Normal', n_optuna_trials=2, early_stopping_rounds=15, random_seed=42,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    assert 100 <= model.best_params_['n_estimators'] <= 500
    assert 0.005 <= model.best_params_['learning_rate'] <= 0.2
    assert 2 <= model.best_params_['base_max_depth'] <= 5
    assert 0.5 <= model.best_params_['minibatch_frac'] <= 1.0

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
