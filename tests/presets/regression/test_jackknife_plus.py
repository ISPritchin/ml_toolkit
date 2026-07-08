"""Тесты JackknifePlusRegressor (ml_toolkit/presets/regression/jackknife_plus.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.regression import JackknifePlusRegressor

_SMALL_PARAMS = {'iterations': 80, 'depth': 4, 'verbose': 0}


def _make_data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    cols = [f'f{i}' for i in range(5)]
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    y = pd.Series(X['f0'] * 2.0 - X['f1'] + rng.normal(scale=0.5, size=n))
    return X, y


@pytest.fixture
def holdout_data():
    return _make_data(99, 300)


# ── 1. Валидация конструктора ───────────────────────────────────────────────

def test_constructor_rejects_too_few_folds():
    with pytest.raises(ValueError):
        JackknifePlusRegressor(n_folds=1)


def test_fit_rejects_n_folds_exceeding_rows():
    X, y = _make_data(0, 5)
    model = JackknifePlusRegressor(n_folds=10, base_params=_SMALL_PARAMS)
    with pytest.raises(ValueError, match='n_folds'):
        model.fit(X, y)


# ── 2. Смоук fit/predict без X_valid ────────────────────────────────────────

def test_fit_predict_without_valid(regression_data):
    X_train, y_train, _, _ = regression_data
    model = JackknifePlusRegressor(n_folds=5, base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train)

    assert len(model.models_) == 5
    assert model.fold_id_.shape == (len(X_train),)
    assert model.abs_residuals_.shape == (len(X_train),)
    assert model.valid_pred_ is None
    assert model.train_pred_.shape == (len(X_train),)

    pred = model.predict(X_train)
    assert pred.shape == (len(X_train),)
    assert np.all(np.isfinite(pred))


# ── 3. X_valid, если передан, добавляется в общий пул перед K-fold ─────────

def test_fit_folds_in_valid_when_provided(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = JackknifePlusRegressor(n_folds=5, base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.fold_id_.shape == (len(X_train) + len(X_valid),)
    assert model.valid_pred_ is not None
    assert model.valid_pred_.shape == (len(X_valid),)


# ── 4. predict_interval: базовая корректность ───────────────────────────────

def test_predict_interval_shape_and_ordering(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = JackknifePlusRegressor(n_folds=5, base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train)

    lower, upper = model.predict_interval(X_valid, alpha=0.1)
    assert lower.shape == (len(X_valid),)
    assert np.all(lower <= upper + 1e-9)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))


def test_predict_interval_rejects_invalid_alpha(regression_data):
    X_train, y_train, _, _ = regression_data
    model = JackknifePlusRegressor(n_folds=5, base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train)
    with pytest.raises(ValueError):
        model.predict_interval(X_train, alpha=0.0)


def test_looser_alpha_gives_narrower_interval(regression_data):
    X_train, y_train, X_valid, _ = regression_data
    model = JackknifePlusRegressor(n_folds=5, base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train)

    lo_tight, hi_tight = model.predict_interval(X_valid, alpha=0.05)
    lo_loose, hi_loose = model.predict_interval(X_valid, alpha=0.4)
    assert (hi_loose - lo_loose).mean() < (hi_tight - lo_tight).mean()


# ── 5. Эмпирическое покрытие на независимой выборке близко к (1 - alpha) ───

def test_empirical_coverage_close_to_target_on_holdout(regression_data, holdout_data):
    X_train, y_train, _, _ = regression_data
    X_hold, y_hold = holdout_data
    alpha = 0.2

    model = JackknifePlusRegressor(n_folds=8, base_params=_SMALL_PARAMS, random_seed=42)
    model.fit(X_train, y_train)

    lower, upper = model.predict_interval(X_hold, alpha=alpha)
    covered = (y_hold.values >= lower) & (y_hold.values <= upper)
    coverage = covered.mean()
    assert coverage >= (1 - alpha) - 0.12
