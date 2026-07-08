"""Тесты ConformalRegressionWrapper (ml_toolkit/presets/regression/conformal_wrapper.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.models import CatBoostRegressor
from ml_toolkit.presets.regression import ConformalRegressionWrapper


def _make_data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    cols = [f'f{i}' for i in range(5)]
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    # Гетероскедастичность: дисперсия шума растёт с |f2| — нужна для проверки
    # score='normalized' (интервал должен сужаться/расширяться по x).
    noise_scale = 0.3 + 1.5 * np.abs(X['f2'])
    y = pd.Series(X['f0'] * 2.0 - X['f1'] + rng.normal(scale=noise_scale, size=n))
    return X, y


@pytest.fixture
def hetero_data():
    X_train, y_train = _make_data(0, 400)
    X_valid, y_valid = _make_data(1, 150)
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def holdout_data():
    """Независимая от калибровки выборка — для честной проверки покрытия."""
    return _make_data(2, 500)


def _base() -> CatBoostRegressor:
    return CatBoostRegressor(params={'iterations': 200, 'depth': 4, 'verbose': 0, 'random_seed': 42})


# ── 1. Валидация конструктора ───────────────────────────────────────────────

def test_constructor_rejects_invalid_score():
    with pytest.raises(ValueError):
        ConformalRegressionWrapper(_base(), score='bogus')


def test_constructor_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalRegressionWrapper(_base(), alpha=0.0)
    with pytest.raises(ValueError):
        ConformalRegressionWrapper(_base(), alpha=1.0)


# ── 2. Смоук fit/predict, оба режима score ──────────────────────────────────

@pytest.mark.parametrize('score', ['absolute', 'normalized'])
def test_fit_predict_and_interval(hetero_data, score):
    X_train, y_train, X_valid, y_valid = hetero_data
    model = ConformalRegressionWrapper(_base(), alpha=0.1, score=score)
    model.fit(X_train, y_train, X_valid, y_valid)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))

    lower, upper = model.predict_interval(X_valid)
    assert np.all(lower <= upper + 1e-9)
    assert lower.shape == (len(X_valid),)
    assert model.q_hat_ > 0
    assert model.best_params_['score'] == score


# ── 3. normalized-интервал варьируется по x, absolute — постоянной ширины ──

def test_normalized_interval_width_varies_absolute_does_not(hetero_data):
    X_train, y_train, X_valid, y_valid = hetero_data

    abs_model = ConformalRegressionWrapper(_base(), alpha=0.1, score='absolute')
    abs_model.fit(X_train, y_train, X_valid, y_valid)
    lo_a, hi_a = abs_model.predict_interval(X_valid)
    width_abs = hi_a - lo_a
    assert np.allclose(width_abs, width_abs[0])  # постоянная ширина = 2*q_hat

    norm_model = ConformalRegressionWrapper(_base(), alpha=0.1, score='normalized')
    norm_model.fit(X_train, y_train, X_valid, y_valid)
    lo_n, hi_n = norm_model.predict_interval(X_valid)
    width_norm = hi_n - lo_n
    assert np.std(width_norm) > 1e-6  # varies across rows
    # Ширина должна коррелировать с |f2| (источник гетероскедастичности)
    corr = np.corrcoef(width_norm, np.abs(X_valid['f2'].values))[0, 1]
    assert corr > 0.2


# ── 4. Более широкий alpha => более узкий интервал ──────────────────────────

def test_looser_alpha_gives_narrower_interval(hetero_data):
    X_train, y_train, X_valid, y_valid = hetero_data
    model = ConformalRegressionWrapper(_base(), alpha=0.1, score='absolute')
    model.fit(X_train, y_train, X_valid, y_valid)

    lo_tight, hi_tight = model.predict_interval(X_valid, alpha=0.05)
    lo_loose, hi_loose = model.predict_interval(X_valid, alpha=0.4)
    assert (hi_loose - lo_loose)[0] < (hi_tight - lo_tight)[0]


def test_predict_interval_rejects_invalid_alpha(hetero_data):
    X_train, y_train, X_valid, y_valid = hetero_data
    model = ConformalRegressionWrapper(_base(), alpha=0.1)
    model.fit(X_train, y_train, X_valid, y_valid)
    with pytest.raises(ValueError):
        model.predict_interval(X_valid, alpha=1.5)


# ── 5. Эмпирическое покрытие на независимой выборке близко к (1 - alpha) ───

@pytest.mark.parametrize('score', ['absolute', 'normalized'])
def test_empirical_coverage_close_to_target_on_holdout(hetero_data, holdout_data, score):
    X_train, y_train, X_valid, y_valid = hetero_data
    X_hold, y_hold = holdout_data
    alpha = 0.2

    model = ConformalRegressionWrapper(_base(), alpha=alpha, score=score)
    model.fit(X_train, y_train, X_valid, y_valid)

    lower, upper = model.predict_interval(X_hold)
    covered = (y_hold.values >= lower) & (y_hold.values <= upper)
    coverage = covered.mean()
    # Конечновыборочная гарантия ± шум малой выборки (n_calib=150) — допускаем запас
    assert coverage >= (1 - alpha) - 0.12
