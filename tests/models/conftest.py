"""Общие fixtures/helpers для тестов ml_toolkit/models/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series((X_train['f0'] + X_train['f1'] + rng.normal(scale=0.5, size=n_train) > 0).astype(int))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series((X_valid['f0'] + X_valid['f1'] + rng.normal(scale=0.5, size=n_valid) > 0).astype(int))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(X_train['f0'] * 2.0 - X_train['f1'] + rng.normal(scale=0.5, size=n_train))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(X_valid['f0'] * 2.0 - X_valid['f1'] + rng.normal(scale=0.5, size=n_valid))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def positive_regression_data():
    """Строго положительный таргет — нужен для Tweedie (power>1 требует y > 0)."""
    rng = np.random.default_rng(2)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(np.exp(0.5 * X_train['f0'] + rng.normal(scale=0.3, size=n_train)))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(np.exp(0.5 * X_valid['f0'] + rng.normal(scale=0.3, size=n_valid)))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def classification_data_with_cat():
    """Классификация с одним категориальным признаком (для cat_features/cat_encoder тестов)."""
    rng = np.random.default_rng(3)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(4)]

    def _make(n, seed):
        r = np.random.default_rng(seed)
        X = pd.DataFrame(r.normal(size=(n, 4)), columns=cols)
        X['cat_col'] = r.choice(['a', 'b', 'c'], size=n)
        cat_effect = X['cat_col'].map({'a': 1.0, 'b': -1.0, 'c': 0.0}).to_numpy()
        y = pd.Series((X['f0'] + cat_effect + r.normal(scale=0.5, size=n) > 0).astype(int))
        return X, y

    X_train, y_train = _make(n_train, 10)
    X_valid, y_valid = _make(n_valid, 11)
    return X_train, y_train, X_valid, y_valid


MULTI_CAT_FEATURES = ['cat_binary', 'cat_low_card', 'cat_high_card']


def _make_multi_cat_frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Строит DataFrame с тремя категориальными признаками разной кардинальности.

    2 / 4 / 10 уровней, плюс числовые признаки. Возвращает (X, binary_effect, low_card_effect)
    — эффекты используются вызывающим для построения таргета классификации/регрессии.
    """
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 3)), columns=[f'f{i}' for i in range(3)])
    X['cat_binary'] = r.choice(['yes', 'no'], size=n)
    X['cat_low_card'] = r.choice(['a', 'b', 'c', 'd'], size=n)
    X['cat_high_card'] = r.choice([f'g{i}' for i in range(10)], size=n)
    binary_effect = X['cat_binary'].map({'yes': 1.0, 'no': -1.0}).to_numpy()
    low_card_effect = X['cat_low_card'].map({'a': 1.0, 'b': 0.5, 'c': -0.5, 'd': -1.0}).to_numpy()
    return X, binary_effect, low_card_effect


@pytest.fixture
def classification_data_multi_cat():
    """Классификация с тремя категориальными признаками разной кардинальности (2/4/10 уровней)."""
    n_train, n_valid = 300, 80

    def _make(n, seed):
        X, binary_effect, low_card_effect = _make_multi_cat_frame(n, seed)
        r = np.random.default_rng(seed + 1)
        y = pd.Series(
            (X['f0'] + binary_effect + low_card_effect + r.normal(scale=0.5, size=n) > 0).astype(int)
        )
        return X, y

    X_train, y_train = _make(n_train, 20)
    X_valid, y_valid = _make(n_valid, 21)
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def regression_data_multi_cat():
    """Регрессия с тремя категориальными признаками разной кардинальности (2/4/10 уровней)."""
    n_train, n_valid = 300, 80

    def _make(n, seed):
        X, binary_effect, low_card_effect = _make_multi_cat_frame(n, seed)
        r = np.random.default_rng(seed + 1)
        y = pd.Series(X['f0'] * 2.0 + binary_effect + low_card_effect + r.normal(scale=0.5, size=n))
        return X, y

    X_train, y_train = _make(n_train, 30)
    X_valid, y_valid = _make(n_valid, 31)
    return X_train, y_train, X_valid, y_valid


def assert_valid_proba(model, X_valid) -> np.ndarray:
    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))
    return proba


def assert_valid_predictions(model, X_valid) -> np.ndarray:
    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    return pred
