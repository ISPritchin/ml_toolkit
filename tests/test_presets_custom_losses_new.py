"""Смоук-тесты fit/predict для новых пресетов с кастомными лоссами (201-207).

CLAUDE.md отмечает, что ни один пресет в high_pr_auc/ не имеет юнит-тестов
(M3 аудита) — эти тесты хотя бы покрывают базовый сценарий fit -> predict
для только что добавленных классов, включая новый мультиклассовый подпакет
multiclass_imbalance/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification.high_pr_auc import (
    AsymmetricPolyLossClassifier,
    DiceLossClassifier,
    GHMLossClassifier,
    InfluenceBalancedLossClassifier,
)
from ml_toolkit.presets.classification.multiclass_imbalance import (
    BalancedSoftmaxClassifier,
    EqualizationLossClassifier,
    LogitNormLossClassifier,
)

_BASE_PARAMS = {'iterations': 50, 'verbose': 0, 'random_seed': 42}


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series((rng.random(n_train) < 0.15).astype(int))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series((rng.random(n_valid) < 0.15).astype(int))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    probs = [0.6, 0.25, 0.1, 0.05]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(rng.choice([0, 1, 2, 3], size=n_train, p=probs))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(rng.choice([0, 1, 2, 3], size=n_valid, p=probs))
    return X_train, y_train, X_valid, y_valid


@pytest.mark.parametrize(
    'cls', [GHMLossClassifier, InfluenceBalancedLossClassifier, DiceLossClassifier, AsymmetricPolyLossClassifier]
)
def test_binary_preset_fit_predict(binary_data, cls):
    X_train, y_train, X_valid, y_valid = binary_data
    model = cls(base_params=_BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))

    pred = model.predict(X_valid, threshold=0.5)
    assert set(np.unique(pred)) <= {0, 1}
    assert model.train_pred_.shape == (len(X_train),)


@pytest.mark.parametrize('cls', [EqualizationLossClassifier, BalancedSoftmaxClassifier, LogitNormLossClassifier])
def test_multiclass_preset_fit_predict(multiclass_data, cls):
    X_train, y_train, X_valid, y_valid = multiclass_data
    model = cls(base_params=_BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred = model.predict(X_valid)
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    assert model.n_classes_ == 4


def test_multiclass_base_rejects_binary_target(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    model = EqualizationLossClassifier(base_params=_BASE_PARAMS)
    with pytest.raises(ValueError):
        model.fit(X_train, y_train, X_valid, y_valid)
