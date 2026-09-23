"""Pickle-контракт BasePreset для всех 3 классов multiclass_imbalance.

save()/load() — общая реализация BaseModel, наследуемая без переопределения
(см. tests/models/test_pickle.py для полного описания контракта).
"""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.multiclass_imbalance import (
    BalancedSoftmaxClassifier,
    EqualizationLossClassifier,
    LogitNormLossClassifier,
)
from tests.presets.classification.multiclass_imbalance.conftest import BASE_PARAMS


def _assert_proba_pickle_roundtrip(model, X_valid, tmp_path) -> None:
    pred_before = model.predict_proba(X_valid)
    path = tmp_path / 'model.pkl'
    model.save(path)
    loaded = type(model).load(path)
    assert isinstance(loaded, type(model))
    pred_after = loaded.predict_proba(X_valid)
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-9, atol=1e-12)


class TestMulticlassImbalancePresetsPickle:
    def test_balanced_softmax(self, multiclass_data, tmp_path):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = BalancedSoftmaxClassifier(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_equalization_loss(self, multiclass_data, tmp_path):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = EqualizationLossClassifier(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_logitnorm_loss(self, multiclass_data, tmp_path):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LogitNormLossClassifier(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)
