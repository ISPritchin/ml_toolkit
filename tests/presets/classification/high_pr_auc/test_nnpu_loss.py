"""Тесты для NNPUClassifier (ml_toolkit/presets/classification/high_pr_auc/nnpu_loss.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import NNPUClassifier
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestNNPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = NNPUClassifier(class_prior=0.15, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_rejects_invalid_class_prior(self):
        with pytest.raises(ValueError, match='class_prior должен быть'):
            NNPUClassifier(class_prior=1.5)
