"""Тесты для BaggingPUClassifier (ml_toolkit/presets/classification/high_pr_auc/bagging_pu.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import BaggingPUClassifier
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestBaggingPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BaggingPUClassifier(n_estimators=10, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 0.0 < model.oob_coverage_ <= 1.0
        assert len(model.estimators_) == 10

    def test_rejects_too_few_estimators(self):
        with pytest.raises(ValueError):
            BaggingPUClassifier(n_estimators=1)
