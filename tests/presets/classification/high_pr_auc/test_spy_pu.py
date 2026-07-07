"""Тесты для SpyPUClassifier (ml_toolkit/presets/classification/high_pr_auc/spy_pu.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import SpyPUClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestSpyPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SpyPUClassifier(spy_frac=0.15, spy_threshold_pct=10.0, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.n_spies_ > 0
        assert model.n_reliable_negative_ >= 0

    def test_rejects_invalid_spy_frac(self):
        with pytest.raises(ValueError):
            SpyPUClassifier(spy_frac=0.9)
