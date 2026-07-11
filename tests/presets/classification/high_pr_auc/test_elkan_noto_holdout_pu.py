"""Тесты для ElkanNotoHoldoutPU (ml_toolkit/presets/classification/high_pr_auc/elkan_noto_holdout_pu.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import ElkanNotoHoldoutPU
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestElkanNotoHoldoutPU:
    def test_fit_predict_and_ci(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = ElkanNotoHoldoutPU(c_holdout_frac=0.3, n_bootstrap=30, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.c_ci_[0] <= model.c_ci_[1]
        assert model.c_bootstrap_std_ >= 0.0

    def test_rejects_too_few_bootstrap(self):
        with pytest.raises(ValueError, match='n_bootstrap должен быть'):
            ElkanNotoHoldoutPU(n_bootstrap=1)
