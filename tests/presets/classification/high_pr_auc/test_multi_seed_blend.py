"""Тесты для MultiSeedBlend (ml_toolkit/presets/classification/high_pr_auc/multi_seed_blend.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import MultiSeedBlend
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestMultiSeedBlend:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MultiSeedBlend(n_seeds=4, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert len(model.seed_scores_) == 4
        assert len(model.models_) == 4
