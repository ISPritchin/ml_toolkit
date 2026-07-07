"""Тесты для DriftRobustClassifier (ml_toolkit/presets/classification/high_pr_auc/drift_robust.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_toolkit.presets.classification.high_pr_auc import DriftRobustClassifier
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestDriftRobustClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = DriftRobustClassifier(target_auc=0.55, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.psi_report_ is not None
        assert len(model.adversarial_auc_history_) >= 1

    def test_removes_deliberately_drifted_feature(self):
        rng = np.random.default_rng(2)
        n = 400
        X_train = pd.DataFrame({
            'stable': rng.normal(size=n),
            'drifted': rng.normal(loc=0.0, size=n),
        })
        y_train = pd.Series((rng.random(n) < 0.15).astype(int))
        X_valid = pd.DataFrame({
            'stable': rng.normal(size=100),
            'drifted': rng.normal(loc=8.0, size=100),  # сильный сдвиг среднего
        })
        y_valid = pd.Series((rng.random(100) < 0.15).astype(int))

        model = DriftRobustClassifier(target_auc=0.55, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'drifted' in model.removed_features_
        assert 'stable' in model.selected_features_
