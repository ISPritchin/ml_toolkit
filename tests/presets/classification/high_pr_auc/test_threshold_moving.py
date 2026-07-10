"""Тесты для ThresholdMovingCV (ml_toolkit/presets/classification/high_pr_auc/threshold_moving.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.classification.high_pr_auc import EasyEnsembleClassifier, ThresholdMovingCV
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestThresholdMovingCV:
    @pytest.mark.parametrize('optimize', ['f1', 'f2', 'f0.5'])
    def test_fit_predict(self, optimize, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = ThresholdMovingCV(base, optimize=optimize)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert 0.0 <= model.threshold_ <= 1.0
        assert model.scan_df_ is not None

        labels = model.predict(X_valid)
        assert set(np.unique(labels)) <= {0, 1}

    def test_precision_at_recall_requires_min_recall(self):
        with pytest.raises(ValueError, match='min_recall'):
            ThresholdMovingCV(None, optimize='precision_at_recall')

    def test_invalid_optimize_raises(self):
        with pytest.raises(ValueError, match='optimize'):
            ThresholdMovingCV(None, optimize='not_a_metric')

    def test_precision_at_recall_mode(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = ThresholdMovingCV(base, optimize='precision_at_recall', min_recall=0.5)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_predict_with_explicit_threshold_override(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = ThresholdMovingCV(base, optimize='f2')
        model.fit(X_train, y_train, X_valid, y_valid)

        labels_all_positive = model.predict(X_valid, threshold=0.0)
        assert set(np.unique(labels_all_positive)) <= {1}
