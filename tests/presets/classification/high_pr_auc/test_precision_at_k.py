"""Тесты для PrecisionAtKClassifier (ml_toolkit/presets/classification/high_pr_auc/precision_at_k.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.classification.high_pr_auc import PrecisionAtKClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba


class TestPrecisionAtKClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PrecisionAtKClassifier(k_fraction=0.10, n_optuna_trials=3)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert model.train_pred_.shape == (len(X_train),)
        assert 0.0 <= model.best_precision_at_k_ <= 1.0
        assert 'majority_fraction' in model.best_params_

    def test_invalid_k_fraction_raises(self):
        with pytest.raises(ValueError, match='k_fraction'):
            PrecisionAtKClassifier(k_fraction=1.5)

    def test_predict_uses_threshold(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PrecisionAtKClassifier(k_fraction=0.10, n_optuna_trials=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_valid, threshold=0.5)
        assert set(np.unique(pred)) <= {0, 1}
