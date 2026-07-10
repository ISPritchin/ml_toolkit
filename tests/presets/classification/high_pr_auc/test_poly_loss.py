"""Тесты для PolyLossClassifier (ml_toolkit/presets/classification/high_pr_auc/poly_loss.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import PolyLossClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestPolyLossClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PolyLossClassifier(eps1=2.0, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        pred = model.predict(X_valid, threshold=0.5)
        assert set(np.unique(pred)) <= {0, 1}

    def test_optuna_tunes_loss_and_architecture(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PolyLossClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'eps1' in model.best_params_
