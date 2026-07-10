"""Тесты для AsymmetricLossClassifier (ml_toolkit/presets/classification/high_pr_auc/asymmetric_loss.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import AsymmetricLossClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestAsymmetricLossClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AsymmetricLossClassifier(
            gamma_pos=0.0, gamma_neg=4.0, prob_margin=0.05, base_params=BASE_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        pred = model.predict(X_valid, threshold=0.5)
        assert set(np.unique(pred)) <= {0, 1}

    def test_optuna_tunes_loss_and_architecture(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AsymmetricLossClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        for key in ('gamma_pos', 'gamma_neg', 'prob_margin'):
            assert key in model.best_params_
