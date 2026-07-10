"""Тесты для FocalLossClassifier (ml_toolkit/presets/classification/high_pr_auc/focal_loss.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import FocalLossClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestFocalLossClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FocalLossClassifier(gamma=2.0, alpha=0.25, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        pred = model.predict(X_valid, threshold=0.5)
        assert set(np.unique(pred)) <= {0, 1}
        assert model.train_pred_.shape == (len(X_train),)

    def test_optuna_tunes_loss_and_architecture(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FocalLossClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'gamma' in model.best_params_
        assert 'alpha' in model.best_params_
