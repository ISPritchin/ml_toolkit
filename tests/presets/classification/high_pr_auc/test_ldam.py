"""Тесты для LDAMClassifier (ml_toolkit/presets/classification/high_pr_auc/ldam.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import LDAMClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestLDAMClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = LDAMClassifier(
            max_margin=0.5, reweight_epoch_frac=0.8, base_params=BASE_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        pred = model.predict(X_valid, threshold=0.5)
        assert set(np.unique(pred)) <= {0, 1}

    def test_optuna_tunes_loss_and_architecture(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = LDAMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'max_margin' in model.best_params_
        assert 'reweight_epoch_frac' in model.best_params_
