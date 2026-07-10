"""Тесты для HardNegativeMiner (ml_toolkit/presets/classification/high_pr_auc/hard_negative_mining.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import HardNegativeMiner
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestHardNegativeMiner:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HardNegativeMiner(n_rounds=2, hard_percentile=0.80, hard_weight=4.0, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.pr_auc_per_round_) == 2
        assert len(model.models_) == 2

    def test_best_round_model_is_used(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HardNegativeMiner(n_rounds=3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        best_idx = int(max(range(len(model.pr_auc_per_round_)), key=lambda i: model.pr_auc_per_round_[i]))
        assert model._model is model.models_[best_idx]

    def test_optuna_round0(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HardNegativeMiner(n_rounds=2, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_ is not None
