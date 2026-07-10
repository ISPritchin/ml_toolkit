"""Тесты для SelfTrainingBooster (ml_toolkit/presets/classification/high_pr_auc/self_training.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import SelfTrainingBooster
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'random_seed': 42, 'verbose': 0,
}


class TestSelfTrainingBooster:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SelfTrainingBooster(n_rounds=2, pseudo_weight=0.3, base_params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.round_scores_) >= 1
        assert model.threshold_used_ > 0.0

    def test_max_pseudo_ratio_caps_growth(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SelfTrainingBooster(
            n_rounds=3, max_pseudo_ratio=0.5, base_params=FAST_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        n_pos_orig = int(y_train.sum())
        assert sum(model.pseudo_added_) <= 0.5 * n_pos_orig + 1

    def test_explicit_threshold(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SelfTrainingBooster(n_rounds=2, threshold=0.6, base_params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.threshold_used_ == 0.6

    def test_optuna(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SelfTrainingBooster(n_rounds=2, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
