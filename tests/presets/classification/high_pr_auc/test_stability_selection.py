"""Тесты для StabilitySelectionClassifier
(ml_toolkit/presets/classification/high_pr_auc/stability_selection.py).
"""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import StabilitySelectionClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_BOOTSTRAP_PARAMS = {
    'iterations': 20, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'loss_function': 'Logloss', 'verbose': 0,
}
FAST_FINAL_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'verbose': 0,
}


class TestStabilitySelectionClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = StabilitySelectionClassifier(
            n_bootstrap=5, top_k=3, freq_threshold=0.2,
            bootstrap_params=FAST_BOOTSTRAP_PARAMS, final_params=FAST_FINAL_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.stable_features_) > 0
        assert model.selection_freq_ is not None
        assert set(model.stable_features_) <= set(model.selected_features_)

    def test_invalid_freq_threshold_raises(self):
        with pytest.raises(ValueError, match='freq_threshold'):
            StabilitySelectionClassifier(freq_threshold=1.5)

    def test_too_high_freq_threshold_raises_no_stable_features(self, binary_data):
        """top_k=1 + freq_threshold=1.0: только признак, лидирующий во ВСЕХ бутстрэпах,
        прошёл бы порог. С независимым от X случайным таргетом (binary_data) и
        достаточным числом бутстрэпов вероятность такого совпадения по всем 5
        признакам пренебрежимо мала — раскладка гарантированно даёт пустое ядро.
        """
        X_train, y_train, X_valid, y_valid = binary_data
        model = StabilitySelectionClassifier(
            n_bootstrap=10, top_k=1, freq_threshold=1.0,
            bootstrap_params=FAST_BOOTSTRAP_PARAMS, final_params=FAST_FINAL_PARAMS,
        )
        with pytest.raises(ValueError, match='freq_threshold'):
            model.fit(X_train, y_train, X_valid, y_valid)

    def test_optuna_tunes_final_model(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = StabilitySelectionClassifier(
            n_bootstrap=5, top_k=3, freq_threshold=0.2,
            bootstrap_params=FAST_BOOTSTRAP_PARAMS, n_optuna_trials=2,
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
