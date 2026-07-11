"""Тесты для BoostedEnsemble (ml_toolkit/presets/classification/high_pr_auc/ensemble_losses.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import BoostedEnsemble
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_BASE_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'eval_metric': 'PRAUC', 'verbose': 0,
}
FAST_LOSS_CONFIGS = [
    {'loss_function': 'Logloss', 'scale_pos_weight': 1.0, 'random_seed': 42},
    {'loss_function': 'Logloss', 'scale_pos_weight': 4.0, 'random_seed': 123},
]


class TestBoostedEnsemble:
    @pytest.mark.parametrize('averaging', ['mean', 'rank', 'weighted', 'power'])
    def test_fit_predict(self, averaging, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BoostedEnsemble(
            loss_configs=FAST_LOSS_CONFIGS, averaging=averaging, base_params=FAST_BASE_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.models_) == 2

    def test_invalid_averaging_raises(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BoostedEnsemble(
            loss_configs=FAST_LOSS_CONFIGS, averaging='not_a_mode', base_params=FAST_BASE_PARAMS,
        )
        with pytest.raises(ValueError, match='averaging'):
            model.fit(X_train, y_train, X_valid, y_valid)

    def test_default_loss_configs_include_focal_loss(self, binary_data):
        """Дефолт (loss_configs=None) — 4 конфига (2 Logloss + 2 FocalLoss).

        FocalLoss не поддерживает predict_proba нативно, проверяем что sigmoid-fallback работает.
        """
        X_train, y_train, X_valid, y_valid = binary_data
        model = BoostedEnsemble(averaging='rank', base_params=FAST_BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert len(model.models_) == 4
        assert_valid_proba(model, X_valid)

    def test_optuna_tunes_shared_base_params(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BoostedEnsemble(loss_configs=FAST_LOSS_CONFIGS, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
