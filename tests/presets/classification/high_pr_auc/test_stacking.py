"""Тесты для SubsampleStacking (ml_toolkit/presets/classification/high_pr_auc/stacking.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import SubsampleStacking
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_CONFIGS = [
    {'iterations': 30, 'max_depth': 3, 'learning_rate': 0.2, 'scale_pos_weight': 1.0, 'random_seed': 42},
    {'iterations': 30, 'max_depth': 3, 'learning_rate': 0.2, 'scale_pos_weight': 3.0, 'random_seed': 123},
    {'iterations': 30, 'max_depth': 3, 'learning_rate': 0.2, 'scale_pos_weight': 2.0, 'random_seed': 789},
]


class TestSubsampleStacking:
    @pytest.mark.parametrize('meta', ['logistic', 'weighted', 'catboost'])
    def test_fit_predict(self, meta, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SubsampleStacking(
            n_base_models=3, n_folds=3, base_configs=FAST_CONFIGS, meta=meta,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.base_models_) == 3
        assert len(model.oob_pr_aucs_) == 3
        assert model.meta_model_ is not None

    def test_invalid_subsample_rate_raises(self):
        with pytest.raises(ValueError, match='subsample_rate'):
            SubsampleStacking(subsample_rate=1.5)

    def test_invalid_n_folds_raises(self):
        with pytest.raises(ValueError, match='n_folds'):
            SubsampleStacking(n_folds=1)

    def test_invalid_meta_raises(self):
        with pytest.raises(ValueError, match='meta'):
            SubsampleStacking(meta='not_a_meta')

    def test_optuna_tunes_shared_params(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SubsampleStacking(n_base_models=2, n_folds=3, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
