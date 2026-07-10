"""Тесты для PULearningClassifier (ml_toolkit/presets/classification/high_pr_auc/pu_learning.py)."""

from __future__ import annotations

from catboost import Pool
import pytest

from ml_toolkit.presets.classification.high_pr_auc import PULearningClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'random_seed': 42, 'verbose': 0,
}


class TestPULearningClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PULearningClassifier(base_params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert 0.0 <= model.c_ <= 1.0
        assert 0.0 <= model.raw_pr_auc_ <= 1.0
        assert 0.0 <= model.corrected_pr_auc_ <= 1.0

    def test_correction_does_not_change_ranking(self, binary_data):
        """Коррекция raw/c монотонна — не должна менять ранжирование объектов."""
        X_train, y_train, X_valid, y_valid = binary_data
        model = PULearningClassifier(base_params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        proba = model.predict_proba(X_valid)
        raw = model._model.predict_proba(
            Pool(X_valid[model.selected_features_], cat_features=model.cat_features_)
        )[:, 1]
        assert (proba.argsort() == (raw / model.c_).clip(0, 1).argsort()).all()

    def test_invalid_c_estimation_frac_raises(self):
        with pytest.raises(ValueError, match='c_estimation_frac'):
            PULearningClassifier(c_estimation_frac=1.5)

    def test_optuna(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = PULearningClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
