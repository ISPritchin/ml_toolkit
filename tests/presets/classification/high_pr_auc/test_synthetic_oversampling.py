"""Тесты для SyntheticOversamplingClassifier
(ml_toolkit/presets/classification/high_pr_auc/synthetic_oversampling.py).
"""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import SyntheticOversamplingClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

pytest.importorskip('imblearn')

FAST_CBT_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'random_seed': 42, 'verbose': 0,
}


class TestSyntheticOversamplingClassifier:
    @pytest.mark.parametrize('method', ['smote', 'adasyn', 'borderline'])
    def test_fit_predict_catboost(self, method, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SyntheticOversamplingClassifier(
            method=method, sampling_strategy=0.3, base='catboost', base_params=FAST_CBT_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert model.n_synthetic_ > 0
        assert model.augmented_ratio_ > 0

    def test_fit_predict_lightgbm_base(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SyntheticOversamplingClassifier(
            method='smote', sampling_strategy=0.3, base='lightgbm',
            base_params={'n_estimators': 40, 'max_depth': 3, 'num_leaves': 7, 'verbose': -1},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match='base'):
            SyntheticOversamplingClassifier(base='not_a_base')

    def test_smoteenn_with_cat_features_raises(self, binary_data_with_cat):
        X_train, y_train, X_valid, y_valid = binary_data_with_cat
        model = SyntheticOversamplingClassifier(method='smoteenn', base_params=FAST_CBT_PARAMS)
        with pytest.raises(ValueError, match='smoteenn'):
            model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
