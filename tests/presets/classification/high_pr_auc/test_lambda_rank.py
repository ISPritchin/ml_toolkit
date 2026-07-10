"""Тесты для LambdaRankClassifier (ml_toolkit/presets/classification/high_pr_auc/lambda_rank.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import LambdaRankClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_PARAMS = {
    'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.2, 'n_estimators': 40,
    'min_child_samples': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'n_jobs': -1,
}


class TestLambdaRankClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = LambdaRankClassifier(base_params=FAST_PARAMS, early_stopping_rounds=20)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert 0.0 <= model.map_train_ <= 1.0
        assert 0.0 <= model.map_valid_ <= 1.0

    def test_predict_proba_is_rank_transform_of_raw_score(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = LambdaRankClassifier(base_params=FAST_PARAMS, early_stopping_rounds=20)
        model.fit(X_train, y_train, X_valid, y_valid)

        proba = model.predict_proba(X_valid)
        raw = model._model.predict(X_valid[model.selected_features_])
        # Ранговая нормализация монотонна — порядок должен совпадать с исходным raw score
        assert (np.argsort(proba) == np.argsort(raw)).all()

    def test_truncation_level_param_is_applied(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = LambdaRankClassifier(base_params=FAST_PARAMS, truncation_level=10, early_stopping_rounds=20)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['lambdarank_truncation_level'] == 10
