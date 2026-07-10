"""Тесты для HeterogeneousStacking (ml_toolkit/presets/classification/high_pr_auc/heterogeneous_stacking.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import HeterogeneousStacking
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba


class TestHeterogeneousStacking:
    @pytest.mark.slow
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'logistic'], n_folds=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert set(model.zoo_used_) == {'catboost', 'lightgbm', 'logistic'}

    @pytest.mark.slow
    def test_missing_xgboost_gracefully_skipped(self, binary_data):
        # xgboost не установлен в тестовом окружении — дефолтный зоопарк должен
        # молча отфильтровать его, а не упасть.
        X_train, y_train, X_valid, y_valid = binary_data
        model = HeterogeneousStacking(n_folds=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'xgboost' not in model.zoo_used_

    @pytest.mark.slow
    def test_meta_variants(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        for meta in ('logistic', 'weighted', 'catboost'):
            model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm'], meta=meta, n_folds=3)
            model.fit(X_train, y_train, X_valid, y_valid)
            assert_valid_proba(model, X_valid)

    def test_rejects_too_small_zoo_after_filtering(self):
        with pytest.raises(ValueError):
            HeterogeneousStacking(base_zoo=['xgboost'])
