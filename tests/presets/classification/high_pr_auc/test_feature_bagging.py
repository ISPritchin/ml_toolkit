"""Тесты для FeatureBaggingEnsemble (ml_toolkit/presets/classification/high_pr_auc/feature_bagging.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import FeatureBaggingEnsemble
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestFeatureBaggingEnsemble:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FeatureBaggingEnsemble(n_estimators=4, feature_frac=0.6, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.estimators_) == 4
        assert len(model.feature_subsets_) == 4
        assert all(len(s) <= len(model.selected_features_) for s in model.feature_subsets_)
        assert 0.0 <= model.ensemble_score_ <= 1.0

    def test_invalid_feature_frac_raises(self):
        with pytest.raises(ValueError, match='feature_frac'):
            FeatureBaggingEnsemble(feature_frac=1.5)

    def test_feature_subsets_differ_across_estimators(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FeatureBaggingEnsemble(n_estimators=5, feature_frac=0.6, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        subsets_as_sets = [frozenset(s) for s in model.feature_subsets_]
        assert len(set(subsets_as_sets)) > 1

    def test_optuna_tunes_on_representative_subspace(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FeatureBaggingEnsemble(n_estimators=3, feature_frac=0.6, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
