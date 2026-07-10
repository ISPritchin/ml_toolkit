"""Тесты для MonotonicConstrainedClassifier
(ml_toolkit/presets/classification/high_pr_auc/monotonic_constrained.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.classification.high_pr_auc import MonotonicConstrainedClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba


class TestMonotonicConstrainedClassifier:
    @pytest.mark.parametrize('base', ['lightgbm', 'catboost'])
    def test_fit_predict(self, binary_data, base):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MonotonicConstrainedClassifier(
            monotone_constraints={'f0': 1, 'f1': -1}, base=base,
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    @pytest.mark.parametrize('base', ['lightgbm', 'catboost'])
    def test_prediction_is_monotonic_in_constrained_feature(self, binary_data, base):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MonotonicConstrainedClassifier(
            monotone_constraints={'f0': 1}, base=base,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        base_row = X_valid.iloc[[0]].copy()
        grid = base_row.loc[base_row.index.repeat(25)].reset_index(drop=True)
        grid['f0'] = np.linspace(X_train['f0'].min(), X_train['f0'].max(), 25)
        proba = model.predict_proba(grid)
        assert np.all(np.diff(proba) >= -1e-9), 'proba must be non-decreasing in f0 (+1 constraint)'

    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError, match='monotone_constraints'):
            MonotonicConstrainedClassifier(monotone_constraints={})

    def test_invalid_constraint_value_raises(self):
        with pytest.raises(ValueError, match='monotone_constraints'):
            MonotonicConstrainedClassifier(monotone_constraints={'f0': 2})

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match='base'):
            MonotonicConstrainedClassifier(monotone_constraints={'f0': 1}, base='xgboost')

    def test_unknown_feature_raises(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MonotonicConstrainedClassifier(monotone_constraints={'does_not_exist': 1})
        with pytest.raises(ValueError, match='selected_features'):
            model.fit(X_train, y_train, X_valid, y_valid)

    def test_categorical_constrained_feature_raises(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MonotonicConstrainedClassifier(monotone_constraints={'f0': 1})
        with pytest.raises(ValueError, match='категориальные'):
            model.fit(X_train, y_train, X_valid, y_valid, cat_features=['f0'])
