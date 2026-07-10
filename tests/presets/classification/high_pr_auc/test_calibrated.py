"""Тесты для CalibratedWrapper (ml_toolkit/presets/classification/high_pr_auc/calibrated.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import CalibratedWrapper, EasyEnsembleClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestCalibratedWrapper:
    @pytest.mark.parametrize('method', ['isotonic', 'platt'])
    def test_fit_predict(self, method, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = CalibratedWrapper(base, method=method)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert model.base_ is base
        assert model.calibrator_ is not None
        assert 0.0 <= model.raw_pr_auc_ <= 1.0
        assert 0.0 <= model.calibrated_pr_auc_ <= 1.0

    def test_invalid_method_raises(self):
        base = EasyEnsembleClassifier(n_estimators=3, base_params=BASE_PARAMS)
        with pytest.raises(ValueError, match='method'):
            CalibratedWrapper(base, method='not_a_method')

    def test_proxies_metadata_from_base(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = CalibratedWrapper(base)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.selected_features_ == base.selected_features_
        assert model.cat_features_ == base.cat_features_
