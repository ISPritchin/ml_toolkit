"""Тесты для HistGBMRegressor/HistGBMClassifier (ml_toolkit/models/_hist_gbm.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models._hist_gbm import HistGBMClassifier, HistGBMRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'max_iter': 40, 'max_depth': 3, 'random_state': 42}


class TestHistGBMRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = HistGBMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_requires_valid_for_optuna(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = HistGBMRegressor(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = HistGBMRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_.get('loss') == 'absolute_error'

    def test_handles_nan_natively(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_train.loc[0, 'f0'] = np.nan
        model = HistGBMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestHistGBMClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = HistGBMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_no_calibrator_without_valid(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        assert_valid_proba(model, X_train)


class TestHistGBMCatFeatures:
    def test_categorical_feature_via_indices_explicit_params(self, classification_data_with_cat):
        """Регрессия бага: explicit-params ветка не получала вычисленный extra (categorical_features) вовсе.

        HistGradientBoostingClassifier(**self.params) — cat_features молча теряли
        нативную categorical-обработку HistGBM.
        """
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        assert list(model._model.categorical_features) == [4]  # cat_col — 5-й (индекс 4) признак
        assert model.best_params_['categorical_features'] == [4]

    def test_categorical_feature_via_indices_optuna(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = HistGBMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        assert list(model._model.categorical_features) == [4]

    def test_categorical_feature_regressor_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = HistGBMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_predictions(model, X_valid)
        assert list(model._model.categorical_features) == [5]  # cat_col — 6-й (индекс 5) признак

    def test_multiple_categorical_features_classifier(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        cat_idx = [model.selected_features_.index(c) for c in MULTI_CAT_FEATURES]
        assert list(model._model.categorical_features) == cat_idx

    def test_multiple_categorical_features_regressor(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = HistGBMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_predictions(model, X_valid)
        cat_idx = [model.selected_features_.index(c) for c in MULTI_CAT_FEATURES]
        assert list(model._model.categorical_features) == cat_idx

    def test_no_cat_features_means_no_categorical_features_param(self, classification_data):
        """При отсутствии cat_features extra = {} — параметр вообще не должен просачиваться."""
        X_train, y_train, X_valid, y_valid = classification_data
        model = HistGBMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'categorical_features' not in model.best_params_

