"""Тесты для DecisionTreeRegressor/DecisionTreeClassifier (ml_toolkit/models/_decision_tree.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models._decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'max_depth': 3, 'random_state': 42}


class TestDecisionTreeRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = DecisionTreeRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_requires_valid_for_optuna(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = DecisionTreeRegressor(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = DecisionTreeRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 2 <= model.best_params_['max_depth'] <= 8

    def test_nan_handled_via_imputer(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_train.loc[0, 'f0'] = np.nan
        model = DecisionTreeRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestDecisionTreeClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = DecisionTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = DecisionTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_fit_with_optuna_uses_balanced_class_weight(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = DecisionTreeClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['class_weight'] == 'balanced'

    def test_no_calibrator_without_valid(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = DecisionTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        assert_valid_proba(model, X_train)


class TestDecisionTreeCatEncoder:
    def test_ordinal_encoding_used_by_default(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = DecisionTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        assert 'cat_col' in model.selected_features_

    def test_onehot_encoder_expands_features(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = DecisionTreeClassifier(params=FAST_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        assert 'cat_col' not in model.selected_features_
        assert any(f.startswith('cat_col_') for f in model.selected_features_)

    def test_categorical_feature_regressor(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = DecisionTreeRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert 'cat_col' in model.selected_features_
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_ordinal(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = DecisionTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_

    def test_multiple_categorical_features_onehot(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = DecisionTreeClassifier(params=FAST_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col not in model.selected_features_
            assert any(f.startswith(f'{col}_') for f in model.selected_features_)

