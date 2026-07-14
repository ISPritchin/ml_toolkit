"""Тесты для RuleFitRegressor/RuleFitClassifier (ml_toolkit/models/_tabular/_interpretable/_rulefit.py).

Пакет imodels не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.
"""

from __future__ import annotations

import pytest

pytest.importorskip('imodels')

from ml_toolkit.models._tabular._interpretable._rulefit import RuleFitClassifier, RuleFitRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'max_rules': 50, 'tree_size': 3, 'random_state': 42}


class TestRuleFitRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RuleFitRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RuleFitRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = RuleFitRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestRuleFitClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = RuleFitClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = RuleFitClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = RuleFitClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)
