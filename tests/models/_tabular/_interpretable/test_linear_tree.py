"""Тесты для LinearTreeRegressor/LinearTreeClassifier (ml_toolkit/models/_tabular/_interpretable/_linear_tree.py).

Пакет linear-tree не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен (на момент написания тестов linear-tree==0.3.5
падал на sklearn>=1.6 из-за удалённого BaseEstimator._validate_data — upstream-несовместимость,
не связанная с кодом ml_toolkit; проверить вживую не удалось).
"""

from __future__ import annotations

import pytest

pytest.importorskip('lineartree')

from ml_toolkit.models._tabular._interpretable._linear_tree import LinearTreeClassifier, LinearTreeRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'max_depth': 3, 'min_samples_leaf': 10}


class TestLinearTreeRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearTreeRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearTreeRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = LinearTreeRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestLinearTreeClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearTreeClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = LinearTreeClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)
