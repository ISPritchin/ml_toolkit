"""Тесты для ObliqueForestRegressor/ObliqueForestClassifier (ml_toolkit/models/_oblique_forest.py).

Пакет scikit-tree не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен (на момент написания тестов scikit-tree был
ABI-несовместим с установленным sklearn в тестовом окружении и не мог быть проверен вживую;
код написан по тем же принципам, что и quantile_forest/mondrian).
"""

from __future__ import annotations

import pytest

pytest.importorskip('sktree')

from ml_toolkit.models._oblique_forest import ObliqueForestClassifier, ObliqueForestRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'n_estimators': 30, 'max_depth': 4, 'random_state': 42, 'n_jobs': -1}


class TestObliqueForestRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = ObliqueForestRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = ObliqueForestRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_ordinal(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = ObliqueForestRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_predictions(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_

    def test_multiple_categorical_features_onehot(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = ObliqueForestRegressor(params=FAST_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_predictions(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col not in model.selected_features_
            assert any(f.startswith(f'{col}_') for f in model.selected_features_)


class TestObliqueForestClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ObliqueForestClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_explicit_class_weight_is_not_overridden(self, classification_data):
        """Regression test: dict-literal молча отбрасывал явный class_weight из self.params.

        {**self.params, 'class_weight': 'balanced'} — последний ключ dict-literal побеждает.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        params = {**FAST_PARAMS, 'class_weight': None}
        model = ObliqueForestClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['class_weight'] is None

    def test_fit_with_optuna_uses_balanced_class_weight(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ObliqueForestClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['class_weight'] == 'balanced'

    def test_multiple_categorical_features(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = ObliqueForestClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_
