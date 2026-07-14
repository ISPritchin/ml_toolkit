"""Тесты для InterpretableTreeRegressor/InterpretableTreeClassifier (Soft Decision Tree, Locally Linear Forest).

ml_toolkit/models/_tabular/_interpretable/_interpretable_trees.py. 'locally_linear_forest' поддерживается только
InterpretableTreeRegressor — для классификации Ridge не работает на бинарных таргетах, и этот
вариант лишь дублировал RandomForestClassifier без собственной логики, поэтому убран из
InterpretableTreeClassifier (см. ValueError-тест ниже).

PyTorch не входит в обязательные зависимости проекта (нужен только для 'soft_decision_tree') —
весь модуль пропускается через importorskip, если torch не установлен, хотя
'locally_linear_forest' сам по себе torch не требует.
"""

from __future__ import annotations

import pytest

pytest.importorskip('torch')

from ml_toolkit.models._tabular._interpretable._interpretable_trees import (
    InterpretableTreeClassifier,
    InterpretableTreeRegressor,
)
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

SDT_PARAMS = {'depth': 2, 'lr': 0.05, 'n_epochs': 30, 'patience': 10}
LLF_PARAMS = {'n_estimators': 20, 'max_depth': 4, 'n_neighbors': 20, 'ridge_alpha': 1.0, 'random_state': 42}


class TestSoftDecisionTreeRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == SDT_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(n_optuna_trials=1, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = InterpretableTreeRegressor(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestSoftDecisionTreeClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = InterpretableTreeClassifier(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)


class TestLocallyLinearForestRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(params=LLF_PARAMS, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == LLF_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(n_optuna_trials=2, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = InterpretableTreeRegressor(params=LLF_PARAMS, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestLocallyLinearForestClassifierRemoved:
    """'locally_linear_forest' больше не поддерживается InterpretableTreeClassifier (см. модуль)."""

    def test_raises_value_error(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(
            params={'n_estimators': 20, 'max_depth': 4}, model_settings={'name': 'locally_linear_forest'},
        )
        with pytest.raises(ValueError, match='soft_decision_tree'):
            model.fit(X_train, y_train, X_valid, y_valid)
