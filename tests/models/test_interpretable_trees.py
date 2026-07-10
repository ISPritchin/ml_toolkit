"""Тесты для InterpretableTreeRegressor/InterpretableTreeClassifier
(ml_toolkit/models/_interpretable_trees.py): Soft Decision Tree и Locally Linear Forest.

PyTorch не входит в обязательные зависимости проекта (нужен только для 'soft_decision_tree') —
весь модуль пропускается через importorskip, если torch не установлен, хотя
'locally_linear_forest' сам по себе torch не требует.
"""

from __future__ import annotations

import pytest

pytest.importorskip('torch')

from ml_toolkit.models._interpretable_trees import (  # noqa: E402
    InterpretableTreeClassifier,
    InterpretableTreeRegressor,
)
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba  # noqa: E402

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


class TestLocallyLinearForestClassifier:
    """Классификатор locally_linear_forest использует plain RandomForestClassifier."""

    RF_PARAMS = {'n_estimators': 20, 'max_depth': 4}

    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(params=self.RF_PARAMS, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_explicit_params_default_to_balanced_class_weight(self, classification_data):
        """Regression test: explicit-ветка раньше не форсировала class_weight='balanced' в
        отличие от Optuna-ветки — теперь обе ветки согласованы.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(params=self.RF_PARAMS, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] == 'balanced'

    def test_explicit_class_weight_is_not_overridden(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        params = {**self.RF_PARAMS, 'class_weight': None}
        model = InterpretableTreeClassifier(params=params, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] is None

    def test_fit_with_optuna_uses_balanced_class_weight(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(n_optuna_trials=2, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['class_weight'] == 'balanced'

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = InterpretableTreeClassifier(params=self.RF_PARAMS, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)
