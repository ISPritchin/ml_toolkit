"""Тесты для RandomForest*/ExtraTrees* (ml_toolkit/models/_forest.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models import train_classification_model, train_regression_model
from ml_toolkit.models._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tests.models.conftest import assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'n_estimators': 30, 'max_depth': 4, 'random_state': 42, 'n_jobs': -1}


@pytest.mark.parametrize('RegClass', [RandomForestRegressor, ExtraTreesRegressor])
class TestForestRegressors:
    def test_fit_predict_explicit_params(self, RegClass, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RegClass(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_requires_valid_for_optuna(self, RegClass, regression_data):
        X_train, y_train, _, _ = regression_data
        model = RegClass(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, RegClass, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RegClass(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 'n_estimators' in model.best_params_

    def test_nan_handled_via_imputer(self, RegClass, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_train.loc[0, 'f0'] = np.nan
        model = RegClass(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


@pytest.mark.parametrize('ClsClass', [RandomForestClassifier, ExtraTreesClassifier])
class TestForestClassifiers:
    def test_fit_predict_proba_explicit_params(self, ClsClass, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, ClsClass, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_fit_with_optuna(self, ClsClass, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_.get('class_weight') == 'balanced'


@pytest.mark.parametrize('ClsClass', [RandomForestClassifier, ExtraTreesClassifier])
class TestForestClassWeight:
    def test_explicit_class_weight_is_not_overridden(self, ClsClass, classification_data):
        """Регрессия бага: {**self.params, 'class_weight': 'balanced'} молча отбрасывал
        явный class_weight пользователя (последний ключ dict-literal побеждает).
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params={**FAST_PARAMS, 'class_weight': None})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] is None
        assert model._model.named_steps['estimator'].class_weight is None

    def test_explicit_class_weight_balanced_does_not_raise(self, ClsClass, classification_data):
        """До фикса это тоже проходило бы (случайно, dict-literal без дублирующегося
        keyword-аргумента), но явный тест на этот сценарий не был написан.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params={**FAST_PARAMS, 'class_weight': 'balanced'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_default_is_balanced_when_not_specified(self, ClsClass, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] == 'balanced'


class TestForestCatEncoder:
    def test_ordinal_default_encoder(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = RandomForestClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        # Ordinal encoder не расширяет список признаков
        assert 'cat_col' in model.selected_features_

    def test_onehot_encoder_expands_features(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = RandomForestClassifier(params=FAST_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)
        assert 'cat_col' not in model.selected_features_
        assert any(f.startswith('cat_col_') for f in model.selected_features_)


class TestForestFunctionalAPI:
    def test_train_regression_model_random_forest(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        raw_model, train_pred, valid_pred, infer_pred, best_params = train_regression_model(
            name='random_forest', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            model_settings={'name': 'random_forest'}, n_optuna_trials=2,
        )
        assert raw_model is not None
        assert valid_pred.shape == (len(X_valid),)

    def test_train_classification_model_extra_trees(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        raw_model, train_proba, val_proba, infer_proba, best_params = train_classification_model(
            name='extra_trees', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            n_optuna_trials=2, model_settings={'name': 'extra_trees'},
        )
        assert raw_model is not None
        assert np.all((infer_proba >= 0) & (infer_proba <= 1))
