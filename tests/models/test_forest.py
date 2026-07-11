"""Тесты для RandomForest*/ExtraTrees* (ml_toolkit/models/_forest.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_fit_with_optuna(self, ClsClass, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_.get('class_weight') == 'balanced'


@pytest.mark.parametrize('ClsClass', [RandomForestClassifier, ExtraTreesClassifier])
class TestForestClassWeight:
    def test_explicit_class_weight_is_not_overridden(self, ClsClass, classification_data):
        """Регрессия бага: dict-literal молча отбрасывал явный class_weight пользователя.

        {**self.params, 'class_weight': 'balanced'} — последний ключ dict-literal побеждает.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params={**FAST_PARAMS, 'class_weight': None})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] is None
        assert model._model.named_steps['estimator'].class_weight is None

    def test_explicit_class_weight_balanced_does_not_raise(self, ClsClass, classification_data):
        """До фикса это тоже проходило бы случайно, но явный тест на этот сценарий не был написан.

        dict-literal без дублирующегося keyword-аргумента.
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

    def test_multiple_categorical_features_ordinal(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = RandomForestClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_

    def test_multiple_categorical_features_onehot(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = RandomForestClassifier(params=FAST_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col not in model.selected_features_
            assert any(f.startswith(f'{col}_') for f in model.selected_features_)

