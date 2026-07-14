"""Тесты для QuantileForestRegressor/QuantileForestClassifier (ml_toolkit/models/_tabular/_forests/_quantile_forest.py).

Пакет quantile-forest не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.
"""

from __future__ import annotations

import pytest

pytest.importorskip('quantile_forest')

from ml_toolkit.models._tabular._forests._quantile_forest import QuantileForestClassifier, QuantileForestRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_REG_PARAMS = {'n_estimators': 30, 'max_depth': 4, 'random_state': 42, 'n_jobs': -1}
FAST_CLS_PARAMS = {'C': 1.0}


class TestQuantileForestRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileForestRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_REG_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileForestRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_pipeline_predict_does_not_crash(self, regression_data):
        """Regression test: _QuantileMedianWrapper без BaseEstimator падал в Pipeline.predict().

        На sklearn>=1.6 (__sklearn_tags__ отсутствовал).
        """
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileForestRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        preds = model._model.predict(X_valid[model.selected_features_])
        assert len(preds) == len(X_valid)

    def test_multiple_categorical_features_ordinal(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = QuantileForestRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_predictions(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_

    def test_multiple_categorical_features_onehot(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = QuantileForestRegressor(params=FAST_REG_PARAMS, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_predictions(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col not in model.selected_features_
            assert any(f.startswith(f'{col}_') for f in model.selected_features_)


class TestQuantileForestClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = QuantileForestClassifier(params=FAST_CLS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_explicit_params_do_not_crash_on_max_iter_class_weight(self, classification_data):
        """Regression test: падал с TypeError, если self.params уже содержал max_iter/class_weight.

        LogisticRegression(**self.params, max_iter=500, class_weight='balanced').
        """
        X_train, y_train, X_valid, y_valid = classification_data
        params = {'C': 2.0, 'max_iter': 200, 'class_weight': None}
        model = QuantileForestClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['max_iter'] == 200
        assert model.best_params_['class_weight'] is None

    def test_fit_with_optuna_uses_balanced_class_weight(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = QuantileForestClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = QuantileForestClassifier(params=FAST_CLS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert_valid_proba(model, X_valid)
        for col in MULTI_CAT_FEATURES:
            assert col in model.selected_features_
