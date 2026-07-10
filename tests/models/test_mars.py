"""Тесты для MARSRegressor/MARSClassifier (ml_toolkit/models/_mars.py).

Пакет py-earth не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен (на Python 3.10+ py-earth не собирается ни из
PyPI, ни из git HEAD — использует удалённый из CPython заголовок longintrepr.h; на момент
написания тестов проверить вживую не удалось).
"""

from __future__ import annotations

import pytest

pytest.importorskip('pyearth')

from ml_toolkit.models._mars import MARSClassifier, MARSRegressor  # noqa: E402
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba  # noqa: E402

FAST_REG_PARAMS = {'max_degree': 1, 'max_terms': 20}


class TestMARSRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = MARSRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_REG_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = MARSRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = MARSRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestMARSClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = MARSClassifier(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_non_c_logistic_kwargs_are_not_dropped(self, classification_data):
        """Regression test: clf_p раньше фильтровался как k == 'C' — любой другой валидный
        kwarg LogisticRegression (fit_intercept, class_weight) молча отбрасывался.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        params = {**FAST_REG_PARAMS, 'C': 0.5, 'fit_intercept': False, 'class_weight': None}
        model = MARSClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model._clf.fit_intercept is False
        assert model._clf.class_weight is None
        assert model._clf.C == 0.5

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = MARSClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = MARSClassifier(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)
