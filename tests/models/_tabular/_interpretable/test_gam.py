"""Тесты для PyGAMRegressor/PyGAMClassifier (ml_toolkit/models/_tabular/_interpretable/_gam.py).

Пакет pygam не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.
"""

from __future__ import annotations

import pytest

pytest.importorskip('pygam')

from ml_toolkit.models._tabular._interpretable._gam import PyGAMClassifier, PyGAMRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_PARAMS = {'lam': 0.6}


class TestPyGAMRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = PyGAMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = PyGAMRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_categorical_features_excluded(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = PyGAMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert 'cat_col' not in model._num_feats_
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = PyGAMRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestPyGAMClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = PyGAMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = PyGAMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = PyGAMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)

    def test_predict_proba_is_1d_and_valid(self, classification_data):
        """LogisticGAM.predict_proba() отдаёт 1D массив, не 2D как sklearn.

        Проверяем, что адаптер корректно с этим работает и не индексирует [:, 1].
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = PyGAMClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_valid)
        assert proba.ndim == 1
        assert proba.shape == (len(X_valid),)
