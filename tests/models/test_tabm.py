"""Тесты для TabMRegressor/TabMClassifier (ml_toolkit/models/_tabm.py).

Пакеты torch и tabm не входят в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если они не установлены.
"""

from __future__ import annotations

import pytest

pytest.importorskip('torch')
pytest.importorskip('tabm')

from ml_toolkit.models._tabm import TabMClassifier, TabMRegressor  # noqa: E402
from tests.models.conftest import assert_valid_predictions, assert_valid_proba  # noqa: E402

FAST_PARAMS = {'k': 8, 'd_block': 32, 'n_blocks': 1, 'dropout': 0.0, 'lr': 1e-3, 'weight_decay': 1e-4}
FAST_SETTINGS = {'n_epochs_final': 15, 'patience': 3, 'n_epochs_per_trial': 8}


class TestTabMRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = TabMRegressor(params=FAST_PARAMS, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = TabMRegressor(n_optuna_trials=2, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestTabMClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = TabMClassifier(params=FAST_PARAMS, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_cat_features(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = TabMClassifier(params=FAST_PARAMS, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = TabMClassifier(n_optuna_trials=2, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
