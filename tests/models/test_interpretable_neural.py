"""Тесты для InterpretableNeuralRegressor/InterpretableNeuralClassifier
(ml_toolkit/models/_interpretable_neural.py): GAMINET/NAM.

PyTorch не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.
"""

from __future__ import annotations

import pytest

pytest.importorskip('torch')

from ml_toolkit.models._interpretable_neural import (  # noqa: E402
    InterpretableNeuralClassifier,
    InterpretableNeuralRegressor,
)
from tests.models.conftest import assert_valid_predictions, assert_valid_proba  # noqa: E402

REG_PARAMS = {'hidden_dim': 16, 'n_layers': 1, 'lr': 1e-2, 'n_epochs': 30, 'n_interactions': 0}
CLS_PARAMS = {'C': 1.0}


class TestInterpretableNeuralRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableNeuralRegressor(params=REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == REG_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableNeuralRegressor(n_optuna_trials=1)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestInterpretableNeuralClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableNeuralClassifier(params=CLS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_explicit_params_do_not_crash_on_max_iter_class_weight(self, classification_data):
        """Regression test: LogisticRegression(**self.params, max_iter=2000, class_weight='balanced')
        падал с TypeError, если self.params уже содержал max_iter/class_weight.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        params = {'C': 2.0, 'max_iter': 500, 'class_weight': None}
        model = InterpretableNeuralClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['max_iter'] == 500
        assert model.best_params_['class_weight'] is None

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableNeuralClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
