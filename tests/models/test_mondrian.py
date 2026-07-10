"""Тесты для MondrianForestRegressor/MondrianForestClassifier (ml_toolkit/models/_mondrian.py).

Пакет skgarden/mondrian-forest не входит в обязательные зависимости проекта — весь модуль
пропускается через importorskip, если ни один из вариантов не установлен (scikit-garden,
последний релиз 2018, не собирается в современном окружении с изолированной сборкой — на
момент написания тестов проверить вживую не удалось).
"""

from __future__ import annotations

import pytest

pytest.importorskip('skgarden')

from ml_toolkit.models._mondrian import MondrianForestClassifier, MondrianForestRegressor  # noqa: E402
from tests.models.conftest import assert_valid_predictions, assert_valid_proba  # noqa: E402

FAST_PARAMS = {'n_estimators': 10, 'max_depth': 5, 'random_state': 42}


class TestMondrianForestRegressor:
    def test_fit_predict_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = MondrianForestRegressor(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = MondrianForestRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestMondrianForestClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = MondrianForestClassifier(params=FAST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = MondrianForestClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
