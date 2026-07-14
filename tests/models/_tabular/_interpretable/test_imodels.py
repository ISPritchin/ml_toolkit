"""Тесты для IModelsRegressor/IModelsClassifier (ml_toolkit/models/_tabular/_interpretable/_imodels.py).

Пакет imodels не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.
"""

from __future__ import annotations

import pytest

pytest.importorskip('imodels')

from ml_toolkit.models._tabular._interpretable._imodels import IModelsClassifier, IModelsRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FIGS_PARAMS = {'max_rules': 10, 'max_trees': 5}
SKOPE_PARAMS = {'n_estimators': 10, 'max_depth': 3, 'random_state': 42}
BRL_PARAMS = {'listlengthprior': 3, 'listwidthprior': 1}


class TestIModelsRegressor:
    def test_fit_predict_explicit_params_figs(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = IModelsRegressor(params=FIGS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FIGS_PARAMS

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = IModelsRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = IModelsRegressor(params=FIGS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)


class TestIModelsClassifierFigs:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params=FIGS_PARAMS, model_settings={'name': 'figs'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(n_optuna_trials=2, model_settings={'name': 'figs'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = IModelsClassifier(params=FIGS_PARAMS, model_settings={'name': 'figs'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)


class TestIModelsClassifierSkopeRules:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params=SKOPE_PARAMS, model_settings={'name': 'skope_rules'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)


class TestIModelsClassifierBRL:
    """BRL требует one-hot дискретизированные бины — регрессионные тесты на этот фикс."""

    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params=BRL_PARAMS, model_settings={'name': 'brl'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model._brl_discretizer_ is not None

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(n_optuna_trials=2, model_settings={'name': 'brl'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_predict_uses_discretized_space(self, classification_data):
        """predict_proba() на новых данных должен пройти через тот же дискретизатор, что и train/valid.

        Иначе BRL получит непрерывные признаки вместо one-hot бинов.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params=BRL_PARAMS, model_settings={'name': 'brl'})
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_train)
        assert proba.shape == (len(X_train),)
        assert ((proba >= 0) & (proba <= 1)).all()


class TestIModelsClassifierRipper:
    def test_ripper_raises_clear_import_error(self, classification_data):
        """RIPPERClassifier отсутствует в текущем imodels (2.0.4) — должен падать с понятным сообщением.

        А не с обычным confusing ImportError из самого imodels.
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params={'k': 2}, model_settings={'name': 'ripper'})
        with pytest.raises(ImportError, match='RIPPERClassifier'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestIModelsClassifierDispatch:
    def test_unknown_name_raises(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = IModelsClassifier(params={}, model_settings={'name': 'not_a_real_model'})
        with pytest.raises(ValueError, match='Unknown imodels classifier'):
            model.fit(X_train, y_train, X_valid, y_valid)
