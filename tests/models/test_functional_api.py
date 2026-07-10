"""Тесты для диспетчеризации функционального API (ml_toolkit/models/__init__.py):
train_regression_model / train_classification_model / make_predict_fn / _adapter.

Модель-специфичное поведение (Optuna, cat_features, baseline_col, ...) уже
покрыто test_catboost.py/test_lightgbm.py/.../test_forest.py — здесь только
диспетчеризация по имени, общая для всех 30 адаптеров.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models import (
    make_predict_fn,
    train_classification_model,
    train_regression_model,
)


class TestUnknownModelName:
    def test_train_regression_model_unknown_name_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        with pytest.raises(ValueError, match='Unknown model'):
            train_regression_model(
                name='not_a_real_model', X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid, X_inference=X_valid,
                selected_features=list(X_train.columns), cat_features=[],
                model_settings={}, n_optuna_trials=2,
            )

    def test_train_classification_model_unknown_name_raises(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        with pytest.raises(ValueError, match='Unknown model'):
            train_classification_model(
                name='not_a_real_model', X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid, X_inference=X_valid,
                selected_features=list(X_train.columns), cat_features=[],
                n_optuna_trials=2, model_settings={},
            )

    def test_make_predict_fn_unknown_name_raises(self):
        with pytest.raises(ValueError, match='Unknown model'):
            make_predict_fn('not_a_real_model', model=None, task='regression', selected_features=[])


class TestMakePredictFn:
    def test_tree_based_models_return_none(self):
        """CatBoost/LightGBM/forest поддерживают SHAP нативно — своя predict_fn не нужна."""
        for name in ('catboost', 'lightgbm', 'random_forest', 'extra_trees', 'decision_tree'):
            assert make_predict_fn(name, model=None, task='regression', selected_features=[]) is None

    def test_hist_gbm_returns_callable_for_permutation_importance(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        from ml_toolkit.models._hist_gbm import HistGBMRegressor

        model = HistGBMRegressor(params={'max_iter': 30, 'max_depth': 3, 'random_state': 42})
        model.fit(X_train, y_train, X_valid, y_valid)

        predict_fn = make_predict_fn('hist_gbm', model._model, task='regression', selected_features=list(X_train.columns))
        assert callable(predict_fn)
        pred = predict_fn(X_valid)
        assert pred.shape == (len(X_valid),)


class TestModelDispatchAcrossFamilies:
    """Один и тот же train_regression_model/train_classification_model работает
    для разных семейств адаптеров (boosting vs sklearn forest) без изменения кода вызывающей стороны.
    """

    @pytest.mark.parametrize('name', ['catboost', 'lightgbm', 'random_forest'])
    def test_train_regression_model_returns_consistent_shapes(self, name, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        ms = {'name': name} if name in ('random_forest', 'extra_trees') else {}
        _, train_pred, valid_pred, infer_pred, best_params = train_regression_model(
            name=name, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            model_settings=ms, n_optuna_trials=2,
        )
        assert train_pred.shape == (len(X_train),)
        assert valid_pred.shape == (len(X_valid),)
        assert infer_pred.shape == (len(X_valid),)
        assert isinstance(best_params, dict)

    @pytest.mark.parametrize('name', ['catboost', 'lightgbm', 'random_forest'])
    def test_train_classification_model_returns_valid_proba(self, name, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        ms = {'name': name} if name in ('random_forest', 'extra_trees') else {}
        _, train_proba, val_proba, infer_proba, best_params = train_classification_model(
            name=name, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            n_optuna_trials=2, model_settings=ms,
        )
        assert np.all((infer_proba >= 0) & (infer_proba <= 1))
        assert np.all((val_proba >= 0) & (val_proba <= 1))
