"""Pickle-контракт BasePreset (regression): fit() -> save() -> load() -> predict тот же.

save()/load() — общая реализация BaseModel (ml_toolkit/models/_base.py), наследуемая
без переопределения каждым пресетом регрессии (см. tests/models/test_pickle.py для
адаптеров моделей). Покрывает все 13 классов ml_toolkit.presets.regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models import CatBoostRegressor
from ml_toolkit.presets.regression import (
    AsymmetricCostRegressor,
    ConformalRegressionWrapper,
    HuberOptunaRegressor,
    JackknifePlusRegressor,
    LogCoshRegressor,
    NGBoostPreset,
    QuantileEnsembleRegressor,
    QuantileHuberRegressor,
    RegressionByBinnedClassification,
    RelativeErrorRegressor,
    TargetTransformOptunaRegressor,
    TrimmedLossRegressor,
    TweedieOptunaRegressor,
)
from tests.presets.regression.conftest import BASE_PARAMS


def _assert_pickle_roundtrip(model, X_valid, tmp_path) -> None:
    pred_before = model.predict(X_valid)
    path = tmp_path / 'model.pkl'
    model.save(path)
    loaded = type(model).load(path)
    assert isinstance(loaded, type(model))
    pred_after = loaded.predict(X_valid)
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-9, atol=1e-12)


class TestRegressionPresetsPickle:
    def test_target_transform_optuna(self, positive_regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = positive_regression_data
        model = TargetTransformOptunaRegressor(transforms=['log1p'], base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_huber_optuna(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = HuberOptunaRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_log_cosh(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LogCoshRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_trimmed_loss(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = TrimmedLossRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_relative_error(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RelativeErrorRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_quantile_huber(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileHuberRegressor(quantile=0.5, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_tweedie_optuna(self, positive_regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = positive_regression_data
        model = TweedieOptunaRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_asymmetric_cost(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = AsymmetricCostRegressor(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_binned_classification(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        params = {'iterations': 80, 'max_depth': 4, 'early_stopping_rounds': 30}
        model = RegressionByBinnedClassification(n_bins=16, base_params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_jackknife_plus(self, regression_data, tmp_path):
        X_train, y_train, _, _ = regression_data
        params = {'iterations': 80, 'depth': 4, 'verbose': 0}
        model = JackknifePlusRegressor(n_folds=5, base_params=params)
        model.fit(X_train, y_train)
        _assert_pickle_roundtrip(model, X_train, tmp_path)

    def test_ngboost(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        params = {'n_estimators': 60, 'early_stopping_rounds': 20}
        model = NGBoostPreset(dist='Normal', **params)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_quantile_ensemble(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileEnsembleRegressor(quantiles=[0.2, 0.5, 0.8], base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_conformal_wrapper(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        base = CatBoostRegressor(params={'iterations': 80, 'depth': 4, 'verbose': 0, 'random_seed': 42})
        model = ConformalRegressionWrapper(base)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_pickle_roundtrip(model, X_valid, tmp_path)
