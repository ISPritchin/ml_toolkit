"""Тесты для ml_toolkit/models/_loss_spec.py — единого механизма кастомных лоссов
с тюнингом параметров через Optuna, общего для CatBoost/LightGBM/XGBoost.

Эти тесты проверяют сам адаптерный слой (LossSpec, suggest_loss_params,
build_loss, to_catboost_loss/to_lightgbm_objective/to_xgboost_objective) —
то, что знак/форма grad/hess корректно получаются из calc_ders_range/
calc_ders_multi существующих лоссов (ml_toolkit.losses), а не математику
самих лоссов (она уже покрыта tests/losses/). to_xgboost_objective не требует
установленного xgboost — адаптер этого пакета использует sklearn-обёртку
(objective(y_true, y_pred), без DMatrix), поэтому модуль не импортирует xgboost
вовсе (см. docstring ml_toolkit/models/_loss_spec.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.losses import FocalLoss, LogitNormLoss
from ml_toolkit.models._loss_spec import (
    LossSpec,
    build_loss,
    is_multiclass_loss,
    suggest_loss_params,
    to_catboost_loss,
    to_lightgbm_objective,
    to_xgboost_objective,
)


class _FakeTrial:
    """Минимальная замена optuna.Trial для suggest_loss_params — без реального Optuna study."""

    def __init__(self, values: dict[str, float]):
        self.values = values

    def suggest_float(self, name: str, low: float, high: float) -> float:
        assert low <= self.values[name] <= high
        return self.values[name]


class TestLossSpecAndSuggestParams:
    def test_suggest_loss_params_uses_trial_when_no_custom(self):
        spec = LossSpec(name='focal', loss_cls=FocalLoss, param_bounds={'gamma': (1.0, 3.0), 'alpha': (0.1, 0.9)})
        trial = _FakeTrial({'gamma': 2.0, 'alpha': 0.3})
        assert suggest_loss_params(spec, trial) == {'gamma': 2.0, 'alpha': 0.3}

    def test_suggest_loss_params_custom_overrides_trial(self):
        """custom (из пользовательской param_space) выигрывает — trial.suggest_float для него не вызывается."""
        spec = LossSpec(name='focal', loss_cls=FocalLoss, param_bounds={'gamma': (1.0, 3.0), 'alpha': (0.1, 0.9)})
        trial = _FakeTrial({'alpha': 0.5})  # gamma сюда не кладём — не должен понадобиться
        result = suggest_loss_params(spec, trial, custom={'gamma': 5.0})
        assert result == {'gamma': 5.0, 'alpha': 0.5}

    def test_suggest_loss_params_empty_bounds(self):
        spec = LossSpec(name='fixed', loss_cls=FocalLoss, param_bounds={})
        assert suggest_loss_params(spec, _FakeTrial({})) == {}

    def test_build_loss_constructs_instance(self):
        spec = LossSpec(name='focal', loss_cls=FocalLoss, param_bounds={})
        loss = build_loss(spec, {'gamma': 2.0, 'alpha': 0.25})
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 2.0
        assert loss.alpha == 0.25

    def test_is_multiclass_loss(self):
        assert is_multiclass_loss(LogitNormLoss()) is True
        assert is_multiclass_loss(FocalLoss()) is False


class TestToCatBoostLoss:
    def test_passthrough(self):
        loss = FocalLoss(gamma=2.0)
        assert to_catboost_loss(loss) is loss


class TestToLightGBMObjective:
    def test_binary_matches_calc_ders_range_with_sign_flip(self):
        rng = np.random.default_rng(0)
        n = 50
        y_true = rng.integers(0, 2, size=n).astype(float)
        y_pred = rng.normal(size=n)  # raw margins

        loss = FocalLoss(gamma=2.0, alpha=0.25)
        obj = to_lightgbm_objective(loss)
        grad, hess = obj(y_true, y_pred)

        ders = loss.calc_ders_range(y_pred, y_true, None)
        der1, der2 = zip(*ders, strict=False)
        np.testing.assert_allclose(grad, -np.asarray(der1))
        np.testing.assert_allclose(hess, -np.asarray(der2))
        assert grad.shape == (n,)
        assert hess.shape == (n,)

    def test_multiclass_shape_and_sign(self):
        rng = np.random.default_rng(1)
        n, k = 30, 3
        y_true = rng.integers(0, k, size=n).astype(float)
        y_pred = rng.normal(size=(n, k))

        loss = LogitNormLoss(temperature=0.04)
        obj = to_lightgbm_objective(loss)
        grad, hess = obj(y_true, y_pred)

        assert grad.shape == (n, k)
        assert hess.shape == (n, k)
        for i in range(n):
            der1, der2 = loss.calc_ders_multi(y_pred[i].tolist(), float(y_true[i]), 1.0)
            np.testing.assert_allclose(grad[i], [-v for v in der1])
            np.testing.assert_allclose(hess[i], [-der2[c][c] for c in range(k)])


class TestToXGBoostObjective:
    """Сигнатура sklearn-обёртки XGBoost: objective(y_true, y_pred) -> (grad, hess), без DMatrix.

    Не требует установленного xgboost — сам _loss_spec.py его не импортирует
    (см. docstring модуля). Сквозной прогон с реальным XGBRegressor/XGBClassifier —
    в tests/models/_tabular/_boosting/test_xgboost.py (importorskip('xgboost')).
    """

    def test_binary_matches_calc_ders_range_with_sign_flip(self):
        rng = np.random.default_rng(2)
        n = 50
        y_true = rng.integers(0, 2, size=n).astype(float)
        y_pred = rng.normal(size=n)

        loss = FocalLoss(gamma=2.0, alpha=0.25)
        obj = to_xgboost_objective(loss)
        grad, hess = obj(y_true, y_pred)

        ders = loss.calc_ders_range(y_pred, y_true, None)
        der1, der2 = zip(*ders, strict=False)
        np.testing.assert_allclose(grad, -np.asarray(der1))
        np.testing.assert_allclose(hess, -np.asarray(der2))

    def test_multiclass_shape_and_sign(self):
        rng = np.random.default_rng(3)
        n, k = 30, 3
        y_true = rng.integers(0, k, size=n).astype(float)
        y_pred = rng.normal(size=(n, k))

        loss = LogitNormLoss(temperature=0.04)
        obj = to_xgboost_objective(loss)
        grad, hess = obj(y_true, y_pred)

        assert grad.shape == (n, k)
        assert hess.shape == (n, k)
        for i in range(n):
            der1, der2 = loss.calc_ders_multi(y_pred[i].tolist(), float(y_true[i]), 1.0)
            np.testing.assert_allclose(grad[i], [-v for v in der1])
            np.testing.assert_allclose(hess[i], [-der2[c][c] for c in range(k)])


class TestFocalLossGammaValidation:
    def test_build_loss_propagates_constructor_validation(self):
        """build_loss не глотает ValueError лосса (gamma < 1 у FocalLoss) — важно для Optuna:
        trial с плохими параметрами должен упасть явно, а не тихо построить некорректный лосс."""
        spec = LossSpec(name='focal', loss_cls=FocalLoss, param_bounds={})
        with pytest.raises(ValueError, match='gamma'):
            build_loss(spec, {'gamma': 0.5})
