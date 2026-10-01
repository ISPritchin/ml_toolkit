"""Единый механизм кастомных лоссов с тюнингом их параметров через Optuna.

CatBoost/LightGBM/XGBoost используют РАЗНЫЕ сигнатуры кастомных лоссов:

- CatBoost (бинарная классификация/регрессия):
  ``calc_ders_range(predictions, targets, weights) -> [(der1, der2), ...]``
  — вызывается один раз на весь батч, der1=-dL/df (антиградиент), der2=-d²L/df².
- CatBoost (мультикласс):
  ``calc_ders_multi(approx, target, weight) -> (der1: list[float], der2: list[list[float]])``
  — вызывается ПО ОДНОМУ РАЗУ НА ОБЪЕКТ (не векторизовано), der2 — полная
  n_classes x n_classes матрица (см. ml_toolkit/presets/classification/
  multiclass_imbalance/_custom_loss_base.py — тот же контракт, что здесь).
- LightGBM: ``objective(y_true, y_pred) -> (grad, hess)`` — обычный градиент/
  гессиан (БЕЗ минуса, в отличие от CatBoost). Для мультикласса y_pred/grad/hess
  — 2D массив (n_samples, n_classes) (см. lightgbm.sklearn.LGBMModel.fit docstring,
  подтверждено эмпирически на lightgbm==4.6.0 — никакого флэттенинга, чистый 2D).
- XGBoost: наши адаптеры (``_tabular/_boosting/_xgboost.py``) используют sklearn-
  обёртку (``XGBRegressor``/``XGBClassifier(objective=...)``), НЕ нативный
  Learning API (``xgb.train``) — поэтому сигнатура кастомного objective здесь
  такая же, как у LightGBM: ``objective(y_true, y_pred) -> (grad, hess)``, БЕЗ
  ``DMatrix`` — sklearn-обёртка сама достаёт label и прокидывает их первым
  аргументом (``xgboost.sklearn._objective_decorator``: ``inner(preds, dmatrix) =
  func(dmatrix.get_label(), preds)``). Отдельный ``objective(y_pred, dtrain)``
  с ``dtrain.get_label()`` — конвенция только нативного ``xgb.train()``,
  который этот модуль не использует.

Ключевое наблюдение: CatBoost'овские der1/der2 — это ровно ``-grad, -hess`` в
терминах LightGBM/XGBoost. Поэтому ЛЮБОЙ существующий лосс из ml_toolkit.losses
(бинарный, calc_ders_range) или ml_toolkit.presets.classification.multiclass_imbalance
(мультикласс, calc_ders_multi) можно переиспользовать для всех трёх фреймворков
через тонкие адаптеры ниже — переписывать сами лоссы не нужно.

LossSpec — тот же паттерн, что уже был в ml_toolkit.presets.regression._custom_loss_base
(_LossSpec) и ml_toolkit.presets.classification.multiclass_imbalance._custom_loss_base
(_MulticlassLossSpec), вынесенный в общее место, чтобы его мог использовать не
только слой пресетов, но и базовые адаптеры ml_toolkit.models (через
model_settings['loss_spec']).

Ограничение (мультикласс, LightGBM/XGBoost): калибруются только ДИАГОНАЛЬНЫЕ
элементы der2 (per-class Hessian) — оба фреймворка не поддерживают кросс-классовые
члены гессиана в custom objective API, это ограничение их API, не этого модуля
(нативный multiclass-лосс самого CatBoost использует полную матрицу).

XGBoost: реализовано по документированному API sklearn-обёртки, НЕ верифицировано
вживую (xgboost не установлен в окружении разработки этого модуля, см. также
tests/models/_tabular/_boosting/test_xgboost.py — весь файл через
pytest.importorskip('xgboost')) — проверьте на реальных данных перед
продакшн-использованием, особенно мультикласс (форма y_pred, которую XGBoost
передаёт в custom objective при multi:softprob, зависит от версии — см.
to_xgboost_objective).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class _CalcDersRangeLoss(Protocol):
    """Бинарная классификация / регрессия — см. ml_toolkit.losses, ml_toolkit.presets.regression._losses."""

    def calc_ders_range(
        self, predictions: Any, targets: Any, weights: Any,
    ) -> list[tuple[float, float]]: ...


class _CalcDersMultiLoss(Protocol):
    """Мультикласс — см. ml_toolkit.presets.classification.multiclass_imbalance._custom_loss_base."""

    def calc_ders_multi(
        self, approx: list[float], target: float, weight: float,
    ) -> tuple[list[float], list[list[float]]]: ...


CustomLoss = _CalcDersRangeLoss | _CalcDersMultiLoss


@dataclass(frozen=True)
class LossSpec:
    """Описание одного кастомного лосса для model_settings['loss_spec'].

    Parameters
    ----------
    name:
        Короткое имя для логов.
    loss_cls:
        Класс лосса — calc_ders_range-совместимый (бинарная/регрессия) или
        calc_ders_multi-совместимый (мультикласс). Конструируется как
        ``loss_cls(**loss_params)`` на каждом Optuna trial.
    param_bounds:
        ``{имя_параметра: (low, high)}`` — границы Optuna ``trial.suggest_float``
        для параметров лосса. Пустой словарь — у лосса нет тюнящихся параметров
        (используется с фиксированными значениями из ``loss_params`` конструктора).

    Example::

        from ml_toolkit.losses import FocalLoss
        from ml_toolkit.models._loss_spec import LossSpec

        model_settings = {
            'loss_spec': LossSpec(name='focal', loss_cls=FocalLoss,
                                   param_bounds={'gamma': (0.5, 5.0), 'alpha': (0.1, 0.9)}),
        }
        model = CatBoostClassifier(n_optuna_trials=50, model_settings=model_settings)
        # то же самое — LightGBMClassifier, XGBoostClassifier, *Regressor

    """

    name: str
    loss_cls: type
    param_bounds: dict[str, tuple[float, float]]


def suggest_loss_params(
    spec: LossSpec, trial: Any, custom: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Сэмплирует параметры лосса для одного Optuna trial.

    ``custom`` — то, что вернула пользовательская ``param_space`` (см. другие
    Optuna-адаптеры пакета): ключи, уже заданные там явно, не тюнятся заново.
    """
    custom = custom or {}
    return {
        k: (custom[k] if k in custom else trial.suggest_float(k, *bounds))
        for k, bounds in spec.param_bounds.items()
    }


def build_loss(spec: LossSpec, loss_params: dict[str, float]) -> CustomLoss:
    """Конструирует объект лосса из LossSpec + сэмплированных параметров."""
    return spec.loss_cls(**loss_params)


def is_multiclass_loss(loss_obj: CustomLoss) -> bool:
    return hasattr(loss_obj, 'calc_ders_multi')


# ── CatBoost ─────────────────────────────────────────────────────────────────

def to_catboost_loss(loss_obj: CustomLoss) -> CustomLoss:
    """CatBoost принимает calc_ders_range/calc_ders_multi объекты нативно — без обёртки."""
    return loss_obj


# ── LightGBM ─────────────────────────────────────────────────────────────────

def to_lightgbm_objective(loss_obj: CustomLoss) -> Callable:
    """objective(y_true, y_pred) -> (grad, hess) — см. докстринг модуля про знаки/форму."""
    if is_multiclass_loss(loss_obj):
        def _multiclass_obj(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            n, k = y_pred.shape
            grad = np.empty((n, k), dtype=np.float64)
            hess = np.empty((n, k), dtype=np.float64)
            for i in range(n):
                der1, der2 = loss_obj.calc_ders_multi(y_pred[i].tolist(), float(y_true[i]), 1.0)
                grad[i] = [-v for v in der1]
                hess[i] = [-der2[c][c] for c in range(k)]
            return grad, hess
        return _multiclass_obj

    def _binary_or_reg_obj(
        y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ders = loss_obj.calc_ders_range(np.asarray(y_pred), np.asarray(y_true), weight)
        der1, der2 = zip(*ders, strict=False)
        return -np.asarray(der1, dtype=np.float64), -np.asarray(der2, dtype=np.float64)
    return _binary_or_reg_obj


# ── XGBoost ──────────────────────────────────────────────────────────────────

def to_xgboost_objective(loss_obj: CustomLoss) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """objective(y_true, y_pred) -> (grad, hess) — конвенция sklearn-обёртки XGBoost
    (``XGBRegressor``/``XGBClassifier(objective=...)``, которую используют адаптеры
    этого пакета), БЕЗ ``DMatrix`` — см. докстринг модуля. Без поддержки sample weight
    (sklearn-обёртка не прокидывает его в кастомный objective). Мультикласс не
    верифицирован вживую — см. докстринг модуля."""
    if is_multiclass_loss(loss_obj):
        def _multiclass_obj(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            n, k = y_pred.shape
            grad = np.empty((n, k), dtype=np.float64)
            hess = np.empty((n, k), dtype=np.float64)
            for i in range(n):
                der1, der2 = loss_obj.calc_ders_multi(y_pred[i].tolist(), float(y_true[i]), 1.0)
                grad[i] = [-v for v in der1]
                hess[i] = [-der2[c][c] for c in range(k)]
            return grad, hess
        return _multiclass_obj

    def _binary_or_reg_obj(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ders = loss_obj.calc_ders_range(np.asarray(y_pred), np.asarray(y_true), None)
        der1, der2 = zip(*ders, strict=False)
        return -np.asarray(der1, dtype=np.float64), -np.asarray(der2, dtype=np.float64)
    return _binary_or_reg_obj
