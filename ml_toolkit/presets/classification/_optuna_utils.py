"""Общие Optuna-утилиты для Optuna-тюнинга CatBoost в пресетах high_pr_auc.

MedianPruner сам по себе ничего не отсекает: ему нужны intermediate-значения
метрики по ходу обучения (через trial.report на каждом шаге), иначе
should_prune() не имеет с чем сравнивать. `optuna-integration[catboost]` даёт
готовый CatBoostPruningCallback, но это отдельная опциональная зависимость;
CatBoostPruningCallback ниже реализует тот же протокол (CatBoost `callbacks=`
+ report/should_prune + check_pruned после fit) без неё, поверх штатного
`info.metrics[valid_name][metric]` из CatBoost >= 1.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optuna

from ml_toolkit.models._utils import resolve_pruner

# (kind, (low, high), suggest_kwargs) — единственный источник границ для
# iterations/max_depth/learning_rate/l2_leaf_reg/subsample/min_data_in_leaf,
# использованных с идентичными значениями в доброй дюжине ансамблевых
# пресетов high_pr_auc (было продублировано построчно в каждом).
_ARCH_BOUNDS: dict[str, tuple[str, tuple[float, float], dict[str, Any]]] = {
    'iterations': ('int', (300, 1000), {'step': 100}),
    'max_depth': ('int', (3, 7), {}),
    'learning_rate': ('float', (0.001, 0.3), {'log': True}),
    'l2_leaf_reg': ('float', (1e-5, 10.0), {'log': True}),
    'subsample': ('float', (0.5, 1.0), {}),
    'min_data_in_leaf': ('int', (1, 30), {}),
}


def catboost_arch_space(
    trial: optuna.Trial,
    custom: dict[str, Any] | None = None,
    keys: Sequence[str] = tuple(_ARCH_BOUNDS),
) -> dict[str, Any]:
    """Стандартный Optuna search space для архитектурных гиперпараметров CatBoost.

    `keys` ограничивает тюнящееся подмножество из iterations/max_depth/
    learning_rate/l2_leaf_reg/subsample/min_data_in_leaf (по умолчанию — все
    шесть; например, Stacking тюнит только 4 из них — max_depth/learning_rate
    у него общие для всех уровней стека и не входят в этот search space).

    `custom`, если задан, — результат пользовательского `param_space`:
    любой ключ из `keys`, присутствующий в `custom`, берётся оттуда как есть
    (без вызова trial.suggest_*); отсутствующие тюнятся дефолтным способом.
    Так реализована per-key merge семантика part_space в части пресетов
    (cascade/hard_negative_mining и т.п.); пресеты с full-override семантикой
    (param_space целиком заменяет возврат этой функции) вызывают её вовсе
    без `custom`, только когда param_space is None.

    Границы параметров, отличающихся от этого набора в отдельных пресетах
    (например, bagging_pu/co_teaching/spy_pu — более дешёвые iterations для
    многократного бэггинга, pu_learning/precision_at_k — свои learning_rate/
    min_data_in_leaf), — намеренные варианты и через этот helper не заданы.
    """
    custom = custom or {}
    out: dict[str, Any] = {}
    for key in keys:
        if key in custom:
            out[key] = custom[key]
            continue
        kind, bounds, kw = _ARCH_BOUNDS[key]
        if kind == 'int':
            out[key] = trial.suggest_int(key, *bounds, **kw)
        else:
            out[key] = trial.suggest_float(key, *bounds, **kw)
    return out


def make_pruner(
    spec: str | optuna.pruners.BasePruner | None = 'none',
    n_warmup_steps: int = 10,
) -> optuna.pruners.BasePruner:
    """Резолвит прунер пресета из `optuna_pruner`-параметра конструктора.

    `spec='none'` (дефолт) — прунинг выключен (`NopPruner`): раньше был жёстко
    включён MedianPruner всегда, без возможности отключить, что могло
    отсекать trial'ы с медленным, но в итоге лучшим сходом (низкий
    learning_rate/высокая регуляризация). `spec=None`/`'median'` — MedianPruner
    с прогревом `n_warmup_steps` (первые N итераций каждого trial не
    сравниваются с другими — ранние boosting-итерации шумные, отсечение по
    ним даёт много ложных прунов); остальные строковые алиасы
    ('hyperband'/'percentile'/'successive_halving') и готовый экземпляр
    optuna.pruners.BasePruner — через тот же resolve_pruner, что и в
    ml_toolkit.models (без прогрева — он специфичен для MedianPruner).
    """
    if spec in (None, 'median'):
        return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
    return resolve_pruner({'optuna_pruner': spec})


class CatBoostPruningCallback:
    """CatBoost `callbacks=[...]` callback, репортящий eval_metric в Optuna trial.

    Использование::

        cb = CatBoostPruningCallback(trial, 'PRAUC')
        model.fit(tr_pool, eval_set=va_pool, callbacks=[cb], verbose=False)
        cb.check_pruned()  # бросает optuna.TrialPruned, если trial был отсечён

    Не бросает исключение прямо из after_iteration — CatBoost должен корректно
    остановить обучение (after_iteration → False) и вернуть управление; сам
    TrialPruned поднимается уже после model.fit() через check_pruned().
    """

    def __init__(self, trial: optuna.Trial, metric: str, valid_name: str = 'validation') -> None:
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._pruned = False
        self._message = ''

    def after_iteration(self, info: Any) -> bool:  # noqa: ANN401 — info: непубличный C++ объект CatBoost без экспортируемого типа
        values = info.metrics.get(self._valid_name, {}).get(self._metric)
        if not values:
            return True
        self._trial.report(values[-1], step=info.iteration)
        if self._trial.should_prune():
            self._pruned = True
            self._message = f'Trial was pruned at iteration {info.iteration}.'
            return False
        return True

    def check_pruned(self) -> None:
        if self._pruned:
            raise optuna.TrialPruned(self._message)
