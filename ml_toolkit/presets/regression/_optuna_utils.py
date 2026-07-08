"""Общие Optuna-утилиты для Optuna-тюнинга CatBoost в пресетах регрессии.

Зеркало ml_toolkit/presets/classification/_optuna_utils.py — то же MedianPruner
+ CatBoostPruningCallback, но без завязки на classification-namespace (импорт
из classification._optuna_utils внутри presets/regression выглядел бы странно
для читателя, хотя реализация была бы идентичной).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optuna

# (kind, (low, high), suggest_kwargs) — тот же набор границ архитектурных
# гиперпараметров CatBoost, что и в classification._optuna_utils; для регрессии
# используется идентично (iterations/max_depth/learning_rate/l2_leaf_reg/
# subsample/min_data_in_leaf не зависят от типа задачи).
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

    `custom`, если задан, — результат пользовательского `param_space`: любой
    ключ из `keys`, присутствующий в `custom`, берётся оттуда как есть (без
    вызова trial.suggest_*); отсутствующие тюнятся дефолтным способом.
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


def make_pruner(n_warmup_steps: int = 10) -> optuna.pruners.MedianPruner:
    """MedianPruner с прогревом: первые n_warmup_steps итераций каждого trial

    не сравниваются с другими — ранние boosting-итерации шумные, отсечение по
    ним даёт много ложных прунов.
    """
    return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)


class CatBoostPruningCallback:
    """CatBoost `callbacks=[...]` callback, репортящий eval_metric в Optuna trial.

    Использование::

        cb = CatBoostPruningCallback(trial, 'MAE')
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

    def after_iteration(self, info: Any) -> bool:
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
