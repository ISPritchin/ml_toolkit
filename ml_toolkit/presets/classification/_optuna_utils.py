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

from typing import Any

import optuna


def make_pruner(n_warmup_steps: int = 10) -> optuna.pruners.MedianPruner:
    """MedianPruner с прогревом: первые n_warmup_steps итераций каждого trial

    не сравниваются с другими — ранние boosting-итерации шумные, отсечение по
    ним даёт много ложных пруnов.
    """
    return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)


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
