"""Общий движок для пресетов вида «CatBoostRegressor + один (возможно, параметризованный) лосс».

Зеркало ml_toolkit/presets/classification/high_pr_auc/_custom_loss_base.py, но с
двумя отличиями, специфичными для регрессии:

1. Лосс бывает двух видов — `_LossSpec.loss_cls` (Python calc_ders_range-класс,
   как в classification) ИЛИ `_LossSpec.loss_function` (фабрика, строящая имя
   встроенного параметризованного лосса CatBoost вида `'Huber:delta=1.0'` /
   `'Quantile:alpha=0.3'` / `'Tweedie:variance_power=1.5'` — для них Python-объект
   не нужен и был бы медленнее нативной C++ реализации). Ровно один из двух
   должен быть задан.
2. eval_metric у CatBoost всегда 'MAE' (совпадает с дефолтной reg_metric проекта,
   см. CLAUDE.md) — используется и для pruning (CatBoost считает его нативно
   независимо от loss_function), и как метрика отбора trial по умолчанию.
   Подклассы могут переопределить `_trial_score` (например, quantile-лоссам
   нужен pinball loss на своём квантиле, а не MAE) — тогда `_direction` тоже
   переопределяется при необходимости ('minimize' по умолчанию).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression._base import BasePreset
from ml_toolkit.presets.regression._optuna_utils import (
    CatBoostPruningCallback,
    make_pruner,
)

logger = logging.getLogger(__name__)

_DEFAULT_ARCH_PARAMS: dict[str, Any] = {
    'iterations': 800,
    'max_depth': 6,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'min_data_in_leaf': 5,
    'early_stopping_rounds': 80,
    'verbose': 0,
}


@dataclass(frozen=True)
class _LossSpec:
    """Описание одного лосса для _CustomLossRegressorBase.

    name:
        Короткое имя для логов.
    param_bounds:
        {имя_параметра_лосса: (low, high)} — границы Optuna suggest_float.
        Пустой словарь — у лосса нет тюнящихся параметров (тюнится только
        архитектура CatBoost).
    loss_cls:
        Python calc_ders_range-класс (см. _losses.py). Взаимоисключающе с loss_function.
    loss_function:
        Callable(loss_params) -> str, строит имя встроенного лосса CatBoost
        (например, `lambda p: f"Huber:delta={p['delta']}"`). Взаимоисключающе с loss_cls.
    """

    name: str
    param_bounds: dict[str, tuple[float, float]]
    loss_cls: type | None = None
    loss_function: Callable[[dict[str, float]], str] | None = None

    def __post_init__(self) -> None:
        if (self.loss_cls is None) == (self.loss_function is None):
            raise ValueError('_LossSpec: ровно один из loss_cls/loss_function должен быть задан')


class _CustomLossRegressorBase(BasePreset):
    """Общая fit/tune/predict логика для CatBoostRegressor с одним (возможно параметризованным) лоссом.

    Подкласс обязан задать класс-атрибут `_loss_spec: _LossSpec` и передать в
    `super().__init__` уже собранный `loss_params: dict[str, float]` со своими
    именованными параметрами лосса (может быть пустым словарём).
    """

    _loss_spec: _LossSpec

    def __init__(
        self,
        loss_params: dict[str, float],
        base_params: dict[str, Any] | None,
        n_optuna_trials: int,
        optuna_timeout: int | None,
        random_seed: int,
        cat_features: list[str] | None,
        selected_features: list[str] | None,
        param_space: Callable[[Any], dict[str, Any]] | None = None,
        optuna_verbose: bool = False,
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.loss_params = dict(loss_params)
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
        self.param_space = param_space
        self.optuna_verbose = optuna_verbose
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

    # ── хуки, переопределяемые подклассами ──────────────────────────────────

    def _build_loss(self, loss_params: dict[str, float], *, tr_pool: Any) -> Any:
        """Строит объект/строку лосса. tr_pool — для лоссов, которым нужна

        статистика обучающей выборки (например, WAPE — глобальный denom по
        train-таргету); большинству лоссов он не нужен.
        """
        spec = self._loss_spec
        if spec.loss_cls is not None:
            return spec.loss_cls(**loss_params)
        return spec.loss_function(loss_params)

    def _trial_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Метрика отбора Optuna trial (и финального логирования). По умолчанию — MAE."""
        return float(mean_absolute_error(y_true, y_pred))

    _direction: str = 'minimize'

    # ── обучение одной модели ───────────────────────────────────────────────

    def _fit_model(
        self,
        tr_pool: Any,
        va_pool: Any,
        arch_params: dict,
        loss_params: dict[str, float],
        callbacks: list | None = None,
    ) -> Any:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            loss_function=self._build_loss(loss_params, tr_pool=tr_pool),
            eval_metric='MAE',
            **arch_params,
        )
        model.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=callbacks)
        return model

    def _tune(self, tr_pool: Any, va_pool: Any, y_va: np.ndarray) -> tuple[Any, dict]:
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        esr = _DEFAULT_ARCH_PARAMS['early_stopping_rounds']
        loss_keys = list(self._loss_spec.param_bounds)

        def objective(trial: optuna.Trial) -> float:
            custom = self.param_space(trial) if self.param_space is not None else {}

            loss_p = {
                k: (custom[k] if k in custom else trial.suggest_float(k, *self._loss_spec.param_bounds[k]))
                for k in loss_keys
            }

            def arch_val(key: str, suggest: Callable[[], Any]) -> Any:
                return custom[key] if key in custom else suggest()

            arch_p = {
                'iterations': arch_val('iterations',
                    lambda: trial.suggest_int('iterations', 300, 1000, step=100)),
                'max_depth': arch_val('max_depth',
                    lambda: trial.suggest_int('max_depth', 3, 7)),
                'learning_rate': arch_val('learning_rate',
                    lambda: trial.suggest_float('learning_rate', 0.01, 0.2, log=True)),
                'l2_leaf_reg': arch_val('l2_leaf_reg',
                    lambda: trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True)),
                'subsample': arch_val('subsample',
                    lambda: trial.suggest_float('subsample', 0.5, 1.0)),
                'min_data_in_leaf': arch_val('min_data_in_leaf',
                    lambda: trial.suggest_int('min_data_in_leaf', 1, 30)),
                'early_stopping_rounds': custom.get('early_stopping_rounds', esr),
                'random_seed': custom.get('random_seed', self.random_seed),
                'verbose': custom.get('verbose', 0),
            }
            trial.set_user_attr('loss_p', loss_p)
            trial.set_user_attr('arch_p', arch_p)
            pruning_cb = CatBoostPruningCallback(trial, 'MAE')
            m = self._fit_model(tr_pool, va_pool, arch_p, loss_p, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict(va_pool)
            return self._trial_score(y_va, p)

        study = optuna.create_study(
            direction=self._direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(),
        )
        # Как и в classification-версии: первый trial — конструкторские значения,
        # чтобы не потерялись среди случайных стартовых точек. Пропускается при
        # кастомном param_space — границы неизвестны заранее, enqueue может
        # оказаться вне новых границ (см. подробный комментарий в
        # classification/high_pr_auc/_custom_loss_base.py).
        if self.param_space is None:
            study.enqueue_trial({
                **self.loss_params,
                'iterations':       _DEFAULT_ARCH_PARAMS['iterations'],
                'max_depth':        _DEFAULT_ARCH_PARAMS['max_depth'],
                'learning_rate':    _DEFAULT_ARCH_PARAMS['learning_rate'],
                'l2_leaf_reg':      _DEFAULT_ARCH_PARAMS['l2_leaf_reg'],
                'subsample':        _DEFAULT_ARCH_PARAMS['subsample'],
                'min_data_in_leaf': _DEFAULT_ARCH_PARAMS['min_data_in_leaf'],
            })
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best_loss = dict(study.best_trial.user_attrs['loss_p'])
        best_arch = dict(study.best_trial.user_attrs['arch_p'])
        model = self._fit_model(tr_pool, va_pool, best_arch, best_loss)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return model, {**best_loss, **best_arch}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> _CustomLossRegressorBase:
        from catboost import Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        y_tr = y_train.values
        y_va = y_valid.values
        tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        if self.n_optuna_trials > 0:
            self._model, best = self._tune(tr_pool, va_pool, y_va)
            self.best_params_ = best
        else:
            arch_params = {**(self.base_params or _DEFAULT_ARCH_PARAMS), 'random_seed': self.random_seed}
            self._model = self._fit_model(tr_pool, va_pool, arch_params, self.loss_params)
            self.best_params_ = {**self.loss_params, **arch_params}

        self.train_pred_ = self._model.predict(tr_pool)
        self.valid_pred_ = self._model.predict(va_pool)
        score = self._trial_score(y_va, self.valid_pred_)
        logger.info(
            '[%s] params=%s  val score=%.4f',
            self._loss_spec.name,
            {k: self.best_params_.get(k) for k in self._loss_spec.param_bounds},
            score,
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict(pool)
