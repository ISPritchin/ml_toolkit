"""Общий движок для пресетов вида «CatBoost + один кастомный Python-лосс».

FocalLossClassifier, TverskyLossClassifier, PolyLossClassifier и
AsymmetricLossClassifier различаются только классом лосса (из ml_toolkit.losses)
и границами Optuna-поиска его 1-3 параметров — вся остальная логика (fit,
Optuna-тюнинг архитектурных параметров CatBoost, pruning, predict_proba)
идентична и живёт здесь один раз. Подкласс объявляет только `_loss_spec`
(`_LossSpec`) и явные именованные kwargs своего лосса в `__init__`.

Почему eval_metric='AUC', а не 'PRAUC': eval_metric у CatBoost не зависит от
loss_function, но во всех существующих в проекте пресетах с кастомным
Python-лоссом (BoostedEnsemble, исходный AsymmetricLossClassifier) уже
закреплено 'AUC' — сохраняем то же соглашение для новых лоссов.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    make_pruner,
)

if TYPE_CHECKING:
    from catboost import CatBoostClassifier, Pool
    import optuna
    from optuna.pruners import BasePruner

logger = logging.getLogger(__name__)


class _CalcDersRangeLoss(Protocol):
    """Duck-типизированный интерфейс лоссов ml_toolkit.losses (без общего базового класса)."""

    def calc_ders_range(
        self, predictions: list[float], targets: list[float], weights: list[float] | None,
    ) -> list[tuple[float, float]]: ...


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
    """Описание одного лосса для _CustomLossClassifierBase.

    loss_cls:
        Класс лосса из ml_toolkit.losses (calc_ders_range-совместимый).
    param_bounds:
        {имя_параметра_лосса: (low, high)} — границы Optuna suggest_float.
        Ключи должны совпадать с именами kwargs конструктора loss_cls.
    name:
        Короткое имя для логов (например, 'Focal').
    """

    loss_cls: type
    param_bounds: dict[str, tuple[float, float]]
    name: str


class _CustomLossClassifierBase(BasePreset):
    """Общая fit/tune/predict логика для CatBoost с одним кастомным Python-лоссом.

    Подкласс обязан задать класс-атрибут `_loss_spec: _LossSpec` и передать в
    `super().__init__` уже собранный `loss_params: dict[str, float]` со своими
    именованными параметрами лосса.
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
        param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | BasePruner | None = 'none',
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.loss_params = dict(loss_params)
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
        self.param_space = param_space
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

    def _make_loss(
        self, loss_params: dict[str, float], *, tr_pool: Pool, arch_params: dict,
    ) -> _CalcDersRangeLoss:
        """Строит объект лосса. tr_pool/arch_params — для лоссов, которым нужна.

        статистика датасета (n_pos/n_neg) или число итераций модели (LDAMLoss);
        большинству лоссов (Focal/Tversky/Poly/Asymmetric) они не нужны, и
        параметр можно игнорировать — дефолтная реализация так и делает.
        """
        return self._loss_spec.loss_cls(**loss_params)

    def _fit_model(
        self,
        tr_pool: Pool,
        va_pool: Pool,
        arch_params: dict,
        loss_params: dict[str, float],
        callbacks: list | None = None,
    ) -> CatBoostClassifier:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function=self._make_loss(loss_params, tr_pool=tr_pool, arch_params=arch_params),
            eval_metric='AUC',
            **arch_params,
        )
        model.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=callbacks)
        return model

    def _tune(self, tr_pool: Pool, va_pool: Pool) -> tuple[CatBoostClassifier, dict]:
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        esr = _DEFAULT_ARCH_PARAMS['early_stopping_rounds']
        loss_keys = list(self._loss_spec.param_bounds)

        def objective(trial: optuna.Trial) -> float:
            # custom — то, что вернула кастомная param_space (может покрывать
            # loss-параметры, архитектурные параметры или и то, и другое сразу;
            # частично или полностью). Всё, чего в custom нет, тюнится дефолтным
            # search space (loss — по self._loss_spec.param_bounds, архитектура —
            # по фиксированным границам ниже).
            custom = self.param_space(trial) if self.param_space is not None else {}

            loss_p = {
                k: (custom[k] if k in custom else trial.suggest_float(k, *self._loss_spec.param_bounds[k]))
                for k in loss_keys
            }

            def arch_val(key: str, suggest: Callable[[], int | float]) -> int | float:
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
            # study.best_params содержит только то, что реально прошло через
            # trial.suggest_*; параметры, зафиксированные custom как голое
            # значение (не suggest), туда не попадут — поэтому сохраняем
            # собранные loss_p/arch_p целиком через user_attr и забираем их
            # из best_trial после оптимизации, а не реконструируем из best_params.
            trial.set_user_attr('loss_p', loss_p)
            trial.set_user_attr('arch_p', arch_p)
            pruning_cb = CatBoostPruningCallback(trial, 'AUC')
            m = self._fit_model(tr_pool, va_pool, arch_p, loss_p, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(va_pool.get_label(), p))

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(self.optuna_pruner),
        )
        # Первый триал — значения из __init__ (loss_params + дефолтная архитектура),
        # чтобы они не терялись молча среди случайных стартовых точек Optuna.
        # Безопасно только для дефолтного search space — значения-константы
        # гарантированно попадают в границы self._loss_spec.param_bounds/
        # _DEFAULT_ARCH_PARAMS ниже. При кастомном param_space границы неизвестны
        # заранее (мы не можем вызвать trial.suggest_* без реального trial), и
        # enqueue со старыми дефолтами может оказаться вне новых границ — Optuna
        # в этом случае всё равно "продавливает" это значение как отдельный
        # валидный trial, который на шумных/маленьких выборках может случайно
        # выиграть по метрике и молча испортить результат тюнинга. Поэтому
        # пропускаем enqueue целиком, если задан кастомный param_space.
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
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> _CustomLossClassifierBase:
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
            self._model, best = self._tune(tr_pool, va_pool)
            self.best_params_ = best
        else:
            arch_params = {**(self.base_params or _DEFAULT_ARCH_PARAMS), 'random_seed': self.random_seed}
            self._model = self._fit_model(tr_pool, va_pool, arch_params, self.loss_params)
            self.best_params_ = {**self.loss_params, **arch_params}

        self.train_pred_ = self._model.predict_proba(tr_pool)[:, 1]
        self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]
        pr_auc = float(average_precision_score(y_va, self.valid_pred_))
        logger.info(
            '[%s] params=%s  val PR-AUC=%.4f',
            self._loss_spec.name,
            {k: self.best_params_.get(k) for k in self._loss_spec.param_bounds},
            pr_auc,
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict_proba(pool)[:, 1]
