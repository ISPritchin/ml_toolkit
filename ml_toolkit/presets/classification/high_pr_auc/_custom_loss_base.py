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

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

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
    ) -> None:
        super().__init__(params=None, n_optuna_trials=n_optuna_trials)
        self.loss_params = dict(loss_params)
        self.base_params = base_params
        self.optuna_timeout = optuna_timeout
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.selected_features = selected_features or []

    def _make_loss(self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict) -> Any:
        """Строит объект лосса. tr_pool/arch_params — для лоссов, которым нужна

        статистика датасета (n_pos/n_neg) или число итераций модели (LDAMLoss);
        большинству лоссов (Focal/Tversky/Poly/Asymmetric) они не нужны, и
        параметр можно игнорировать — дефолтная реализация так и делает.
        """
        return self._loss_spec.loss_cls(**loss_params)

    def _fit_model(
        self,
        tr_pool: Any,
        va_pool: Any,
        arch_params: dict,
        loss_params: dict[str, float],
        callbacks: list | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function=self._make_loss(loss_params, tr_pool=tr_pool, arch_params=arch_params),
            eval_metric='AUC',
            **arch_params,
        )
        model.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=callbacks)
        return model

    def _tune(self, tr_pool: Any, va_pool: Any) -> tuple[Any, dict]:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        esr = _DEFAULT_ARCH_PARAMS['early_stopping_rounds']
        loss_keys = list(self._loss_spec.param_bounds)

        def objective(trial: optuna.Trial) -> float:
            loss_p = {
                k: trial.suggest_float(k, *self._loss_spec.param_bounds[k])
                for k in loss_keys
            }
            arch_p = {
                'iterations':        trial.suggest_int('iterations', 300, 1000, step=100),
                'max_depth':         trial.suggest_int('max_depth', 3, 7),
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 1, 30),
                'early_stopping_rounds': esr,
                'random_seed': self.random_seed,
                'verbose': 0,
            }
            pruning_cb = CatBoostPruningCallback(trial, 'AUC')
            m = self._fit_model(tr_pool, va_pool, arch_p, loss_p, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(va_pool.get_label(), p))

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(),
        )
        # Первый триал — значения из __init__ (loss_params + дефолтная архитектура),
        # чтобы они не терялись молча среди случайных стартовых точек Optuna.
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
        best = study.best_params
        best_loss = {k: best[k] for k in loss_keys}
        best_arch = {k: v for k, v in best.items() if k not in loss_keys}
        best_arch['early_stopping_rounds'] = esr
        best_arch['random_seed'] = self.random_seed
        best_arch['verbose'] = 0
        model = self._fit_model(tr_pool, va_pool, best_arch, best_loss)
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
    ) -> '_CustomLossClassifierBase':
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
