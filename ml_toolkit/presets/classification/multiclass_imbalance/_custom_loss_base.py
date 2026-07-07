"""Общий движок для пресетов вида «CatBoost + один кастомный Python-лосс мультикласса».

Мультиклассовые CatBoost custom-objective используют другой протокол, чем
бинарные (см. ml_toolkit/presets/classification/high_pr_auc/_custom_loss_base.py):
`calc_ders_multi(self, approx, target, weight)` вызывается CatBoost по одному
разу НА ОБЪЕКТ (не на весь батч сразу, как calc_ders_range), и должен
возвращать (der1: list[float], der2: list[list[float]]) — der2 обязан быть
полной n_classes x n_classes матрицей (даже если внедиагональные элементы
нулевые: CatBoost падает при плоском der2). Так как лосс использует `self.*`,
CatBoost не может JIT-скомпилировать calc_ders_multi (numba выдаёт "Can't
optimize... because self argument is used") — это ожидаемо и не ошибка,
объектив исполняется в интерпретируемом Python построчно.

eval_metric='TotalF1:average=Macro' (не 'Accuracy'/'AUC', как в binary-модуле):
все три текущих мультиклассовых лосса (Equalization/BalancedSoftmax/LogitNorm)
нацелены на длиннохвостый дисбаланс, macro-F1 отражает качество на редких
классах, а accuracy — нет (доминируется головным классом). Optuna использует
ту же метрику как objective.

y_train/y_valid кодируются через LabelEncoder перед обучением (произвольные
исходные метки → плотные 0..n_classes-1 индексы, которые ожидает
calc_ders_multi'т target), и декодируются обратно в predict().
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import CatBoostPruningCallback, make_pruner

logger = logging.getLogger(__name__)

_EVAL_METRIC = 'TotalF1:average=Macro'

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
class _MulticlassLossSpec:
    """Описание одного мультиклассового лосса для _CustomLossClassifierMulticlassBase.

    loss_cls:
        Класс лосса из ml_toolkit.losses (calc_ders_multi-совместимый).
    param_bounds:
        {имя_параметра_лосса: (low, high)} — границы Optuna suggest_float.
    name:
        Короткое имя для логов.
    """

    loss_cls: type
    param_bounds: dict[str, tuple[float, float]]
    name: str


class _CustomLossClassifierMulticlassBase(BasePreset):
    """Общая fit/tune/predict логика для CatBoost с одним кастомным мультиклассовым лоссом.

    Подкласс обязан задать класс-атрибут `_loss_spec: _MulticlassLossSpec` и
    передать в `super().__init__` уже собранный `loss_params: dict[str, float]`
    со своими именованными параметрами лосса.
    """

    _loss_spec: _MulticlassLossSpec

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
        self._label_encoder: LabelEncoder | None = None

    def _make_loss(
        self, loss_params: dict[str, float], *, tr_pool: Any, arch_params: dict, n_classes: int
    ) -> Any:
        """Строит объект лосса. Большинству лоссов нужна class_counts из train —

        подклассы переопределяют этот метод (по образцу LDAMClassifier/
        InfluenceBalancedLossClassifier в binary-модуле).
        """
        return self._loss_spec.loss_cls(**loss_params)

    def _fit_model(
        self,
        tr_pool: Any,
        va_pool: Any,
        arch_params: dict,
        loss_params: dict[str, float],
        n_classes: int,
        callbacks: list | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function=self._make_loss(loss_params, tr_pool=tr_pool, arch_params=arch_params, n_classes=n_classes),
            eval_metric=_EVAL_METRIC,
            classes_count=n_classes,
            **arch_params,
        )
        model.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=callbacks)
        return model

    def _tune(self, tr_pool: Any, va_pool: Any, n_classes: int) -> tuple[Any, dict]:
        import optuna

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
            # study.best_params содержит только то, что реально прошло через
            # trial.suggest_*; параметры, зафиксированные custom как голое
            # значение (не suggest), туда не попадут — поэтому сохраняем
            # собранные loss_p/arch_p целиком через user_attr и забираем их
            # из best_trial после оптимизации, а не реконструируем из best_params.
            trial.set_user_attr('loss_p', loss_p)
            trial.set_user_attr('arch_p', arch_p)
            pruning_cb = CatBoostPruningCallback(trial, _EVAL_METRIC)
            m = self._fit_model(tr_pool, va_pool, arch_p, loss_p, n_classes, callbacks=[pruning_cb])
            pruning_cb.check_pruned()
            pred = m.predict_proba(va_pool).argmax(axis=1)
            return float(f1_score(va_pool.get_label(), pred, average='macro'))

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=make_pruner(),
        )
        # Безопасно только для дефолтного search space (см. binary _custom_loss_base.py
        # для подробного объяснения) — при кастомном param_space пропускаем enqueue
        # целиком, иначе дефолтные значения, оказавшиеся вне новых границ, могут
        # случайно "выиграть" trial и молча испортить результат тюнинга.
        if self.param_space is None:
            study.enqueue_trial({
                **self.loss_params,
                'iterations':       _DEFAULT_ARCH_PARAMS['iterations'],
                'max_depth':        _DEFAULT_ARCH_PARAMS['max_depth'],
                'learning_rate':    _DEFAULT_ARCH_PARAMS['learning_rate'],
                'l2_leaf_reg':       _DEFAULT_ARCH_PARAMS['l2_leaf_reg'],
                'subsample':        _DEFAULT_ARCH_PARAMS['subsample'],
                'min_data_in_leaf': _DEFAULT_ARCH_PARAMS['min_data_in_leaf'],
            })
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        best_loss = dict(study.best_trial.user_attrs['loss_p'])
        best_arch = dict(study.best_trial.user_attrs['arch_p'])
        model = self._fit_model(tr_pool, va_pool, best_arch, best_loss, n_classes)
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
    ) -> '_CustomLossClassifierMulticlassBase':
        from catboost import Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features or self.selected_features or None)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or self.cat_features

        self._label_encoder = LabelEncoder()
        y_tr = self._label_encoder.fit_transform(y_train.values)
        y_va = self._label_encoder.transform(y_valid.values)
        n_classes = len(self._label_encoder.classes_)
        if n_classes < 3:
            raise ValueError(
                f"{type(self).__name__} рассчитан на мультикласс (>=3 классов), "
                f"получено {n_classes} — для бинарной классификации см. high_pr_auc/."
            )
        self.n_classes_ = n_classes

        tr_pool = Pool(X_train[feats], y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_valid[feats], y_va, cat_features=self.cat_features_)

        if self.n_optuna_trials > 0:
            self._model, best = self._tune(tr_pool, va_pool, n_classes)
            self.best_params_ = best
        else:
            arch_params = {**(self.base_params or _DEFAULT_ARCH_PARAMS), 'random_seed': self.random_seed}
            self._model = self._fit_model(tr_pool, va_pool, arch_params, self.loss_params, n_classes)
            self.best_params_ = {**self.loss_params, **arch_params}

        self.train_pred_ = self._model.predict_proba(tr_pool)
        self.valid_pred_ = self._model.predict_proba(va_pool)
        macro_f1 = f1_score(y_va, self.valid_pred_.argmax(axis=1), average='macro')
        logger.info(
            '[%s] params=%s  val macro-F1=%.4f',
            self._loss_spec.name,
            {k: self.best_params_.get(k) for k in self._loss_spec.param_bounds},
            macro_f1,
        )
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        pool = Pool(X[self.selected_features_], cat_features=self.cat_features_)
        return self._model.predict_proba(pool)

    def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
        """Мультиклассовое предсказание по argmax — в отличие от BasePreset.predict,

        порог здесь неприменим (не бинарная классификация).
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self._label_encoder.inverse_transform(idx)
