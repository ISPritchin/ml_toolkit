"""KnowledgeDistillationPreset: дистилляция большого ансамбля в одну маленькую модель.

Обучает тяжёлого «учителя» (любой BasePreset — EasyEnsembleClassifier,
BoostedEnsemble, HeterogeneousStacking, ...), затем маленький CatBoost-
«студент» учится не на исходных 0/1-метках, а на смягчённых по температуре
вероятностях учителя (soft labels) через CatBoost `loss_function='CrossEntropy'`
— единственный нативный loss CatBoost, принимающий непрерывную метку в [0, 1]
вместо жёсткого класса.

Зачем смягчать (temperature): вероятности учителя близко к 0/1 несут почти
тот же сигнал, что и жёсткие метки — greedy-обучение студента на них теряет
информацию о ТОЧНОЙ уверенности учителя между похожими объектами. Деление
логита учителя на temperature > 1 растягивает вероятности к 0.5, обнажая
относительный порядок уверенности внутри «лёгких» и «трудных» случаев —
именно это градиент CrossEntropy студента видит и пытается воспроизвести.

CatBoost eval_metric='PRAUC' не поддерживает непрерывную train-метку
(падает с "No element of a positive class" — проверено эмпирически на
catboost 1.2.10); поэтому внутренний eval_metric студента — 'AUC' (работает
с CrossEntropy штатно), а РЕАЛЬНЫЙ отчётный/optuna-objective PR-AUC считается
отдельно через sklearn.average_precision_score на честных (не смягчённых)
y_valid — то, что реально репортится, не завязано на это ограничение CatBoost.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml_toolkit.models._base import XInput, YInput
from ml_toolkit.presets.classification._base import BasePreset
from ml_toolkit.presets.classification._optuna_utils import (
    CatBoostPruningCallback,
    catboost_arch_space,
    make_pruner,
)

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

_DEFAULT_STUDENT_PARAMS: dict[str, Any] = {
    'iterations': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8,
    'early_stopping_rounds': 50,
    'verbose': 0,
}


def _soften(p: np.ndarray, temperature: float) -> np.ndarray:
    """Температурное смягчение вероятности через логит: p -> sigmoid(logit(p) / T).

    T=1.0 — без изменений. T>1 — вероятности стягиваются к 0.5 (мягче), T<1 —
    расходятся к краям (резче). Клип перед логитом — иначе p=0/1 у уверенного
    учителя даёт ±inf.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / temperature))


class KnowledgeDistillationPreset(BasePreset):
    """Дистилляция обученного учителя (любой BasePreset) в маленький CatBoost.

    Parameters
    ----------
    teacher_preset:
        Любой экземпляр BasePreset (EasyEnsembleClassifier, BoostedEnsemble,
        HeterogeneousStacking и т.д.). Должен быть необученным — fit() будет
        вызван внутри KnowledgeDistillationPreset.fit().
    student_params:
        Параметры CatBoost-студента (без loss_function/eval_metric — задаются
        автоматически: CrossEntropy/AUC). None → дефолтные (неглубокая,
        быстрая модель). Игнорируется, если n_optuna_trials > 0.
    temperature:
        Температура смягчения вероятностей учителя перед обучением студента
        (рекомендуется 1.0–4.0; 1.0 — без смягчения).
    n_optuna_trials:
        Если > 0, архитектура студента подбирается через Optuna. Objective —
        честный PR-AUC студента на настоящих (не смягчённых) y_valid, а не
        внутренний AUC CatBoost, которым тренировка лишь мониторит сходимость.
    param_space:
        Кастомная функция `f(trial) -> dict` — search space для Optuna вместо
        дефолтного. loss_function/eval_metric в param_space не участвуют —
        фиксированы (CrossEntropy/AUC). Действует только при n_optuna_trials > 0.
    optuna_timeout:
        Ограничение по времени (сек) на весь Optuna-поиск. None — без ограничения.
    optuna_verbose:
        Если True — не глушит логи Optuna. Если False (по умолчанию) —
        форсирует WARNING на время поиска.
    optuna_pruner:
        None/строковый алиас ('median'/'hyperband'/'percentile'/
        'successive_halving'/'none')/готовый optuna.pruners.BasePruner —
        см. ml_toolkit.models model_settings.md. 'none' (по умолчанию) —
        прунинг выключен.
    random_seed:
        Зерно CatBoost-студента и Optuna sampler'а.

    Атрибуты после fit::

        teacher_          — обученный teacher_preset
        teacher_score_    — val PR-AUC учителя (честные метки)
        student_score_    — val PR-AUC студента (честные метки)

    Пример::

        from ml_toolkit.presets.classification.high_pr_auc import (
            EasyEnsembleClassifier, KnowledgeDistillationPreset,
        )

        model = KnowledgeDistillationPreset(
            teacher_preset=EasyEnsembleClassifier(n_estimators=15, neg_ratio=10),
            temperature=2.0,
        )
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...])
        print(f'teacher={model.teacher_score_:.4f}  student={model.student_score_:.4f}')
        proba = model.predict_proba(X_test)  # один быстрый CatBoost, не весь ансамбль

    """

    def __init__(
        self,
        teacher_preset: BasePreset,
        student_params: dict[str, Any] | None = None,
        temperature: float = 2.0,
        n_optuna_trials: int = 0,
        param_space: Callable[[optuna.Trial], dict[str, Any]] | None = None,
        optuna_timeout: int | None = None,
        optuna_verbose: bool = False,
        optuna_pruner: str | object | None = 'none',
        random_seed: int = 42,
    ) -> None:
        super().__init__(params=student_params, n_optuna_trials=n_optuna_trials)
        if temperature <= 0:
            raise ValueError(f'temperature должен быть > 0, получено {temperature}')
        self.teacher_preset = teacher_preset
        self.student_params = student_params
        self.temperature = temperature
        self.param_space = param_space
        self.optuna_timeout = optuna_timeout
        self.optuna_verbose = optuna_verbose
        self.optuna_pruner = optuna_pruner
        self.random_seed = random_seed

        self.teacher_: BasePreset | None = None
        self.teacher_score_: float = 0.0
        self.student_score_: float = 0.0

    # ── Optuna ────────────────────────────────────────────────────────────────

    def _tune(
        self, X_tr: pd.DataFrame, soft_y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray,
    ) -> dict[str, Any]:
        from catboost import CatBoostClassifier, Pool
        import optuna

        _optuna_prev_verbosity = optuna.logging.get_verbosity()
        if not self.optuna_verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        tr_pool = Pool(X_tr, soft_y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va, y_va, cat_features=self.cat_features_)

        def objective(trial: optuna.Trial) -> float:
            tunable = self.param_space(trial) if self.param_space is not None else catboost_arch_space(trial)
            params = {
                'loss_function': 'CrossEntropy',
                'eval_metric': 'AUC',
                'early_stopping_rounds': 80,
                'random_seed': self.random_seed,
                'verbose': 0,
                **tunable,
            }
            trial.set_user_attr('cb_params', params)
            m = CatBoostClassifier(**params)
            if params.get('task_type') == 'GPU':
                # CatBoost GPU не поддерживает user-defined callbacks — прунинг
                # для GPU-trial'ов недоступен, trial всегда доучивается до конца.
                m.fit(tr_pool, eval_set=va_pool, verbose=False)
            else:
                pruning_cb = CatBoostPruningCallback(trial, params['eval_metric'])
                m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_cb])
                pruning_cb.check_pruned()
            # honest PR-AUC на настоящих метках — не внутренний AUC CatBoost,
            # которым тренировка лишь мониторит сходимость (см. докстринг модуля).
            p = m.predict_proba(va_pool)[:, 1]
            return float(average_precision_score(y_va, p))

        logger.info('[KnowledgeDistillation] Optuna: %d trials', self.n_optuna_trials)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.random_seed),
                                    pruner=make_pruner(self.optuna_pruner))
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=self.optuna_timeout,
                       show_progress_bar=False)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return dict(study.best_trial.user_attrs['cb_params'])

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput,
        y_valid: YInput,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> KnowledgeDistillationPreset:
        from catboost import CatBoostClassifier, Pool

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(
            X_train, y_train, X_valid, y_valid
        )
        feats = self._resolve_features(X_train, selected_features)
        self.selected_features_ = feats
        self.cat_features_ = cat_features or []

        y_va = y_valid.values

        logger.info('[KnowledgeDistillation] Обучаем teacher: %s', type(self.teacher_preset).__name__)
        self.teacher_preset.fit(
            X_train, y_train, X_valid, y_valid,
            selected_features=selected_features,
            cat_features=cat_features,
        )
        self.teacher_ = self.teacher_preset

        teacher_raw_tr = (
            self.teacher_.train_pred_ if self.teacher_.train_pred_ is not None
            else self.teacher_.predict_proba(X_train)
        )
        teacher_raw_va = (
            self.teacher_.valid_pred_ if self.teacher_.valid_pred_ is not None
            else self.teacher_.predict_proba(X_valid)
        )
        self.teacher_score_ = float(average_precision_score(y_va, teacher_raw_va))

        soft_y_tr = _soften(teacher_raw_tr, self.temperature)

        X_tr_feats = X_train[feats]
        X_va_feats = X_valid[feats]

        if self.n_optuna_trials > 0:
            student_params = self._tune(X_tr_feats, soft_y_tr, X_va_feats, y_va)
        else:
            student_params = dict(self.student_params or _DEFAULT_STUDENT_PARAMS)

        params = {
            **student_params,
            'loss_function': 'CrossEntropy',
            'eval_metric': 'AUC',
            'random_seed': self.random_seed,
        }
        self._model = CatBoostClassifier(**params)
        tr_pool = Pool(X_tr_feats, soft_y_tr, cat_features=self.cat_features_)
        va_pool = Pool(X_va_feats, y_va, cat_features=self.cat_features_)
        self._model.fit(tr_pool, eval_set=va_pool, verbose=False)

        self.train_pred_ = self._model.predict_proba(Pool(X_tr_feats, cat_features=self.cat_features_))[:, 1]
        self.valid_pred_ = self._model.predict_proba(va_pool)[:, 1]
        self.student_score_ = float(average_precision_score(y_va, self.valid_pred_))

        logger.info(
            '[KnowledgeDistillation] teacher PR-AUC=%.4f  student PR-AUC=%.4f  Δ=%.4f  temperature=%.2f',
            self.teacher_score_, self.student_score_, self.student_score_ - self.teacher_score_, self.temperature,
        )

        self.best_params_ = {'student_params': params, 'temperature': self.temperature}
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool
        X_feats = X[self.selected_features_]
        return self._model.predict_proba(Pool(X_feats, cat_features=self.cat_features_))[:, 1]
