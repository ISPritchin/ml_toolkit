"""Интерпретируемые rule-based и tree-based модели из пакета imodels.

Поддерживаемые имена (model_settings['name']):
    'figs'        — Fast Interpretable Greedy-Tree Sums: сумма небольших деревьев, очень читаемо.
    'skope_rules' — SkopeRules: отбор правил по precision/recall (только классификация).
    'brl'         — Bayesian Rule List: байесовские цепочки if-then с uncertainty.
    'ripper'      — RIPPER: жадная индукция правил (Repeated Incremental Pruning to Produce Error Reduction).

Регрессия для skope_rules/brl/ripper: FIGSRegressor (те модели classification-first).
Пакет: imodels (pip install imodels)
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_sample_weight

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._tabular._interpretable._common import make_impute_scale_pipeline, numeric_features
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    make_study,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

if TYPE_CHECKING:
    from imodels import FIGSClassifier, FIGSRegressor

logger = logging.getLogger(__name__)

_CLS_FIRST_NAMES = frozenset({'skope_rules', 'brl', 'ripper'})


def _safe_proba(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Извлекает вероятности класса 1, обрабатывая разные API rule-based моделей."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return np.clip(proba[:, 1], 0.0, 1.0)
        return np.clip(proba, 0.0, 1.0)
    return np.clip(model.predict(X).astype(float), 0.0, 1.0)


def _make_figs_reg(params: dict) -> FIGSRegressor:
    from imodels import FIGSRegressor
    return FIGSRegressor(**params)


def _make_figs_cls(params: dict) -> FIGSClassifier:
    from imodels import FIGSClassifier
    return FIGSClassifier(**params)


# ── Классы (новый API) ────────────────────────────────────────────────────────

class IModelsRegressor(BaseModel):
    """Регрессия через FIGS (FIGSRegressor) с подбором max_rules и max_trees через Optuna.

    Для 'skope_rules', 'brl', 'ripper' также использует FIGSRegressor (те модели classification-first).
    Категориальные признаки исключаются. params=None → Optuna; params=dict → прямое обучение.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> IModelsRegressor:
        try:
            from imodels import FIGSRegressor as _check  # noqa: F401
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)
        name = ms.get('name', 'figs')

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[%s Reg] features=%d', name.upper(), len(self._num_feats_))

        self._prep = make_impute_scale_pipeline()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = _make_figs_reg(self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_rules': trial.suggest_int('max_rules', 5, 30),
                    'max_trees': trial.suggest_int('max_trees', 5, 30),
                }
                m = _make_figs_reg(params)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict(X_va))

            study = make_study(direction, self.model_settings)
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[%s Reg] Best score=%.4f params=%s', name.upper(), study.best_value, self.best_params_)

            self._model = _make_figs_reg(self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict(X_va)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(
            self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        ))


class IModelsClassifier(BaseModel):
    """Классификация через rule-based модели из imodels с подбором параметров через Optuna.

    Dispatch по model_settings['name']: 'figs' | 'skope_rules' | 'brl' | 'ripper'.
    Категориальные признаки исключаются. Вероятности калибруются изотонической регрессией.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> IModelsClassifier:
        try:
            from imodels import FIGSClassifier as _check  # noqa: F401
        except ImportError as exc:
            raise ImportError('Установи пакет: pip install imodels') from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)
        name = ms.get('name', 'figs')

        self._num_feats_ = numeric_features(self.selected_features_, self.cat_features_)
        logger.info('[%s Cls] features=%d', name.upper(), len(self._num_feats_))

        self._prep = make_impute_scale_pipeline()
        X_tr = self._prep.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)
        sw_tr = compute_sample_weight('balanced', y_tr)

        if X_valid is not None:
            X_va = self._prep.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=int)
        else:
            X_va = y_va = None

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        self._brl_discretizer_ = None  # см. _fit_brl/_to_model_space — только для name='brl'
        if name == 'figs':
            fitted_model, bp = self._fit_figs(X_tr, y_tr, X_va, y_va, metric_fn, direction, sw_tr)
        elif name == 'skope_rules':
            fitted_model, bp = self._fit_skope(X_tr, y_tr, X_va, y_va, metric_fn, direction, sw_tr)
        elif name == 'brl':
            fitted_model, bp = self._fit_brl(X_tr, y_tr, X_va, y_va, metric_fn, direction)
        elif name == 'ripper':
            fitted_model, bp = self._fit_ripper(X_tr, y_tr, X_va, y_va, metric_fn, direction)
        else:
            raise ValueError(f'Unknown imodels classifier: {name!r}. Valid: figs, skope_rules, brl, ripper')

        self._model = fitted_model
        self.best_params_ = bp

        self.train_pred_ = _safe_proba(self._model, self._to_model_space(X_tr))
        if X_valid is not None:
            self.valid_pred_ = _safe_proba(self._model, self._to_model_space(X_va))
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _to_model_space(self, X: np.ndarray) -> np.ndarray:
        """BRL требует one-hot дискретизированные бины, а не непрерывные self._prep-признаки.

        На них работают figs/skope_rules/ripper как есть.
        """
        if self._brl_discretizer_ is not None:
            return self._brl_discretizer_.transform(X)
        return X

    def _fit_figs(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray | None,
        y_va: np.ndarray | None,
        metric_fn: Callable,
        direction: str,
        sw_tr: np.ndarray,
    ):
        if self.params is not None:
            m = _make_figs_cls(self.params)
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
            return m, self.params
        if X_va is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')

        def objective(trial: optuna.Trial) -> float:
            m = _make_figs_cls({'max_rules': trial.suggest_int('max_rules', 5, 30),
                                 'max_trees': trial.suggest_int('max_trees', 5, 30)})
            m.fit(X_tr, y_tr, sample_weight=sw_tr)
            return metric_fn(y_va, _safe_proba(m, X_va))

        study = make_study(direction, self.model_settings)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(self.model_settings), show_progress_bar=False)
        bp = study.best_params
        m = _make_figs_cls(bp)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        return m, bp

    def _fit_skope(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray | None,
        y_va: np.ndarray | None,
        metric_fn: Callable,
        direction: str,
        sw_tr: np.ndarray,
    ):
        from imodels import SkopeRulesClassifier
        if self.params is not None:
            m = SkopeRulesClassifier(**self.params)
            m.fit(X_tr, y_tr, feature_names=self._num_feats_, sample_weight=sw_tr)
            return m, self.params
        if X_va is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')

        def objective(trial: optuna.Trial) -> float:
            p = {'n_estimators': trial.suggest_int('n_estimators', 5, 50),
                 'max_depth': trial.suggest_int('max_depth', 2, 5), 'random_state': 42}
            m = SkopeRulesClassifier(**p)
            m.fit(X_tr, y_tr, feature_names=self._num_feats_, sample_weight=sw_tr)
            return metric_fn(y_va, _safe_proba(m, X_va))

        study = make_study(direction, self.model_settings)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(self.model_settings), show_progress_bar=False)
        bp = {**study.best_params, 'random_state': 42}
        m = SkopeRulesClassifier(**bp)
        m.fit(X_tr, y_tr, feature_names=self._num_feats_, sample_weight=sw_tr)
        return m, bp

    def _fit_brl(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray | None,
        y_va: np.ndarray | None,
        metric_fn: Callable,
        direction: str,
    ):
        from imodels import BayesianRuleListClassifier
        from sklearn.preprocessing import KBinsDiscretizer

        # BRL требует строго бинарные (one-hot дискретизированные) признаки — fit() падает
        # с ValueError('All numeric features must be discretized prior to fitting!') на сыром
        # self._prep-выходе. imodels.BasicDiscretizer(encode='onehot') несовместим с текущим
        # sklearn (ломается внутри самого imodels на несовпадении shape), поэтому
        # дискретизируем через sklearn.KBinsDiscretizer напрямую.
        self._brl_discretizer_ = KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile')
        X_tr_d = self._brl_discretizer_.fit_transform(X_tr)
        feat_names = list(self._brl_discretizer_.get_feature_names_out(self._num_feats_))
        X_va_d = self._brl_discretizer_.transform(X_va) if X_va is not None else None

        if self.params is not None:
            m = BayesianRuleListClassifier(**self.params)
            m.fit(X_tr_d, y_tr, feature_names=feat_names)
            return m, self.params
        if X_va_d is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')

        def objective(trial: optuna.Trial) -> float:
            p = {'listlengthprior': trial.suggest_int('listlengthprior', 3, 10),
                 'listwidthprior': trial.suggest_int('listwidthprior', 1, 4)}
            m = BayesianRuleListClassifier(**p)
            m.fit(X_tr_d, y_tr, feature_names=feat_names)
            return metric_fn(y_va, _safe_proba(m, X_va_d))

        study = make_study(direction, self.model_settings)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(self.model_settings), show_progress_bar=False)
        bp = study.best_params
        m = BayesianRuleListClassifier(**bp)
        m.fit(X_tr_d, y_tr, feature_names=feat_names)
        return m, bp

    def _fit_ripper(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray | None,
        y_va: np.ndarray | None,
        metric_fn: Callable,
        direction: str,
    ):
        try:
            from imodels import RIPPERClassifier
        except ImportError as exc:
            # imodels>=1.4 (текущий PyPI-релиз, 2.0.4 на момент написания) больше не
            # экспортирует RIPPERClassifier вовсе — 'ripper' был написан против более
            # старой/иной версии пакета. pip install imodels (как советует докстринг
            # модуля) НЕ даёт рабочий RIPPER — это не транзиентная ошибка окружения.
            raise ImportError(
                "RIPPERClassifier отсутствует в установленной версии imodels "
                "(проверено на imodels==2.0.4 — класс удалён/переименован в пакете). "
                "model_settings['name']='ripper' в этом состоянии пакета нерабочий. "
                "Используйте 'figs'/'skope_rules'/'brl', либо пакет wittgenstein "
                "(pip install wittgenstein) с отдельной реализацией RIPPER."
            ) from exc
        if self.params is not None:
            m = RIPPERClassifier(**self.params)
            m.fit(X_tr, y_tr, feature_names=self._num_feats_)
            return m, self.params
        if X_va is None:
            raise ValueError('X_valid обязателен при params=None (режим Optuna)')

        def objective(trial: optuna.Trial) -> float:
            p = {'k': trial.suggest_int('k', 1, 5)}
            m = RIPPERClassifier(**p)
            m.fit(X_tr, y_tr, feature_names=self._num_feats_)
            return metric_fn(y_va, _safe_proba(m, X_va))

        study = make_study(direction, self.model_settings)
        study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(self.model_settings), show_progress_bar=False)
        bp = study.best_params
        m = RIPPERClassifier(**bp)
        m.fit(X_tr, y_tr, feature_names=self._num_feats_)
        return m, bp

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self._prep.transform(X[self._num_feats_].to_numpy(dtype=float))
        raw = _safe_proba(self._model, self._to_model_space(X_prep))
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

