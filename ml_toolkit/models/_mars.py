"""MARS — Multivariate Adaptive Regression Splines (Friedman, 1991).

Алгоритм Forward/Backward: сначала жадно добавляет hinge-функции (ломаные)
и их произведения до достижения max_terms, потом удаляет наименее значимые.
Результат — читаемая формула: сумма кусочно-линейных термов.
Для классификации: Earth как feature transformer + LogisticRegression.

Поддерживаемые имена (model_settings['name']): 'mars'

Пакет: py-earth (pip install sklearn-contrib-py-earth)
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


# ── Классы (новый API) ────────────────────────────────────────────────────────

class MARSRegressor(BaseModel):
    """MARS (pyearth.Earth) для регрессии с подбором max_degree и max_terms через Optuna.

    Категориальные признаки исключаются. Хранит _imputer и _num_feats_.
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
    ) -> MARSRegressor:
        try:
            from pyearth import Earth
        except ImportError as exc:
            raise ImportError(
                'Установи пакет: pip install sklearn-contrib-py-earth. '
                'На Python 3.11+ может потребоваться: pip install git+https://github.com/scikit-learn-contrib/py-earth'
            ) from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[MARS Reg] features=%d', len(self._num_feats_))

        self._imputer = SimpleImputer(strategy='median')
        X_tr = self._imputer.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = Earth(**self.params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._imputer.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    'max_degree': trial.suggest_int('max_degree', 1, 3),
                    'max_terms': trial.suggest_int('max_terms', 10, 100),
                }
                m = Earth(**params)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict(X_va))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[MARS Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = Earth(**self.best_params_)
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict(X_tr)
        if X_valid is not None:
            X_va = self._imputer.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._model.predict(X_va)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(
            self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float))
        ))


class MARSClassifier(BaseModel):
    """MARS (Earth) как feature transformer + LogisticRegression. Подбор параметров через Optuna.

    Категориальные признаки исключаются. Хранит _model (Earth), _clf (LogisticRegression), _imputer.
    Вероятности калибруются изотонической регрессией. params=None → Optuna.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> MARSClassifier:
        try:
            from pyearth import Earth
        except ImportError as exc:
            raise ImportError(
                'Установи пакет: pip install sklearn-contrib-py-earth. '
                'На Python 3.11+ может потребоваться: pip install git+https://github.com/scikit-learn-contrib/py-earth'
            ) from exc

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[MARS Cls] features=%d', len(self._num_feats_))

        self._imputer = SimpleImputer(strategy='median')
        X_tr = self._imputer.fit_transform(X_train[self._num_feats_].to_numpy(dtype=float))
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            # Earth раскрывает только max_degree/max_terms как тюнимые (тот же выбор, что и в
            # search space ниже) — всё остальное в params относится к LogisticRegression.
            # Раньше clf_p брал только 'C' — любой другой валидный kwarg LogisticRegression
            # (fit_intercept, penalty, свой class_weight) молча отбрасывался.
            earth_p = {k: v for k, v in self.params.items() if k in ('max_degree', 'max_terms')}
            clf_p = {k: v for k, v in self.params.items() if k not in ('max_degree', 'max_terms')}
            self._model = Earth(**earth_p)
            self._model.fit(X_tr, y_tr)
            X_tr_t = self._model.transform(X_tr)
            direct_clf_params = {
                'solver': 'lbfgs', 'max_iter': 500, 'class_weight': 'balanced', 'random_state': 42, **clf_p,
            }
            self._clf = LogisticRegression(**direct_clf_params)
            self._clf.fit(X_tr_t, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_va = self._imputer.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                max_degree = trial.suggest_int('max_degree', 1, 3)
                max_terms = trial.suggest_int('max_terms', 10, 100)
                C = trial.suggest_float('C', 1e-3, 1e2, log=True)
                earth = Earth(max_degree=max_degree, max_terms=max_terms)
                earth.fit(X_tr, y_tr)
                clf = LogisticRegression(C=C, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=42)
                clf.fit(earth.transform(X_tr), y_tr)
                return metric_fn(y_va, clf.predict_proba(earth.transform(X_va))[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[MARS Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = Earth(max_degree=self.best_params_['max_degree'], max_terms=self.best_params_['max_terms'])
            self._model.fit(X_tr, y_tr)
            X_tr_t = self._model.transform(X_tr)
            X_va_t = self._model.transform(X_va)
            self._clf = LogisticRegression(
                C=self.best_params_['C'], solver='lbfgs', max_iter=500,
                class_weight='balanced', random_state=42,
            )
            self._clf.fit(X_tr_t, y_tr)

        self.train_pred_ = self._clf.predict_proba(self._model.transform(X_tr))[:, 1]
        if X_valid is not None:
            X_va = self._imputer.transform(X_valid[self._num_feats_].to_numpy(dtype=float))
            self.valid_pred_ = self._clf.predict_proba(self._model.transform(X_va))[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float))
        raw = self._clf.predict_proba(self._model.transform(X_imp))[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

