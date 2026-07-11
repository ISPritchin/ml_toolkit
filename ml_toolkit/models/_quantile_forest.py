"""Quantile Random Forest.

Предсказывает квантили распределения таргета, а не только среднее.
Для регрессии: использует медиану (q=0.5), что оптимально для MAE.

Требует: pip install quantile-forest

Модель возвращается как Pipeline([imputer, estimator]) для нативной обработки NaN.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError as e:
    raise ImportError(
        'Quantile Random Forest requires quantile-forest package: pip install quantile-forest'
    ) from e

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_cat_encoder,
    build_cat_encoder,
    fit_calibrator,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_MEDIAN_QUANTILE = 0.5


class _QuantileMedianWrapper(BaseEstimator, RegressorMixin):
    """Обёртка вокруг RandomForestQuantileRegressor для sklearn Pipeline-совместимости.

    Pipeline.predict() вызывает estimator.predict() без параметров. Эта обёртка
    подменяет predict() → predict(quantiles=0.5).

    Наследование от BaseEstimator обязательно, не только стилистически: начиная
    со sklearn 1.6+ Pipeline.__sklearn_is_fitted__()/check_is_fitted() читают
    __sklearn_tags__(), которого нет у обычного объекта — без BaseEstimator
    Pipeline.predict() падает с AttributeError на этой обёртке.
    """

    def __init__(self, **params) -> None:
        self._model = RandomForestQuantileRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Обучает внутренний RandomForestQuantileRegressor и копирует feature_importances_."""
        self._model.fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает медианное предсказание (quantile=0.5) для совместимости с Pipeline.predict()."""
        return self._model.predict(X, quantiles=_MEDIAN_QUANTILE)

    def predict_quantiles(self, X: np.ndarray, quantiles: float | list[float]) -> np.ndarray:
        """Возвращает предсказания для произвольных квантилей `quantiles`."""
        return self._model.predict(X, quantiles=quantiles)


def _make_reg_pipeline(params: dict) -> Pipeline:
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('estimator', _QuantileMedianWrapper(**params))])


def _suggest(trial: optuna.Trial) -> dict:
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1,
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class QuantileForestRegressor(BaseModel):
    """QuantileForestRegressor — медианное предсказание QRF с подбором через Optuna.

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
    ) -> QuantileForestRegressor:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        Xtr = X_train_enc[self.selected_features_]
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = _make_reg_pipeline(self.params)
            self._model.fit(Xtr, y_tr)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            Xva = X_valid_enc[self.selected_features_]
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                pipe = _make_reg_pipeline(_suggest(trial))
                pipe.fit(Xtr, y_tr)
                return metric_fn(y_va, pipe.predict(Xva))

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42, 'n_jobs': -1}
            logger.info('[QUANTILE_FOREST Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = _make_reg_pipeline(self.best_params_)
            self._model.fit(Xtr, y_tr)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            self.valid_pred_ = self._model.predict(X_valid_enc[self.selected_features_])
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        return self._model.predict(X_enc[self.selected_features_])


class QuantileForestClassifier(BaseModel):
    """QuantileForestClassifier — QRF-признаки (квантили) + LogisticRegression с подбором через Optuna.

    Хранит: _qrf (QRF), _clf (LogisticRegression), _imp (SimpleImputer).
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
    ) -> QuantileForestClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_, self.selected_features_ = \
            build_cat_encoder(X_train, self.selected_features_, self.cat_features_, ms)
        X_train_enc = apply_cat_encoder(X_train, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)

        self._imp = SimpleImputer(strategy='median')
        X_tr = self._imp.fit_transform(X_train_enc[self.selected_features_])
        y_tr = y_train.to_numpy(dtype=int)

        qrf_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5,
                      'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
        self._qrf = RandomForestQuantileRegressor(**qrf_params)
        self._qrf.fit(X_tr, y_tr.astype(float))
        self._model = self._qrf  # for _check_fitted

        def _qrf_feats(X_arr: np.ndarray) -> np.ndarray:
            q25 = self._qrf.predict(X_arr, quantiles=0.25)
            q50 = self._qrf.predict(X_arr, quantiles=0.5)
            q75 = self._qrf.predict(X_arr, quantiles=0.75)
            return np.column_stack([q25, q50, q75, q75 - q25])

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        F_tr = _qrf_feats(X_tr)

        if self.params is not None:
            # Дефолты этого адаптера побеждаются явным max_iter/class_weight в self.params,
            # а не наоборот — LogisticRegression(**self.params, max_iter=..., class_weight=...)
            # падал с TypeError('multiple values for keyword argument'), если self.params уже
            # содержал любой из этих ключей.
            direct_params = {'max_iter': 500, 'class_weight': 'balanced', **self.params}
            self._clf = LogisticRegression(**direct_params)
            self._clf.fit(F_tr, y_tr)
            self.best_params_ = direct_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = self._imp.transform(X_valid_enc[self.selected_features_])
            y_va = y_valid.to_numpy(dtype=int)
            F_va = _qrf_feats(X_va)

            def objective(trial: optuna.Trial) -> float:
                C = trial.suggest_float('C', 1e-3, 100.0, log=True)
                clf = LogisticRegression(C=C, max_iter=500, class_weight='balanced', random_state=42)
                clf.fit(F_tr, y_tr)
                return metric_fn(y_va, clf.predict_proba(F_va)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = {**study.best_params, 'random_state': 42}
            logger.info('[QUANTILE_FOREST Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._clf = LogisticRegression(**self.best_params_, max_iter=500, class_weight='balanced')
            self._clf.fit(F_tr, y_tr)

        self.train_pred_ = self._clf.predict_proba(F_tr)[:, 1]
        if X_valid is not None:
            X_valid_enc = apply_cat_encoder(X_valid, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
            X_va = self._imp.transform(X_valid_enc[self.selected_features_])
            F_va = _qrf_feats(X_va)
            self.valid_pred_ = self._clf.predict_proba(F_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = apply_cat_encoder(X, self._cat_encoder_, self._cat_in_sel_, self._cat_col_names_)
        X_imp = self._imp.transform(X_enc[self.selected_features_])
        q25 = self._qrf.predict(X_imp, quantiles=0.25)
        q50 = self._qrf.predict(X_imp, quantiles=0.5)
        q75 = self._qrf.predict(X_imp, quantiles=0.75)
        F = np.column_stack([q25, q50, q75, q75 - q25])
        raw = self._clf.predict_proba(F)[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

