# ruff: noqa: N806
from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._undersampling import UndersampleSampler
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    make_xgb_pruning_callback,
    prep_cat_features,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_prep = prep_cat_features


def _default_xgb_param_space(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска XGBoost по умолчанию (переопределяется model_settings['param_space'])."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }


# ── Классы (новый API) ────────────────────────────────────────────────────────

class XGBoostRegressor(BaseModel):
    """XGBRegressor с автоматическим подбором гиперпараметров через Optuna (MAE-loss).

    Categorical features: нативная поддержка через dtype='category' (enable_categorical=True).
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
    ) -> XGBoostRegressor:
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError('XGBoost not installed. Run: pip install xgboost') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)
        has_cat = bool(self.cat_features_)

        # XGBoost uses category dtype — no OrdinalEncoder stored
        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            self._model = xgb.XGBRegressor(**self.params)
            eval_set = []
            if X_valid is not None:
                Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
                eval_set = [(Xva, y_valid.to_numpy(dtype=float))]
            self._model.fit(Xtr, y_tr, eval_set=eval_set or None, verbose=False)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            y_va = y_valid.to_numpy(dtype=float)
            param_space: Callable[[optuna.Trial], dict] | None = ms.get('param_space')

            def objective(trial: optuna.Trial) -> float:
                tunable = param_space(trial) if param_space is not None else _default_xgb_param_space(trial)
                params = {
                    **tunable,
                    'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                    'random_state': 42, 'enable_categorical': has_cat, 'early_stopping_rounds': 100,
                }
                trial.set_user_attr('xgb_params', params)
                m = xgb.XGBRegressor(**params)
                m.fit(
                    Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False,
                    callbacks=[make_xgb_pruning_callback(trial)],
                )
                return metric_fn(y_va, m.predict(Xva))

            logger.info(
                '[XGBoost Reg] Optuna: %d trials, custom_param_space=%s',
                self.n_optuna_trials, param_space is not None,
            )
            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
            )
            study.optimize(
                objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False,
            )
            self.best_params_ = dict(study.best_trial.user_attrs['xgb_params'])
            logger.info('[XGBoost Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = xgb.XGBRegressor(**self.best_params_)
            self._model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)

        self.train_pred_ = self._model.predict(Xtr)
        if X_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            self.valid_pred_ = self._model.predict(Xva)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(_prep(X, self.selected_features_, self.cat_features_))


class XGBoostClassifier(BaseModel):
    """XGBClassifier с автоматическим подбором гиперпараметров через Optuna (PR-AUC).

    Вероятности инференса калибруются изотонической регрессией по валидационной выборке.
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
    ) -> XGBoostClassifier:
        try:
            import xgboost as xgb
        except ImportError as err:
            raise ImportError('XGBoost not installed. Run: pip install xgboost') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)
        has_cat = bool(self.cat_features_)

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            self._model = xgb.XGBClassifier(**self.params)
            eval_set = []
            if X_valid is not None:
                Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
                eval_set = [(Xva, y_valid.to_numpy(dtype=int))]
            self._model.fit(Xtr, y_tr, eval_set=eval_set or None, verbose=False)
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            y_va = y_valid.to_numpy(dtype=int)
            param_space: Callable[[optuna.Trial], dict] | None = ms.get('param_space')
            undersample_majority: bool = ms.get('undersample_majority', True)

            full_idx = np.arange(len(y_tr))
            # XGBoostClassifier поддерживает только бинарную классификацию.
            sampler = UndersampleSampler(y_tr, is_binary=True, log_prefix='[XGBoost Cls]') if undersample_majority else None
            if not undersample_majority:
                logger.info('[XGBoost Cls] undersample_majority=False — обучение на полных данных (n=%d)', len(y_tr))

            def objective(trial: optuna.Trial) -> float:
                if sampler is not None:
                    fraction_value = sampler.suggest_fraction(trial)
                    idx = sampler.sample_idx(fraction_value, trial.number)
                else:
                    idx = full_idx
                Xtr_trial, ytr_trial = Xtr.iloc[idx], y_tr[idx]

                tunable = param_space(trial) if param_space is not None else _default_xgb_param_space(trial)
                params = {
                    **tunable,
                    'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                    'random_state': 42, 'enable_categorical': has_cat, 'early_stopping_rounds': 100,
                }
                trial.set_user_attr('xgb_params', params)
                m = xgb.XGBClassifier(**params)
                m.fit(
                    Xtr_trial, ytr_trial, eval_set=[(Xva, y_va)], verbose=False,
                    callbacks=[make_xgb_pruning_callback(trial)],
                )
                return metric_fn(y_va, m.predict_proba(Xva)[:, 1])

            logger.info(
                '[XGBoost Cls] Optuna: %d trials, custom_param_space=%s, undersample_majority=%s',
                self.n_optuna_trials, param_space is not None, undersample_majority,
            )
            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
            )
            study.optimize(
                objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False,
            )

            best_trial = study.best_trial
            self.best_params_ = dict(best_trial.user_attrs['xgb_params'])

            if sampler is not None:
                fraction_value = best_trial.params[sampler.fraction_key]
                idx = sampler.sample_idx(fraction_value, best_trial.number)
                logger.info(
                    '[XGBoost Cls] Best score=%.4f | %s=%.3f (best trial #%d, n=%d/%d) | params=%s',
                    study.best_value, sampler.fraction_key, fraction_value, best_trial.number,
                    len(idx), len(y_tr), self.best_params_,
                )
            else:
                idx = full_idx
                logger.info('[XGBoost Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            Xtr_final, ytr_final = Xtr.iloc[idx], y_tr[idx]
            self._model = xgb.XGBClassifier(**self.best_params_)
            self._model.fit(Xtr_final, ytr_final, eval_set=[(Xva, y_va)], verbose=False)

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if X_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            self.valid_pred_ = self._model.predict_proba(Xva)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict_proba(_prep(X, self.selected_features_, self.cat_features_))[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw


# ── Backward-compat functional wrappers ──────────────────────────────────────

def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    model_settings: dict[str, Any],
    n_optuna_trials: int,
    postprocess_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = XGBoostRegressor(n_optuna_trials=n_optuna_trials, model_settings=model_settings)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    _pp = postprocess_fn or (lambda _X, p: p)
    train_pred = _pp(X_train, model.train_pred_)
    valid_pred = _pp(X_valid, model.valid_pred_)
    infer_pred = _pp(X_inference, model.predict(X_inference))
    logger.info('[XGBoost Reg] Final MAE: %.3f', mean_absolute_error(y_valid, valid_pred))
    return model._model, train_pred, valid_pred, infer_pred, model.best_params_


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    n_optuna_trials: int,
    model_settings: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
    model = XGBoostClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)
    logger.info('[XGBoost Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, model.valid_pred_))
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """XGBoost поддерживает SHAP нативно; отдельная predict_fn не нужна."""
    return
