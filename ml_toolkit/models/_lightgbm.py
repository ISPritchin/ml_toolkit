# ruff: noqa: N806
"""LightGBM адаптер: классы LightGBMRegressor и LightGBMClassifier.

Residual learning (регрессия): обучается на (y - baseline_col), при predict
добавляет baseline обратно. baseline_col передаётся через model_settings.

Optuna выбирает boosting_type ∈ {gbdt, dart, goss} вместе с остальными
гиперпараметрами. Если params передан в конструктор, Optuna не запускается.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._undersampling import UndersampleSampler
from ml_toolkit.models._utils import (
    CLS_METRICS, REG_METRICS, calibrate_proba, fit_calibrator, make_lgb_pruning_callback,
    prep_cat_features, resolve_metric_fn, resolve_pruner, resolve_timeout, set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_prep = prep_cat_features


def _lgb_callbacks(boosting_type: str = 'gbdt') -> list:
    """Создаёт LightGBM callbacks. DART не поддерживает early stopping."""
    import lightgbm as lgb
    callbacks = [lgb.log_evaluation(-1)]
    if boosting_type != 'dart':
        callbacks.insert(0, lgb.early_stopping(100, verbose=False))
    return callbacks


def _boosting_lgb_params(lgb: Any, boosting_type: str) -> dict:
    if boosting_type == 'dart':
        return {'boosting_type': 'dart'}
    if boosting_type == 'goss':
        lgb_ver = tuple(int(x) for x in lgb.__version__.split('.')[:2])
        if lgb_ver >= (4, 0):
            return {'data_sample_strategy': 'goss'}
        return {'boosting_type': 'goss'}
    return {}


def _detect_boosting_type(params: dict) -> str:
    """Определяет boosting_type из словаря параметров (в т.ч. из data_sample_strategy)."""
    if params.get('data_sample_strategy') == 'goss':
        return 'goss'
    return params.get('boosting_type', 'gbdt')


def _default_lgb_param_space(trial: Any) -> dict[str, Any]:
    """Пространство поиска LightGBM по умолчанию (переопределяется model_settings['param_space']).

    Общее для регрессии и классификации — включает выбор boosting_type; ключ
    'boosting_type' извлекается вызывающей стороной перед передачей params в конструктор.
    """
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    dart_params = {}
    if boosting_type == 'dart':
        dart_params = {
            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.5),
            'max_drop': trial.suggest_int('max_drop', 10, 100),
        }
    subsample_params = {} if boosting_type == 'goss' else {
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    return {
        'boosting_type': boosting_type,
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        **subsample_params,
        **dart_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Регрессор
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMRegressor(BaseModel):
    """LightGBM регрессор с опциональным Optuna-тюнингом.

    Поддерживает residual learning: если model_settings содержит 'baseline_col',
    модель обучается на (y - baseline), при predict добавляет baseline обратно.

    Примеры::

        # С Optuna (params=None)
        model = LightGBMRegressor(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=['a', 'b'])
        pred = model.predict(X_new)

        # Без Optuna (явные параметры)
        model = LightGBMRegressor(params={'n_estimators': 500, 'num_leaves': 31})
        model.fit(X_train, y_train)
        pred = model.predict(X_new)
        print(model.best_params_)
    """

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any | None = None,
        y_valid: Any | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> "LightGBMRegressor":
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError('LightGBM not installed. Run: pip install lightgbm') from err

        set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]

        baseline_col: str | None = self.model_settings.get('baseline_col')
        pp: Callable = self.model_settings.get('postprocess_fn') or (lambda _X, p: p)

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        baseline_tr = X_train[baseline_col].values if baseline_col and baseline_col in X_train.columns else None
        resid_tr = y_train.values - baseline_tr if baseline_tr is not None else y_train.values

        Xva = resid_va = baseline_va = None
        if X_valid is not None and y_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            baseline_va = X_valid[baseline_col].values if baseline_col and baseline_col in X_valid.columns else None
            resid_va = y_valid.values - baseline_va if baseline_va is not None else y_valid.values

        if self.params is None:
            if Xva is None:
                raise ValueError(
                    "X_valid и y_valid обязательны при params=None (нужны для Optuna)"
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel,
                X_valid, y_valid, baseline_va, pp,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel,
            )

        self.train_pred_ = pp(X_train, self._model.predict(Xtr) + (baseline_tr if baseline_tr is not None else 0))
        if Xva is not None:
            self.valid_pred_ = pp(X_valid, self._model.predict(Xva) + (baseline_va if baseline_va is not None else 0))
            logger.info('[LGB Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))

        return self

    def _fit_with_optuna(self, lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel,
                         X_valid, y_valid, baseline_va, pp):
        import optuna

        baseline_col = self.model_settings.get('baseline_col', 'fee_nds_amount')
        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS,
        )
        param_space: Callable[[Any], dict] | None = self.model_settings.get('param_space')

        def objective(trial: optuna.Trial) -> float:
            tunable = dict(param_space(trial) if param_space is not None else _default_lgb_param_space(trial))
            boosting_type = tunable.pop('boosting_type', 'gbdt')
            params = {
                **tunable,
                'objective': 'mae',
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
                **_boosting_lgb_params(lgb, boosting_type),
            }
            trial.set_user_attr('lgb_params', params)
            trial.set_user_attr('boosting_type', boosting_type)
            m = lgb.LGBMRegressor(**params)
            m.fit(
                Xtr, resid_tr, eval_set=[(Xva, resid_va)],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=[*_lgb_callbacks(boosting_type), make_lgb_pruning_callback(trial)],
            )
            pred = pp(X_valid, m.predict(Xva) + (baseline_va if baseline_va is not None else 0))
            return metric_fn(y_valid.values, pred)

        logger.info(
            '[LGB Reg] Optuna: %d trials, baseline=%s, custom_param_space=%s',
            self.n_optuna_trials, baseline_col, param_space is not None,
        )
        ms = self.model_settings
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_trial = study.best_trial
        best_params = dict(best_trial.user_attrs['lgb_params'])
        best_bt = best_trial.user_attrs['boosting_type']
        logger.info('[LGB Reg] Best: boosting=%s score=%.4f params=%s', best_bt, study.best_value, best_params)

        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            Xtr, resid_tr, eval_set=[(Xva, resid_va)],
            categorical_feature=cat_in_sel or 'auto',
            callbacks=_lgb_callbacks(best_bt),
        )
        return model, best_params

    def _fit_direct(self, lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel):
        bt = _detect_boosting_type(self.params)
        model = lgb.LGBMRegressor(**self.params)
        if Xva is not None:
            model.fit(
                Xtr, resid_tr, eval_set=[(Xva, resid_va)],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=_lgb_callbacks(bt),
            )
        else:
            model.fit(Xtr, resid_tr, categorical_feature=cat_in_sel or 'auto')
        return model, dict(self.params)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _prep(X, self.selected_features_, self.cat_features_)
        raw = self._model.predict(Xp)
        baseline_col = self.model_settings.get('baseline_col')
        if baseline_col and baseline_col in X.columns:
            return raw + X[baseline_col].values
        return raw


# ─────────────────────────────────────────────────────────────────────────────
# Классификатор
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMClassifier(BaseModel):
    """LightGBM классификатор с опциональным Optuna-тюнингом.

    Если передана валидационная выборка, автоматически обучает изотонический
    калибратор на val-вероятностях. predict_proba() всегда возвращает
    откалиброванные вероятности (при наличии калибратора).

    Примеры::

        # С Optuna
        model = LightGBMClassifier(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_new)
        print(model.best_params_)

        # Без Optuna
        model = LightGBMClassifier(params={'n_estimators': 300, 'num_leaves': 31})
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_new)
    """

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any | None = None,
        y_valid: Any | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> "LightGBMClassifier":
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError('LightGBM not installed. Run: pip install lightgbm') from err

        set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        Xva = None
        if X_valid is not None and y_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)

        if self.params is None:
            if Xva is None:
                raise ValueError(
                    "X_valid и y_valid обязательны при params=None (нужны для Optuna)"
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                lgb, Xtr, y_train, Xva, y_valid, cat_in_sel,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                lgb, Xtr, y_train, Xva, y_valid, cat_in_sel,
            )

        self.train_pred_ = self._model.predict_proba(Xtr)[:, 1]
        if Xva is not None:
            self.valid_pred_ = self._model.predict_proba(Xva)[:, 1]
            logger.info('[LGB Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, self.valid_pred_))
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.values)
            logger.info('[LGB Cls] Isotonic calibration fitted (n=%d)', len(self.valid_pred_))

        return self

    def _fit_with_optuna(self, lgb, Xtr, y_train, Xva, y_valid, cat_in_sel):
        import optuna

        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS,
        )
        param_space: Callable[[Any], dict] | None = self.model_settings.get('param_space')
        undersample_majority: bool = self.model_settings.get('undersample_majority', True)

        y_arr = np.asarray(y_train)
        full_idx = np.arange(len(y_arr))
        # LightGBMClassifier поддерживает только бинарную классификацию.
        sampler = UndersampleSampler(y_arr, is_binary=True, log_prefix='[LGB Cls]') if undersample_majority else None
        if not undersample_majority:
            logger.info('[LGB Cls] undersample_majority=False — обучение на полных данных (n=%d)', len(y_arr))

        def objective(trial: optuna.Trial) -> float:
            if sampler is not None:
                fraction_value = sampler.suggest_fraction(trial)
                idx = sampler.sample_idx(fraction_value, trial.number)
            else:
                idx = full_idx
            Xtr_trial, ytr_trial = Xtr.iloc[idx], y_arr[idx]

            tunable = dict(param_space(trial) if param_space is not None else _default_lgb_param_space(trial))
            boosting_type = tunable.pop('boosting_type', 'gbdt')
            params = {
                **tunable,
                'objective': 'binary',
                'metric': 'average_precision',
                # undersample_majority уже балансирует классы физически — is_unbalance
                # (внутреннее переваживание LightGBM) включаем только если сэмплирование выключено,
                # чтобы не применять два механизма балансировки одновременно.
                'is_unbalance': not undersample_majority,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
                **_boosting_lgb_params(lgb, boosting_type),
            }
            trial.set_user_attr('lgb_params', params)
            trial.set_user_attr('boosting_type', boosting_type)
            m = lgb.LGBMClassifier(**params)
            m.fit(
                Xtr_trial, ytr_trial, eval_set=[(Xva, y_valid)],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=[*_lgb_callbacks(boosting_type), make_lgb_pruning_callback(trial)],
            )
            return metric_fn(y_valid.values, m.predict_proba(Xva)[:, 1])

        logger.info(
            '[LGB Cls] Optuna: %d trials, custom_param_space=%s, undersample_majority=%s',
            self.n_optuna_trials, param_space is not None, undersample_majority,
        )
        ms = self.model_settings
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_trial = study.best_trial
        best_params = dict(best_trial.user_attrs['lgb_params'])
        best_bt = best_trial.user_attrs['boosting_type']

        if sampler is not None:
            fraction_value = best_trial.params[sampler.fraction_key]
            idx = sampler.sample_idx(fraction_value, best_trial.number)
            logger.info(
                '[LGB Cls] Best: boosting=%s score=%.4f | %s=%.3f (best trial #%d, n=%d/%d) | params=%s',
                best_bt, study.best_value, sampler.fraction_key, fraction_value, best_trial.number,
                len(idx), len(y_arr), best_params,
            )
        else:
            idx = full_idx
            logger.info('[LGB Cls] Best: boosting=%s score=%.4f params=%s', best_bt, study.best_value, best_params)

        Xtr_final, ytr_final = Xtr.iloc[idx], y_arr[idx]
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            Xtr_final, ytr_final, eval_set=[(Xva, y_valid)],
            categorical_feature=cat_in_sel or 'auto',
            callbacks=_lgb_callbacks(best_bt),
        )
        return model, best_params

    def _fit_direct(self, lgb, Xtr, y_train, Xva, y_valid, cat_in_sel):
        bt = _detect_boosting_type(self.params)
        model = lgb.LGBMClassifier(**self.params)
        if Xva is not None:
            model.fit(
                Xtr, y_train, eval_set=[(Xva, y_valid)],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=_lgb_callbacks(bt),
            )
        else:
            model.fit(Xtr, y_train, categorical_feature=cat_in_sel or 'auto')
        return model, dict(self.params)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        Xp = _prep(X, self.selected_features_, self.cat_features_)
        raw = self._model.predict_proba(Xp)[:, 1]
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw


# ─────────────────────────────────────────────────────────────────────────────
# Backward-совместимые функции (thin wrappers над классами)
# ─────────────────────────────────────────────────────────────────────────────

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
    ms = dict(model_settings)
    if postprocess_fn is not None:
        ms['postprocess_fn'] = postprocess_fn
    model = LightGBMRegressor(n_optuna_trials=n_optuna_trials, model_settings=ms)
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_pred = model.predict(X_inference)
    return model._model, model.train_pred_, model.valid_pred_, infer_pred, model.best_params_


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
    model = LightGBMClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)  # calibrated by class
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """LightGBM поддерживает SHAP нативно; отдельная predict_fn не нужна."""
    return None
