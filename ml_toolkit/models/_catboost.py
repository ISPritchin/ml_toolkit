"""CatBoost адаптер: классы CatBoostRegressor и CatBoostClassifier.

baseline_col передаётся через model_settings и используется как Pool(baseline=...)
— CatBoost включает baseline в предсказание автоматически.

Если params передан в конструктор, Optuna не запускается.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._undersampling import UndersampleSampler
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_multiclass_calibrators,
    fit_calibrator,
    fit_multiclass_calibrators,
    make_catboost_pruning_callback,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)


def _import_catboost():
    """Ленивый импорт catboost — не ломает импорт модуля если пакет не установлен."""
    try:
        from catboost import CatBoostClassifier as _cls
        from catboost import CatBoostRegressor as _reg
        from catboost import Pool as _pool
        return _cls, _reg, _pool
    except ImportError as err:
        raise ImportError('CatBoost not installed. Run: pip install catboost') from err


def _make_pool(
    Pool: type,
    X: pd.DataFrame,
    y: pd.Series | None,
    cat_features: list[str],
    baseline: np.ndarray | None = None,
):
    return Pool(X, label=y, cat_features=cat_features, baseline=baseline)


def _default_reg_param_space(trial: Any, task_type: str) -> dict[str, Any]:
    """Пространство поиска CatBoost по умолчанию для регрессии (переопределяется model_settings['param_space'])."""
    params: dict[str, Any] = {
        'iterations': trial.suggest_int('iterations', 500, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    if task_type == 'GPU':
        # см. _default_cls_param_space: subsample несовместим с дефолтным
        # bootstrap_type=Bayesian на GPU, rsm на GPU не поддерживается.
        params['task_type'] = 'GPU'
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    else:
        params['bootstrap_type'] = 'Bernoulli'
        params['subsample'] = trial.suggest_float('subsample', 0.3, 1.0)
        params['rsm'] = trial.suggest_float('rsm', 0.3, 1.0)
    return params


def _default_cls_param_space(trial: Any, task_type: str) -> dict[str, Any]:
    """Пространство поиска CatBoost по умолчанию для классификации (переопределяется model_settings['param_space'])."""
    params: dict[str, Any] = {
        'iterations': trial.suggest_int('iterations', 500, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    if task_type == 'GPU':
        # subsample несовместим с дефолтным bootstrap_type=Bayesian на GPU;
        # bagging_temperature — эквивалентный параметр для Bayesian-бутстрапа.
        # rsm не поддерживается на GPU.
        params['task_type'] = 'GPU'
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
    else:
        # bootstrap_type по умолчанию зависит от loss_function (для MultiClass — Bayesian,
        # не поддерживающий subsample) — фиксируем Bernoulli, чтобы subsample был валиден всегда.
        params['bootstrap_type'] = 'Bernoulli'
        params['subsample'] = trial.suggest_float('subsample', 0.3, 1.0)
        params['rsm'] = trial.suggest_float('rsm', 0.3, 1.0)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Регрессор
# ─────────────────────────────────────────────────────────────────────────────

class CatBoostRegressor(BaseModel):
    """CatBoost регрессор с опциональным Optuna-тюнингом.

    Если model_settings содержит 'baseline_col', Pool передаётся с baseline=
    — CatBoost включает его в предсказание нативно. predict(X) тоже использует
    Pool с baseline, поэтому предсказание уже включает baseline (не нужно
    прибавлять вручную, в отличие от LightGBM).

    Примеры::

        # С Optuna
        model = CatBoostRegressor(n_optuna_trials=50,
                                   model_settings={'baseline_col': 'fee_nds_amount'})
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=['a', 'b'])
        pred = model.predict(X_new)

        # Без Optuna
        model = CatBoostRegressor(params={'iterations': 700, 'max_depth': 5})
        model.fit(X_train, y_train)
        pred = model.predict(X_new)
    """

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Any | None = None,
        y_valid: Any | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> CatBoostRegressor:
        _CB_Classifier, _CB_Regressor, Pool = _import_catboost()

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []

        baseline_col: str | None = self.model_settings.get('baseline_col')
        pp: Callable = self.model_settings.get('postprocess_fn') or (lambda _X, p: p)

        baseline_tr = X_train[baseline_col].values if baseline_col and baseline_col in X_train.columns else None
        tr_pool = _make_pool(Pool, X_train[self.selected_features_], y_train, self.cat_features_, baseline_tr)

        va_pool = None
        baseline_va = None
        if X_valid is not None and y_valid is not None:
            baseline_va = X_valid[baseline_col].values if baseline_col and baseline_col in X_valid.columns else None
            va_pool = _make_pool(Pool, X_valid[self.selected_features_], y_valid, self.cat_features_, baseline_va)

        if self.params is None:
            if va_pool is None:
                raise ValueError(
                    'X_valid и y_valid обязательны при params=None (нужны для Optuna)'
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                _CB_Regressor, tr_pool, va_pool,
                X_train, X_valid, y_valid, baseline_col, pp,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(_CB_Regressor, tr_pool, va_pool)

        tr_pred_pool = _make_pool(Pool, X_train[self.selected_features_], None, self.cat_features_, baseline_tr)
        self.train_pred_ = pp(X_train, self._model.predict(tr_pred_pool))
        if va_pool is not None:
            va_pred_pool = _make_pool(Pool, X_valid[self.selected_features_], None, self.cat_features_, baseline_va)
            self.valid_pred_ = pp(X_valid, self._model.predict(va_pred_pool))
            logger.info('[CatBoost Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(self, _CB_Regressor, tr_pool, va_pool,
                         X_train, X_valid, y_valid, baseline_col, pp):
        import optuna

        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS,
        )
        param_space: Callable[[Any], dict] | None = self.model_settings.get('param_space')
        ms = self.model_settings
        task_type: str = ms.get('task_type', 'CPU')

        def objective(trial: optuna.Trial) -> float:
            tunable = param_space(trial) if param_space is not None else _default_reg_param_space(trial, task_type)
            params = {
                **tunable,
                'loss_function': 'MAE',
                'eval_metric': 'MAE',
                'verbose': 0,
                'early_stopping_rounds': 100,
                'random_seed': 42,
            }
            trial.set_user_attr('cb_params', params)
            m = _CB_Regressor(**params)
            if task_type == 'GPU':
                # CatBoost GPU не поддерживает пользовательские callbacks — прунинг для
                # GPU-trial'ов недоступен, trial всегда доучивается до конца.
                m.fit(tr_pool, eval_set=va_pool, verbose=False)
            else:
                pruning_callback = make_catboost_pruning_callback(trial)
                m.fit(tr_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_callback])
                if pruning_callback.pruned:
                    raise optuna.TrialPruned(f'Trial pruned (best iteration {m.get_best_iteration()}).')
            pred = pp(X_valid, m.predict(va_pool))
            return metric_fn(y_valid.values, pred)

        if task_type == 'GPU':
            logger.warning(
                '[CatBoost Reg] task_type=GPU: CatBoost не поддерживает user-defined callbacks '
                'на GPU — Optuna-прунинг для этого тюнинга отключён, trial\'ы доучиваются до конца.'
            )
        logger.info(
            '[CatBoost Reg] Optuna: %d trials, baseline=%s, custom_param_space=%s',
            self.n_optuna_trials, baseline_col, param_space is not None,
        )
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_params = dict(study.best_trial.user_attrs['cb_params'])
        logger.info('[CatBoost Reg] Best score=%.4f params=%s', study.best_value, best_params)

        model = _CB_Regressor(**best_params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False)
        return model, best_params

    def _fit_direct(self, _CB_Regressor, tr_pool, va_pool):
        model = _CB_Regressor(**self.params)
        if va_pool is not None:
            model.fit(tr_pool, eval_set=va_pool, verbose=False)
        else:
            model.fit(tr_pool, verbose=False)
        return model, dict(self.params)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        _, _, Pool = _import_catboost()
        baseline_col = self.model_settings.get('baseline_col')
        baseline = X[baseline_col].values if baseline_col and baseline_col in X.columns else None
        pool = _make_pool(Pool, X[self.selected_features_], None, self.cat_features_, baseline)
        return self._model.predict(pool)


# ─────────────────────────────────────────────────────────────────────────────
# Классификатор
# ─────────────────────────────────────────────────────────────────────────────

class CatBoostClassifier(BaseModel):
    """CatBoost классификатор с опциональным Optuna-тюнингом.

    Если передана валидационная выборка, автоматически обучает изотонический
    калибратор на val-вероятностях. predict_proba() всегда возвращает
    откалиброванные вероятности (при наличии калибратора).

    Примеры::

        # С Optuna
        model = CatBoostClassifier(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_new)
        print(model.best_params_)

        # Без Optuna
        model = CatBoostClassifier(params={'iterations': 700, 'max_depth': 5})
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
    ) -> CatBoostClassifier:
        _CB_Classifier, _CB_Regressor, Pool = _import_catboost()

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        self.n_classes_ = len(np.unique(np.asarray(y_train)))
        self.calibrators_ = None  # для мультикласса; бинарный использует self.calibrator_

        tr_pool = _make_pool(Pool, X_train[self.selected_features_], y_train, self.cat_features_)
        va_pool = None
        if X_valid is not None and y_valid is not None:
            va_pool = _make_pool(Pool, X_valid[self.selected_features_], y_valid, self.cat_features_)

        if self.params is None:
            if va_pool is None:
                raise ValueError(
                    'X_valid и y_valid обязательны при params=None (нужны для Optuna)'
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                _CB_Classifier, Pool, va_pool,
                X_train[self.selected_features_], y_train, y_valid,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(_CB_Classifier, tr_pool, va_pool)

        tr_pred_pool = _make_pool(Pool, X_train[self.selected_features_], None, self.cat_features_)
        full_tr = self._model.predict_proba(tr_pred_pool)
        self.train_pred_ = full_tr[:, 1] if self.n_classes_ == 2 else full_tr

        if va_pool is not None:
            va_pred_pool = _make_pool(Pool, X_valid[self.selected_features_], None, self.cat_features_)
            full_va = self._model.predict_proba(va_pred_pool)
            if self.n_classes_ == 2:
                self.valid_pred_ = full_va[:, 1]
                logger.info('[CatBoost Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, self.valid_pred_))
                self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.values)
                logger.info('[CatBoost Cls] Isotonic calibration fitted (n=%d)', len(self.valid_pred_))
            else:
                self.valid_pred_ = full_va
                roc = roc_auc_score(y_valid.values, full_va, multi_class='ovr', average='macro')
                logger.info('[CatBoost Cls] Final ROC-AUC macro OvR: %.3f', roc)
                self.calibrators_ = fit_multiclass_calibrators(full_va, y_valid.values)
                logger.info('[CatBoost Cls] Isotonic calibration fitted (%d calibrators)', self.n_classes_)

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(self, _CB_Classifier, Pool, va_pool, X_train_feats, y_train, y_valid):
        import optuna

        ms = self.model_settings
        task_type: str = ms.get('task_type', 'CPU')
        metric_fn, direction = resolve_metric_fn(
            ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS,
        )
        param_space: Callable[[Any], dict] | None = ms.get('param_space')
        undersample_majority: bool = ms.get('undersample_majority', True)

        y_arr = np.asarray(y_train)
        is_binary = self.n_classes_ == 2
        full_idx = np.arange(len(y_arr))

        cb_loss = ms.get('loss_function', 'Logloss' if is_binary else 'MultiClass')
        cb_eval = ms.get('eval_metric', 'PRAUC' if is_binary else 'AUC')

        # undersample_majority=True (по умолчанию): урезаем мажоритарный класс, финальная
        # модель обучается на том же сэмпле, что и лучший trial (тот же fraction и тот же
        # seed) — не на всех данных, чтобы гиперпараметры не оценивались на одном объёме
        # данных, а обучались на другом. undersample_majority=False: без сэмплирования,
        # всегда полные данные (как обычный Optuna-тюнинг).
        sampler = UndersampleSampler(y_arr, is_binary, log_prefix='[CatBoost Cls]') if undersample_majority else None
        if not undersample_majority:
            logger.info('[CatBoost Cls] undersample_majority=False — обучение на полных данных (n=%d)', len(y_arr))

        def objective(trial: optuna.Trial) -> float:
            if sampler is not None:
                fraction_value = sampler.suggest_fraction(trial)
                idx = sampler.sample_idx(fraction_value, trial.number)
            else:
                idx = full_idx

            trial_pool = _make_pool(Pool, X_train_feats.iloc[idx], y_arr[idx], self.cat_features_)

            tunable = param_space(trial) if param_space is not None else _default_cls_param_space(trial, task_type)
            params = {
                **tunable,
                'loss_function': cb_loss,
                'eval_metric': cb_eval,
                'verbose': 0,
                'early_stopping_rounds': 100,
                'random_seed': 42,
            }
            trial.set_user_attr('cb_params', params)

            m = _CB_Classifier(**params)
            if task_type == 'GPU':
                # CatBoost GPU не поддерживает пользовательские callbacks (нативный цикл
                # обучения не возвращает управление в Python на каждой итерации) — прунинг
                # для GPU-trial'ов недоступен, trial всегда доучивается до конца.
                m.fit(trial_pool, eval_set=va_pool, verbose=False)
            else:
                pruning_callback = make_catboost_pruning_callback(trial)
                m.fit(trial_pool, eval_set=va_pool, verbose=False, callbacks=[pruning_callback])
                if pruning_callback.pruned:
                    raise optuna.TrialPruned(f'Trial pruned (best iteration {m.get_best_iteration()}).')
            raw = m.predict_proba(va_pool)
            proba = raw[:, 1] if is_binary else raw
            return metric_fn(y_valid.values, proba)

        if task_type == 'GPU':
            logger.warning(
                '[CatBoost Cls] task_type=GPU: CatBoost не поддерживает user-defined callbacks '
                'на GPU — Optuna-прунинг для этого тюнинга отключён, trial\'ы доучиваются до конца.'
            )
        logger.info(
            '[CatBoost Cls] Optuna: %d trials, task_type=%s, custom_param_space=%s, undersample_majority=%s',
            self.n_optuna_trials, task_type, param_space is not None, undersample_majority,
        )
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
        )
        study.optimize(objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False)

        best_trial = study.best_trial
        best_params = dict(best_trial.user_attrs['cb_params'])

        if sampler is not None:
            fraction_value = best_trial.params[sampler.fraction_key]
            idx = sampler.sample_idx(fraction_value, best_trial.number)
            logger.info(
                '[CatBoost Cls] Best score=%.4f | %s=%.3f (best trial #%d, n=%d/%d) | params=%s',
                study.best_value, sampler.fraction_key, fraction_value, best_trial.number,
                len(idx), len(y_arr), best_params,
            )
        else:
            idx = full_idx
            logger.info(
                '[CatBoost Cls] Best score=%.4f (best trial #%d, n=%d) | params=%s',
                study.best_value, best_trial.number, len(idx), best_params,
            )

        final_pool = _make_pool(Pool, X_train_feats.iloc[idx], y_arr[idx], self.cat_features_)
        model = _CB_Classifier(**best_params)
        model.fit(final_pool, eval_set=va_pool, verbose=False)
        return model, best_params

    def _fit_direct(self, _CB_Classifier, tr_pool, va_pool):
        model = _CB_Classifier(**self.params)
        if va_pool is not None:
            model.fit(tr_pool, eval_set=va_pool, verbose=False)
        else:
            model.fit(tr_pool, verbose=False)
        return model, dict(self.params)

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        _, _, Pool = _import_catboost()
        pool = _make_pool(Pool, X[self.selected_features_], None, self.cat_features_)
        raw = self._model.predict_proba(pool)
        if self.n_classes_ == 2:
            score = raw[:, 1]
            return self.calibrator_.predict(score) if self.calibrator_ is not None else score
        if self.calibrators_ is not None:
            return apply_multiclass_calibrators(raw, self.calibrators_)
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
    model = CatBoostRegressor(n_optuna_trials=n_optuna_trials, model_settings=ms)
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
    model = CatBoostClassifier(n_optuna_trials=n_optuna_trials, model_settings=model_settings or {})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features, cat_features)
    infer_proba = model.predict_proba(X_inference)  # calibrated by class
    return model._model, model.train_pred_, model.valid_pred_, infer_proba, model.best_params_


def make_predict_fn(model: Any, task: str, selected_features: list[str]) -> None:
    """CatBoost поддерживает SHAP нативно; отдельная predict_fn не нужна."""
    return
