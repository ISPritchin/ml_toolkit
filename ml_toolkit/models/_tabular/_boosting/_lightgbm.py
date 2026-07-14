"""LightGBM адаптер: классы LightGBMRegressor и LightGBMClassifier.

Residual learning (регрессия): обучается на (y - baseline_col), при predict
добавляет baseline обратно. baseline_col передаётся через model_settings.

Optuna выбирает boosting_type ∈ {gbdt, dart, goss} вместе с остальными
гиперпараметрами. Если params передан в конструктор, Optuna не запускается.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ml_toolkit.models._base import BaseModel, XInput, YInput
from ml_toolkit.models._tabular._boosting._common import add_baseline, compute_residual

if TYPE_CHECKING:
    import optuna
from ml_toolkit.models._tabular._boosting._undersampling import UndersampleSampler
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_multiclass_calibrators,
    fit_calibrator,
    fit_multiclass_calibrators,
    make_lgb_pruning_callback,
    make_study,
    prep_cat_features,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
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


def _boosting_lgb_params(lgb: ModuleType, boosting_type: str) -> dict:
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


def _default_lgb_param_space(trial: optuna.Trial) -> dict[str, Any]:
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
    """LightGBM-регрессор с residual learning, авто-выбором boosting_type и Optuna-тюнингом.

    ``params=None`` запускает Optuna (`X_valid`/`y_valid` обязательны, иначе
    `ValueError`); ``params={...}`` — прямое обучение без тюнинга, `best_params_`
    равен переданным `params` как есть.

    Категориальные признаки (`cat_features` в `fit()`) — нативно через
    `categorical_feature=` в `.fit()` (внутри `_prep()` колонки конвертируются
    в dtype `'category'`), `cat_encoder` не нужен.

    model_settings, которые читает этот класс:

    - ``baseline_col`` (`str | None`, дефолт `None`) — residual learning: модель
      обучается на `(y - baseline)`, а не на `y` напрямую. В отличие от
      `CatBoostRegressor` (где baseline передаётся через `Pool` и CatBoost сам
      прибавляет его к предсказанию), здесь `predict()` прибавляет baseline
      **вручную** после `self._model.predict(...)`. Столбца нет в `X` на
      конкретном вызове `predict()` — просто ничего не прибавляется (важно:
      значит и train, и predict должны получать один и тот же столбец,
      иначе предсказание окажется смещено на величину пропущенного baseline).
    - ``reg_metric`` / ``reg_metric_direction`` — метрика Optuna-objective,
      дефолт `'mae'`.
    - ``param_space`` (`Callable[[trial], dict] | None`) — переопределяет
      дефолтный search space. Может (но не обязан) вернуть ключ
      `'boosting_type'` (`'gbdt'`/`'dart'`/`'goss'`) — если не вернул,
      используется `'gbdt'`. `'dart'` не поддерживает early stopping
      (LightGBM ограничение) — колбэк `lgb.early_stopping` для него не
      подключается ни в Optuna, ни в explicit-params ветке.
    - ``optuna_timeout`` / ``optuna_pruner`` / ``optuna_verbose`` — общие для
      всех Optuna-адаптеров, см. `ml_toolkit/models/model_settings.md`.

    Атрибуты после `fit()`: ``best_params_`` (включая фактический `boosting_type`
    при Optuna-тюнинге), ``selected_features_``, ``cat_features_``,
    ``train_pred_``/``valid_pred_`` (уже с прибавленным baseline и применённым
    `model_settings['postprocess_fn']`, если заданы).

    .. note::
        Как и у `CatBoostRegressor`: `postprocess_fn` применяется только внутри
        `fit()` к `train_pred_`/`valid_pred_`. `predict()` класса его не знает —
        для инференса примените `postprocess_fn` к результату `predict()`
        вручную: `postprocess_fn(X_new, model.predict(X_new))`.

    Примеры::

        # С Optuna, автовыбор boosting_type
        model = LightGBMRegressor(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=['a', 'b'])
        pred = model.predict(X_new)

        # Без Optuna, явные параметры
        model = LightGBMRegressor(params={'n_estimators': 500, 'num_leaves': 31})
        model.fit(X_train, y_train)
        pred = model.predict(X_new)
        print(model.best_params_)

        # Residual learning поверх готового бейзлайна
        model = LightGBMRegressor(params={'n_estimators': 300},
                                   model_settings={'baseline_col': 'my_baseline'})
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_new)   # baseline из X_new прибавлен автоматически
    """

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput | None = None,
        y_valid: YInput | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LightGBMRegressor:
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError('LightGBM not installed. Run: pip install lightgbm') from err

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        cat_in_sel = [c for c in self.cat_features_ if c in self.selected_features_]

        baseline_col: str | None = self.model_settings.get('baseline_col')
        pp: Callable = self.model_settings.get('postprocess_fn') or (lambda _X, p: p)

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        resid_tr, baseline_tr = compute_residual(y_train.values, X_train, baseline_col)

        Xva = resid_va = baseline_va = None
        if X_valid is not None and y_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            resid_va, baseline_va = compute_residual(y_valid.values, X_valid, baseline_col)

        if self.params is None:
            if Xva is None:
                raise ValueError(
                    'X_valid и y_valid обязательны при params=None (нужны для Optuna)'
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel,
                X_valid, y_valid, baseline_va, pp,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                lgb, Xtr, resid_tr, Xva, resid_va, cat_in_sel,
            )

        self.train_pred_ = pp(X_train, add_baseline(self._model.predict(Xtr), baseline_tr))
        if Xva is not None:
            self.valid_pred_ = pp(X_valid, add_baseline(self._model.predict(Xva), baseline_va))
            logger.info('[LGB Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(
        self,
        lgb: ModuleType,
        Xtr: pd.DataFrame,
        resid_tr: np.ndarray,
        Xva: pd.DataFrame | None,
        resid_va: np.ndarray | None,
        cat_in_sel: list[str],
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        baseline_va: np.ndarray | None,
        pp: Callable,
    ):
        import optuna

        baseline_col = self.model_settings.get('baseline_col')
        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS,
        )
        param_space: Callable[[optuna.Trial], dict] | None = self.model_settings.get('param_space')

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
            pred = pp(X_valid, add_baseline(m.predict(Xva), baseline_va))
            return metric_fn(y_valid.values, pred)

        logger.info(
            '[LGB Reg] Optuna: %d trials, baseline=%s, custom_param_space=%s',
            self.n_optuna_trials, baseline_col, param_space is not None,
        )
        ms = self.model_settings
        study = make_study(direction, ms)
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

    def _fit_direct(
        self,
        lgb: ModuleType,
        Xtr: pd.DataFrame,
        resid_tr: np.ndarray,
        Xva: pd.DataFrame | None,
        resid_va: np.ndarray | None,
        cat_in_sel: list[str],
    ):
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
        baseline = X[baseline_col].values if baseline_col and baseline_col in X.columns else None
        return add_baseline(raw, baseline)


# ─────────────────────────────────────────────────────────────────────────────
# Классификатор
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMClassifier(BaseModel):
    """LightGBM-классификатор: бинарный и мультикласс в одном классе.

    Тот же контракт, что у `CatBoostClassifier` — определяется автоматически по
    числу уникальных значений `y_train` (`self.n_classes_`), явно указывать не нужно.

    ``params=None`` запускает Optuna (`X_valid`/`y_valid` обязательны); ``params={...}``
    — прямое обучение без тюнинга (в этой ветке `objective`/`num_class` для
    мультикласса подбирает сам LightGBM по данным, если не заданы явно).

    - **Бинарный** (`n_classes_ == 2`): `predict_proba()` возвращает 1D-массив
      `P(y=1)`. Optuna: `objective='binary'`, внутренняя метрика
      `'average_precision'`. Калибратор — `self.calibrator_`.
    - **Мультикласс**: `predict_proba()` возвращает `(n, K)`-матрицу, строки
      нормированы к 1. Optuna: `objective='multiclass'` + `num_class=K`,
      внутренняя метрика `'auc_mu'` (нативный multiclass AUC LightGBM). Внутренний
      `is_unbalance` в этом режиме не поддерживается LightGBM вовсе — не
      выставляется; балансировка классов идёт только через
      `undersample_majority` (`balance_fraction`). Калибраторы —
      `self.calibrators_`, список из `K` `IsotonicRegression` (OvR); бинарный
      `self.calibrator_` в этом случае остаётся `None`.

    Калибратор(ы) обучаются только если передана валидационная выборка.

    Категориальные признаки — нативно, как у `LightGBMRegressor`.

    model_settings, которые читает этот класс:

    - ``cls_metric`` / ``cls_metric_direction`` — метрика Optuna-objective,
      дефолт `'pr_auc'`. `average_precision_score`/`roc_auc_score` из sklearn
      поддерживают мультикласс напрямую (1D `y_true` + 2D `y_score`), поэтому
      кастомную `cls_metric` можно использовать в обоих режимах без обёрток.
    - ``undersample_majority`` (`bool`, дефолт `False`) — `True` включает урезание
      классов на каждом Optuna-триале (бинарный — `majority_fraction`, мультикласс —
      `balance_fraction`, тоже тюнится Optuna); финальная модель обучается на
      том же сэмпле, что и лучший триал.
    - ``param_space`` — как у `LightGBMRegressor`, тот же `boosting_type`-контракт.
    - ``optuna_timeout`` / ``optuna_pruner`` / ``optuna_verbose`` — см.
      `ml_toolkit/models/model_settings.md`.

    Атрибуты после `fit()`: ``n_classes_``, ``calibrator_``/``calibrators_``,
    ``best_params_``, ``selected_features_``, ``cat_features_``,
    ``train_pred_``/``valid_pred_``.

    Примеры::

        # Бинарная классификация, Optuna
        model = LightGBMClassifier(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_new)          # 1D, откалибровано
        print(model.best_params_)

        # Мультикласс — определяется по y_train автоматически
        model = LightGBMClassifier(n_optuna_trials=50)
        model.fit(X_train_3cls, y_train_3cls, X_valid, y_valid)
        proba = model.predict_proba(X_new)          # (n, K), строки суммируются в 1

        # Без Optuna, явные параметры
        model = LightGBMClassifier(params={'n_estimators': 300, 'num_leaves': 31})
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_new)

        # С сэмплированием классов на каждом Optuna-триале (по умолчанию — без него)
        model = LightGBMClassifier(n_optuna_trials=50, model_settings={'undersample_majority': True})
        model.fit(X_train, y_train, X_valid, y_valid)
    """

    def fit(
        self,
        X_train: XInput,
        y_train: YInput,
        X_valid: XInput | None = None,
        y_valid: YInput | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LightGBMClassifier:
        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError('LightGBM not installed. Run: pip install lightgbm') from err

        import optuna
        _optuna_prev_verbosity = set_optuna_verbosity(self.model_settings)
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.n_classes_ = len(np.unique(y_train.values))
        is_binary = self.n_classes_ == 2
        self.calibrators_ = None  # для мультикласса; бинарный использует self.calibrator_
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
                    'X_valid и y_valid обязательны при params=None (нужны для Optuna)'
                )
            self._model, self.best_params_ = self._fit_with_optuna(
                lgb, Xtr, y_train, Xva, y_valid, cat_in_sel, is_binary,
            )
        else:
            self._model, self.best_params_ = self._fit_direct(
                lgb, Xtr, y_train, Xva, y_valid, cat_in_sel,
            )

        full_tr = self._model.predict_proba(Xtr)
        self.train_pred_ = full_tr[:, 1] if is_binary else full_tr
        if Xva is not None:
            full_va = self._model.predict_proba(Xva)
            if is_binary:
                self.valid_pred_ = full_va[:, 1]
                logger.info('[LGB Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, self.valid_pred_))
                self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.values)
                logger.info('[LGB Cls] Isotonic calibration fitted (n=%d)', len(self.valid_pred_))
            else:
                self.valid_pred_ = full_va
                roc = roc_auc_score(y_valid.values, full_va, multi_class='ovr', average='macro')
                logger.info('[LGB Cls] Final ROC-AUC macro OvR: %.3f', roc)
                self.calibrators_ = fit_multiclass_calibrators(full_va, y_valid.values)
                logger.info('[LGB Cls] Isotonic calibration fitted (%d calibrators)', self.n_classes_)

        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _fit_with_optuna(
        self,
        lgb: ModuleType,
        Xtr: pd.DataFrame,
        y_train: pd.Series,
        Xva: pd.DataFrame,
        y_valid: pd.Series,
        cat_in_sel: list[str],
        is_binary: bool,
    ):
        import optuna

        metric_fn, direction = resolve_metric_fn(
            self.model_settings, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS,
        )
        param_space: Callable[[optuna.Trial], dict] | None = self.model_settings.get('param_space')
        undersample_majority: bool = self.model_settings.get('undersample_majority', False)

        y_arr = np.asarray(y_train)
        full_idx = np.arange(len(y_arr))
        sampler = UndersampleSampler(y_arr, is_binary=is_binary, log_prefix='[LGB Cls]') if undersample_majority else None
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
                'objective': 'binary' if is_binary else 'multiclass',
                'metric': 'average_precision' if is_binary else 'auc_mu',
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
                **_boosting_lgb_params(lgb, boosting_type),
            }
            if is_binary:
                # undersample_majority уже балансирует классы физически — is_unbalance
                # (внутреннее переваживание LightGBM) включаем только если сэмплирование выключено,
                # чтобы не применять два механизма балансировки одновременно. Мультиклассовый
                # objective этот параметр не поддерживает вовсе — балансировка там только
                # через undersample_majority (balance_fraction в UndersampleSampler).
                params['is_unbalance'] = not undersample_majority
            else:
                params['num_class'] = self.n_classes_
            trial.set_user_attr('lgb_params', params)
            trial.set_user_attr('boosting_type', boosting_type)
            m = lgb.LGBMClassifier(**params)
            m.fit(
                Xtr_trial, ytr_trial, eval_set=[(Xva, y_valid)],
                categorical_feature=cat_in_sel or 'auto',
                callbacks=[*_lgb_callbacks(boosting_type), make_lgb_pruning_callback(trial)],
            )
            proba = m.predict_proba(Xva)
            return metric_fn(y_valid.values, proba[:, 1] if is_binary else proba)

        logger.info(
            '[LGB Cls] Optuna: %d trials, custom_param_space=%s, undersample_majority=%s',
            self.n_optuna_trials, param_space is not None, undersample_majority,
        )
        ms = self.model_settings
        study = make_study(direction, ms)
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

    def _fit_direct(
        self,
        lgb: ModuleType,
        Xtr: pd.DataFrame,
        y_train: pd.Series,
        Xva: pd.DataFrame | None,
        y_valid: pd.Series | None,
        cat_in_sel: list[str],
    ):
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
        raw = self._model.predict_proba(Xp)
        if self.n_classes_ == 2:
            score = raw[:, 1]
            return self.calibrator_.predict(score) if self.calibrator_ is not None else score
        if self.calibrators_ is not None:
            return apply_multiclass_calibrators(raw, self.calibrators_)
        return raw

