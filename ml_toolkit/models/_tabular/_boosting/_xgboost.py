from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._loss_spec import LossSpec, build_loss, suggest_loss_params, to_xgboost_objective
from ml_toolkit.models._tabular._boosting._common import add_baseline, compute_residual
from ml_toolkit.models._tabular._boosting._undersampling import UndersampleSampler
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    apply_multiclass_calibrators,
    fit_calibrator,
    fit_multiclass_calibrators,
    make_study,
    make_xgb_pruning_callback,
    prep_cat_features,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

logger = logging.getLogger(__name__)

_prep = prep_cat_features


def _resolve_xgb_eval_metric(
    xgb: Any, custom_eval_metric: str | Callable | tuple[Callable, str] | None,
) -> tuple[dict[str, Any], int | None, list]:
    """model_settings['eval_metric'] -> (доп. ключи params, early_stopping_rounds, доп. callbacks).

    XGBoost'овский шорткат ``early_stopping_rounds=<int>`` сам угадывает maximize/minimize
    по ИМЕНИ метрики через маленький вайтлист префиксов (``'auc'``, ``'aucpr'``, ``'pre'``,
    ``'map'``, ``'ndcg'`` — см. ``xgboost.callback.EarlyStopping._update_rounds``) — корректно
    для встроенных строковых метрик XGBoost, но НЕ для произвольных callable: имя
    ``sklearn.metrics.roc_auc_score`` не матчится ни под один префикс и тихо считается
    minimize, из-за чего `best_iteration` схлопывается в 0 (верифицировано вживую на
    xgboost==3.2.0 — без этого фикса кастомная "чем больше, тем лучше" метрика ломает модель
    молча). Поэтому голый callable здесь не принимается — только `(callable, direction)`,
    направление явное, и early stopping идёт через `xgboost.callback.EarlyStopping(maximize=...)`
    вместо шортката. Строки проходят как раньше — встроенные имена XGBoost'овский вайтлист
    матчит корректно.
    """
    if custom_eval_metric is None or isinstance(custom_eval_metric, str):
        params = {'eval_metric': custom_eval_metric} if custom_eval_metric is not None else {}
        return params, 100, []
    if isinstance(custom_eval_metric, tuple):
        metric_fn, direction = custom_eval_metric
        if direction not in ('minimize', 'maximize'):
            raise ValueError(
                f"model_settings['eval_metric'] = (callable, direction) — direction должен быть "
                f"'minimize' или 'maximize', получено {direction!r}"
            )
        es = xgb.callback.EarlyStopping(
            rounds=100, maximize=(direction == 'maximize'),
            metric_name=getattr(metric_fn, '__name__', None), save_best=True,
        )
        return {'eval_metric': metric_fn}, None, [es]
    raise ValueError(
        "model_settings['eval_metric'] для XGBoost: голый callable неоднозначен по направлению "
        "оптимизации — XGBoost угадывает его по имени функции (см. model_settings.md), что для "
        "произвольных функций (например sklearn.metrics.roc_auc_score) даёт неверный результат. "
        "Передайте (callable, 'maximize'|'minimize'), например (roc_auc_score, 'maximize')."
    )


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
    """XGBRegressor с нативными категориями, residual learning и Optuna-тюнингом (`reg:absoluteerror`/MAE).

    ``params=None`` запускает Optuna (`X_valid`/`y_valid` обязательны, иначе
    `ValueError`); ``params={...}`` — прямое обучение без тюнинга. В обеих
    ветках `enable_categorical` форсируется в `bool(cat_features)` —
    выставлять его вручную в `params` не нужно (и бесполезно: значение всё
    равно перезаписывается по факту наличия `cat_features`).

    Категориальные признаки (`cat_features` в `fit()`) — нативно через
    `dtype='category'` (`_prep()` конвертирует колонки) + `enable_categorical=True`;
    `cat_encoder` не нужен.

    model_settings, которые читает этот класс:

    - ``baseline_col`` (`str | None`, дефолт `None`) — residual learning, тот же
      контракт, что у `LightGBMRegressor`: модель обучается на `(y - baseline)`,
      а `predict()` прибавляет baseline обратно вручную (в отличие от
      `CatBoostRegressor`, где это делает нативный `Pool(baseline=...)`).
      Столбца нет в `X` на конкретном вызове — просто ничего не вычитается/не
      прибавляется; train и predict должны получать один и тот же столбец.
    - ``reg_metric`` / ``reg_metric_direction`` — метрика Optuna-objective,
      дефолт `'mae'`.
    - ``eval_metric`` (`str | tuple[callable, Literal['minimize', 'maximize']] | None`,
      без дефолта) — метрика раннего останова/Optuna-пруинга, независима от
      `loss_spec`/`reg_metric`; не задана — ключ не попадает в params, XGBoost
      сам выводит её из `objective`; строка — полностью заменяет. Голый
      callable **не принимается** (`ValueError`) — XGBoost угадывает
      maximize/minimize по имени функции, что для произвольных callable
      (например `sklearn.metrics.roc_auc_score`) даёт неверный результат
      молча (`best_iteration` схлопывается в 0); используйте
      `(callable, 'maximize' | 'minimize')`. См. `model_settings.md`, раздел
      «Метрика раннего останова».
    - ``param_space`` (`Callable[[trial], dict] | None`) — переопределяет
      дефолтный search space (`n_estimators`/`max_depth`/`learning_rate`/
      `subsample`/`colsample_bytree`/`reg_alpha`/`reg_lambda`). `objective`/
      `random_state`/`enable_categorical`/`early_stopping_rounds` (и `eval_metric`,
      если задан через `model_settings`) подставляются адаптером и приоритетнее
      одноимённых ключей из `param_space`.
    - ``optuna_timeout`` / ``optuna_pruner`` / ``optuna_verbose`` — общие для
      всех Optuna-адаптеров, см. `ml_toolkit/models/model_settings.md`.

    Атрибуты после `fit()`: ``best_params_`` (в explicit-params режиме включает
    добавленный `enable_categorical`), ``selected_features_``, ``cat_features_``,
    ``train_pred_``/``valid_pred_`` (уже с прибавленным baseline и применённым
    `model_settings['postprocess_fn']`, если заданы).

    .. note::
        Как и у `CatBoostRegressor`/`LightGBMRegressor`: `postprocess_fn`
        применяется внутри `fit()` к `train_pred_`/`valid_pred_`, и — при
        `params=None` — внутри самого Optuna-objective (метрика trial'ов
        считается на постобработанных предсказаниях, не на сырых). `predict()`
        класса его не знает — для инференса примените `postprocess_fn` к
        результату `predict()` вручную: `postprocess_fn(X_new, model.predict(X_new))`.

    Примеры::

        # С Optuna
        model = XGBoostRegressor(n_optuna_trials=50, model_settings={'reg_metric': 'rmse'})
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=['a', 'b'])
        pred = model.predict(X_new)

        # Без Optuna, явные параметры + категориальные признаки
        model = XGBoostRegressor(params={'n_estimators': 500, 'max_depth': 5})
        model.fit(X_train, y_train, cat_features=['region'])   # enable_categorical добавлен сам
        pred = model.predict(X_new)

        # Residual learning поверх готового бейзлайна
        model = XGBoostRegressor(params={'n_estimators': 300},
                                  model_settings={'baseline_col': 'my_baseline'})
        model.fit(X_train, y_train, X_valid, y_valid)
        pred = model.predict(X_new)   # baseline из X_new прибавлен автоматически
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

        baseline_col: str | None = ms.get('baseline_col')
        pp: Callable = ms.get('postprocess_fn') or (lambda _X, p: p)

        # XGBoost uses category dtype — no OrdinalEncoder stored
        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        resid_tr, baseline_tr = compute_residual(y_train.to_numpy(dtype=float), X_train, baseline_col)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        Xva = baseline_va = resid_va = None
        if X_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            resid_va, baseline_va = compute_residual(y_valid.to_numpy(dtype=float), X_valid, baseline_col)

        if self.params is not None:
            # enable_categorical форсируется по has_cat так же, как в Optuna-ветке ниже —
            # иначе XGBoost падает на dtype='category', выставленный _prep(), при
            # explicit params без ручного enable_categorical=True от вызывающего.
            direct_params = {**self.params, 'enable_categorical': has_cat}
            self._model = xgb.XGBRegressor(**direct_params)
            eval_set = [(Xva, resid_va)] if Xva is not None else []
            self._model.fit(Xtr, resid_tr, eval_set=eval_set or None, verbose=False)
            self.best_params_ = direct_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            param_space: Callable[[optuna.Trial], dict] | None = ms.get('param_space')
            loss_spec: LossSpec | None = ms.get('loss_spec')
            # Без дефолта: не задан — ключ eval_metric не попадает в params, XGBoost сам выводит
            # его из objective (см. model_settings.md). Строка или (callable, direction) —
            # полностью заменяет вывод XGBoost; см. _resolve_xgb_eval_metric про то, почему
            # голый callable не принимается.
            custom_eval_metric = ms.get('eval_metric')
            eval_metric_params, es_rounds, _ = _resolve_xgb_eval_metric(xgb, custom_eval_metric)

            def objective(trial: optuna.Trial) -> float:
                # dict(...) — защитная копия: suggest_loss_params ниже может .pop() совпавшие
                # по имени ключи из tunable, мутировать объект, который вернула пользовательская
                # param_space, не нужно.
                tunable = dict(param_space(trial) if param_space is not None else _default_xgb_param_space(trial))
                if loss_spec is not None:
                    loss_p = suggest_loss_params(loss_spec, trial, tunable)
                    objective_val = to_xgboost_objective(build_loss(loss_spec, loss_p))
                    trial.set_user_attr('loss_p', loss_p)
                else:
                    objective_val = 'reg:absoluteerror'
                params = {
                    **tunable,
                    'objective': objective_val,
                    **eval_metric_params,
                    'random_state': 42, 'enable_categorical': has_cat,
                    **({'early_stopping_rounds': es_rounds} if es_rounds is not None else {}),
                }
                trial.set_user_attr('xgb_params', params)
                # xgboost >= 2.x: callbacks — параметр конструктора, не .fit() (в отличие
                # от early_stopping_rounds, который остаётся валиден и там, и там). Свежий
                # EarlyStopping-колбэк на каждый trial — объект хранит состояние между итерациями,
                # переиспользовать один и тот же экземпляр между trial'ами нельзя.
                _, _, extra_cb = _resolve_xgb_eval_metric(xgb, custom_eval_metric)
                m = xgb.XGBRegressor(**params, callbacks=[make_xgb_pruning_callback(trial), *extra_cb])
                m.fit(Xtr, resid_tr, eval_set=[(Xva, resid_va)], verbose=False)
                pred = pp(X_valid, add_baseline(m.predict(Xva), baseline_va))
                return metric_fn(y_valid.values, pred)

            logger.info(
                '[XGBoost Reg] Optuna: %d trials, baseline=%s, custom_param_space=%s',
                self.n_optuna_trials, baseline_col, param_space is not None,
            )
            study = make_study(direction, ms)
            study.optimize(
                objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False,
            )
            best_trial = study.best_trial
            self.best_params_ = dict(best_trial.user_attrs['xgb_params'])

            _, _, extra_cb_final = _resolve_xgb_eval_metric(xgb, custom_eval_metric)
            self._model = xgb.XGBRegressor(**self.best_params_, callbacks=extra_cb_final)
            self._model.fit(Xtr, resid_tr, eval_set=[(Xva, resid_va)], verbose=False)
            if loss_spec is not None:
                # objective в best_params_ — живой callable (нужен для реконструкции модели
                # выше); для человекочитаемого best_params_/лога добавляем имя+параметры.
                self.best_params_ = {**self.best_params_, 'loss_name': loss_spec.name,
                                      'loss_params': dict(best_trial.user_attrs['loss_p'])}
            logger.info('[XGBoost Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

        self.train_pred_ = pp(X_train, add_baseline(self._model.predict(Xtr), baseline_tr))
        if X_valid is not None:
            self.valid_pred_ = pp(X_valid, add_baseline(self._model.predict(Xva), baseline_va))
            logger.info('[XGBoost Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict(_prep(X, self.selected_features_, self.cat_features_))
        baseline_col = self.model_settings.get('baseline_col')
        baseline = X[baseline_col].values if baseline_col and baseline_col in X.columns else None
        return add_baseline(raw, baseline)


class XGBoostClassifier(BaseModel):
    """XGBClassifier: бинарный и мультикласс в одном классе.

    Тот же контракт, что у `CatBoostClassifier`/`LightGBMClassifier` —
    определяется автоматически по числу уникальных значений `y_train`
    (`self.n_classes_`).

    ``params=None`` запускает Optuna (`X_valid`/`y_valid` обязательны); ``params={...}``
    — прямое обучение без тюнинга (в этой ветке `objective`/`num_class` для
    мультикласса подбирает сам XGBoost по данным, если не заданы явно — так же,
    как у `LightGBMClassifier`). В обеих ветках `enable_categorical` форсируется
    в `bool(cat_features)`, как у `XGBoostRegressor`.

    - **Бинарный** (`n_classes_ == 2`): `predict_proba()` возвращает 1D-массив
      `P(y=1)`. Optuna: `objective='binary:logistic'`. Калибратор — `self.calibrator_`.
    - **Мультикласс**: `predict_proba()` возвращает `(n, K)`-матрицу, строки
      нормированы к 1. Optuna: `objective='multi:softprob'` + `num_class=K`.
      Калибраторы — `self.calibrators_`, список из `K` `IsotonicRegression`
      (OvR); бинарный `self.calibrator_` в этом случае остаётся `None`.
    - ``eval_metric`` (`str | tuple[callable, Literal['minimize', 'maximize']] | None`,
      без дефолта) — метрика раннего останова/Optuna-пруинга; не задана —
      XGBoost сам выводит её из `objective` (`'logloss'` для бинарного,
      `'mlogloss'` для мультикласса — верифицировано вживую на xgboost==3.2.0);
      строка — полностью заменяет. Голый callable **не принимается**
      (`ValueError`) — см. тот же пункт в докстринге `XGBoostRegressor` про
      `(callable, 'maximize' | 'minimize')`. См. `model_settings.md`, раздел
      «Метрика раннего останова».

    Калибратор(ы) обучаются только если передана валидационная выборка.

    Категориальные признаки — нативно, как у `XGBoostRegressor`.

    model_settings, которые читает этот класс:

    - ``cls_metric`` / ``cls_metric_direction`` — метрика Optuna-objective,
      дефолт `'pr_auc'`. `average_precision_score`/`roc_auc_score` из sklearn
      поддерживают мультикласс напрямую (1D `y_true` + 2D `y_score`), поэтому
      кастомную `cls_metric` можно использовать в обоих режимах без обёрток.
    - ``undersample_majority`` (`bool`, дефолт `False`) — `True` включает урезание
      классов на каждом Optuna-триале (бинарный — `majority_fraction`, мультикласс —
      `balance_fraction`, тоже тюнится Optuna); финальная модель обучается на
      том же сэмпле, что и лучший триал. По умолчанию — без сэмплирования.
    - ``param_space`` — как у `XGBoostRegressor`.
    - ``optuna_timeout`` / ``optuna_pruner`` / ``optuna_verbose`` — см.
      `ml_toolkit/models/model_settings.md`.

    Атрибуты после `fit()`: ``n_classes_``, ``calibrator_``/``calibrators_``,
    ``best_params_``, ``selected_features_``, ``cat_features_``,
    ``train_pred_``/``valid_pred_``.

    Примеры::

        # Бинарная классификация, Optuna
        model = XGBoostClassifier(n_optuna_trials=50)
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_new)          # 1D, откалибровано
        print(model.best_params_)

        # Мультикласс — определяется по y_train автоматически
        model = XGBoostClassifier(n_optuna_trials=50)
        model.fit(X_train_3cls, y_train_3cls, X_valid, y_valid)
        proba = model.predict_proba(X_new)          # (n, K), строки суммируются в 1

        # Без Optuna, явные параметры
        model = XGBoostClassifier(params={'n_estimators': 300, 'max_depth': 5})
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_new)

        # С сэмплированием классов на каждом Optuna-триале (по умолчанию — без него)
        model = XGBoostClassifier(n_optuna_trials=50, model_settings={'undersample_majority': True})
        model.fit(X_train, y_train, X_valid, y_valid)
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
        self.n_classes_ = len(np.unique(y_train.values))
        is_binary = self.n_classes_ == 2
        self.calibrators_ = None  # для мультикласса; бинарный использует self.calibrator_
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)
        has_cat = bool(self.cat_features_)

        Xtr = _prep(X_train, self.selected_features_, self.cat_features_)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            # enable_categorical форсируется по has_cat так же, как в Optuna-ветке ниже.
            direct_params = {**self.params, 'enable_categorical': has_cat}
            self._model = xgb.XGBClassifier(**direct_params)
            eval_set = []
            if X_valid is not None:
                Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
                eval_set = [(Xva, y_valid.to_numpy(dtype=int))]
            self._model.fit(Xtr, y_tr, eval_set=eval_set or None, verbose=False)
            self.best_params_ = direct_params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            y_va = y_valid.to_numpy(dtype=int)
            param_space: Callable[[optuna.Trial], dict] | None = ms.get('param_space')
            undersample_majority: bool = ms.get('undersample_majority', False)
            loss_spec: LossSpec | None = ms.get('loss_spec')
            # Без дефолта: не задан — ключ eval_metric не попадает в params, XGBoost сам выводит
            # его из objective. Строка или (callable, direction) — полностью заменяет вывод
            # XGBoost; см. _resolve_xgb_eval_metric про то, почему голый callable не принимается.
            custom_eval_metric = ms.get('eval_metric')
            eval_metric_params, es_rounds, _ = _resolve_xgb_eval_metric(xgb, custom_eval_metric)

            full_idx = np.arange(len(y_tr))
            sampler = UndersampleSampler(y_tr, is_binary=is_binary, log_prefix='[XGBoost Cls]') if undersample_majority else None
            if not undersample_majority:
                logger.info('[XGBoost Cls] undersample_majority=False — обучение на полных данных (n=%d)', len(y_tr))

            def objective(trial: optuna.Trial) -> float:
                if sampler is not None:
                    fraction_value = sampler.suggest_fraction(trial)
                    idx = sampler.sample_idx(fraction_value, trial.number)
                else:
                    idx = full_idx
                Xtr_trial, ytr_trial = Xtr.iloc[idx], y_tr[idx]

                # dict(...) — защитная копия: suggest_loss_params ниже может .pop() совпавшие
                # по имени ключи из tunable, мутировать объект, который вернула пользовательская
                # param_space, не нужно.
                tunable = dict(param_space(trial) if param_space is not None else _default_xgb_param_space(trial))
                if loss_spec is not None:
                    loss_p = suggest_loss_params(loss_spec, trial, tunable)
                    objective_val = to_xgboost_objective(build_loss(loss_spec, loss_p))
                    trial.set_user_attr('loss_p', loss_p)
                else:
                    objective_val = 'binary:logistic' if is_binary else 'multi:softprob'
                params = {
                    **tunable,
                    'objective': objective_val,
                    **eval_metric_params,
                    'random_state': 42, 'enable_categorical': has_cat,
                    **({'early_stopping_rounds': es_rounds} if es_rounds is not None else {}),
                }
                if not is_binary:
                    params['num_class'] = self.n_classes_
                trial.set_user_attr('xgb_params', params)
                # Свежий EarlyStopping-колбэк на каждый trial — объект хранит состояние
                # между итерациями, переиспользовать один и тот же экземпляр нельзя.
                _, _, extra_cb = _resolve_xgb_eval_metric(xgb, custom_eval_metric)
                m = xgb.XGBClassifier(**params, callbacks=[make_xgb_pruning_callback(trial), *extra_cb])
                m.fit(Xtr_trial, ytr_trial, eval_set=[(Xva, y_va)], verbose=False)
                # predict_proba() у XGBoost уже возвращает настоящие вероятности даже при
                # кастомном objective (в отличие от LightGBM) — Booster помнит objective
                # 'multi:softprob'/аналог для бинарного на уровне learner config независимо
                # от Python-функции, считавшей градиенты; верифицировано вживую (xgboost 3.2.0).
                proba = m.predict_proba(Xva)
                return metric_fn(y_va, proba[:, 1] if is_binary else proba)

            logger.info(
                '[XGBoost Cls] Optuna: %d trials, custom_param_space=%s, undersample_majority=%s',
                self.n_optuna_trials, param_space is not None, undersample_majority,
            )
            study = make_study(direction, ms)
            study.optimize(
                objective, n_trials=self.n_optuna_trials, timeout=resolve_timeout(ms), show_progress_bar=False,
            )

            best_trial = study.best_trial
            self.best_params_ = dict(best_trial.user_attrs['xgb_params'])

            if sampler is not None:
                fraction_value = best_trial.params[sampler.fraction_key]
                idx = sampler.sample_idx(fraction_value, best_trial.number)
            else:
                idx = full_idx

            Xtr_final, ytr_final = Xtr.iloc[idx], y_tr[idx]
            _, _, extra_cb_final = _resolve_xgb_eval_metric(xgb, custom_eval_metric)
            self._model = xgb.XGBClassifier(**self.best_params_, callbacks=extra_cb_final)
            self._model.fit(Xtr_final, ytr_final, eval_set=[(Xva, y_va)], verbose=False)
            if loss_spec is not None:
                # objective в best_params_ — живой callable (нужен для реконструкции модели
                # выше); для человекочитаемого best_params_/лога добавляем имя+параметры.
                self.best_params_ = {**self.best_params_, 'loss_name': loss_spec.name,
                                      'loss_params': dict(best_trial.user_attrs['loss_p'])}
            if sampler is not None:
                logger.info(
                    '[XGBoost Cls] Best score=%.4f | %s=%.3f (best trial #%d, n=%d/%d) | params=%s',
                    study.best_value, sampler.fraction_key, fraction_value, best_trial.number,
                    len(idx), len(y_tr), self.best_params_,
                )
            else:
                logger.info('[XGBoost Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

        full_tr = self._model.predict_proba(Xtr)
        self.train_pred_ = full_tr[:, 1] if is_binary else full_tr
        if X_valid is not None:
            Xva = _prep(X_valid, self.selected_features_, self.cat_features_)
            full_va = self._model.predict_proba(Xva)
            y_va_arr = y_valid.to_numpy(dtype=int)
            if is_binary:
                self.valid_pred_ = full_va[:, 1]
                self.calibrator_ = fit_calibrator(self.valid_pred_, y_va_arr)
            else:
                self.valid_pred_ = full_va
                roc = roc_auc_score(y_va_arr, full_va, multi_class='ovr', average='macro')
                logger.info('[XGBoost Cls] Final ROC-AUC macro OvR: %.3f', roc)
                self.calibrators_ = fit_multiclass_calibrators(full_va, y_va_arr)
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        # В отличие от LightGBM, predict_proba() у XGBoost уже возвращает настоящие
        # вероятности даже при кастомном objective (loss_spec) — см. комментарий в
        # _fit_with_optuna выше; никакой дополнительной трансформации здесь не нужно.
        raw = self._model.predict_proba(_prep(X, self.selected_features_, self.cat_features_))
        if self.n_classes_ == 2:
            score = raw[:, 1]
            return self.calibrator_.predict(score) if self.calibrator_ is not None else score
        if self.calibrators_ is not None:
            return apply_multiclass_calibrators(raw, self.calibrators_)
        return raw

