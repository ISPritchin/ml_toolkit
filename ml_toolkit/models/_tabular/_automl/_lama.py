"""LightAutoML (LAMA) adapter.

LAMA управляет hyperparameter tuning внутри себя.
`n_optuna_trials` используется как таймаут: timeout = n_optuna_trials * 60 секунд.
Переопределить через model_settings['timeout'].

Regression: `fit_predict` возвращает in-sample предикты на train,
`predict` используется для valid / inference.

Classification: бинарный и мультикласс в одном классе, режим определяется
автоматически по числу уникальных значений `y_train` (`self.n_classes_`) —
см. `LAMAClassifier`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import apply_multiclass_calibrators, fit_calibrator, fit_multiclass_calibrators

logger = logging.getLogger(__name__)

_TARGET = '__lama_target__'


def _coerce_cat_dtypes(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Кастует категориальные колонки в legacy object dtype.

    pandas>=3.0 по умолчанию (`future.infer_string`) хранит строковые колонки в
    `StringDtype`, а `lightautoml`'s reader делает `np.issubdtype(dtype, np.number)`
    на каждой колонке — этот вызов падает с TypeError на `StringDtype`, которую
    numpy не умеет интерпретировать как dtype.
    """
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def _build_roles(cat_features: list[str], selected_features: list[str]) -> dict[str, Any]:
    """Формирует словарь ролей для LightAutoML: таргет и категориальные признаки.

    LightAutoML ожидает формат {роль: [колонки]} (роль — ключ), а не {колонка: роль} —
    см. `roles_parser` в lightautoml/reader/base.py.
    """
    roles: dict[str, Any] = {'target': _TARGET}
    cats = [col for col in cat_features if col in selected_features]
    if cats:
        roles['category'] = cats
    return roles


def _resolve_cls_task_params(is_binary: bool, model_settings: dict) -> tuple[str, str, str]:
    """Резолвит (task_name, loss, metric) для LightAutoML `Task` классификации.

    Дефолты: `'binary'`/`'logloss'` или `'multiclass'`/`'crossentropy'` в
    зависимости от `is_binary`, метрика `'auc'` в обоих случаях (для
    `'multiclass'` LightAutoML сам резолвит её в ROC-AUC OvR — см.
    `lightautoml.tasks.common_metric._valid_str_multiclass_metric_names`,
    та же семантика, что `roc_auc_score(..., multi_class='ovr')` у
    `CatBoostClassifier` для мультикласса). `model_settings['loss']`/
    `model_settings['metric']` переопределяют оба значения напрямую — сама
    функция не валидирует их против списка допустимых для LightAutoML, это
    сделает сама `Task` при создании.
    """
    task_name = 'binary' if is_binary else 'multiclass'
    default_loss = 'logloss' if is_binary else 'crossentropy'
    loss = model_settings.get('loss', default_loss)
    metric = model_settings.get('metric', 'auc')
    return task_name, loss, metric


# ── Классы (новый API) ────────────────────────────────────────────────────────

class LAMARegressor(BaseModel):
    """LightAutoML (TabularAutoML) для регрессии. LAMA самостоятельно управляет тюнингом.

    n_optuna_trials используется для расчёта таймаута (n_trials * 60 сек).
    model_settings['timeout'] переопределяет таймаут напрямую.
    model_settings['loss'] / model_settings['metric'] переопределяют лосс/метрику
    задачи LightAutoML (дефолт — `'mae'`/`'mae'`); допустимые значения — см.
    `lightautoml.tasks.base._valid_str_loss_names['reg']` / `_valid_str_metric_names['reg']`.
    model_settings['cpu_limit'] переопределяет число процессов LightAutoML (default 4) —
    на macOS internal multiprocessing (fork) под reader'ом иногда падает с
    "OMP: Error #179: pthread_mutex_init failed"; в этом случае поставьте cpu_limit=1.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LAMARegressor:
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as err:
            raise ImportError('LightAutoML not installed. Run: pip install lightautoml') from err
        if self.params is not None:
            raise ValueError(
                "LAMARegressor не поддерживает явные params — LightAutoML управляет тюнингом "
                "самостоятельно. Передайте params=None и настройте таймаут через "
                "model_settings['timeout'] (по умолчанию n_optuna_trials * 60 сек)."
            )

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings

        baseline_col: str | None = ms.get('baseline_col')
        timeout = int(ms.get('timeout', self.n_optuna_trials * 60))
        loss = ms.get('loss', 'mae')
        metric = ms.get('metric', 'mae')
        if baseline_col and baseline_col in X_train.columns:
            self._feats = list(dict.fromkeys([*self.selected_features_, baseline_col]))
        else:
            self._feats = list(self.selected_features_)
        logger.info('[LAMA Reg] timeout=%ds, loss=%s, metric=%s, baseline=%s', timeout, loss, metric, baseline_col)

        cpu_limit = int(ms.get('cpu_limit', 4))
        automl = TabularAutoML(
            task=Task('reg', loss=loss, metric=metric),
            timeout=timeout,
            cpu_limit=cpu_limit,
            reader_params={'cv': 5, 'random_state': 42, 'n_jobs': cpu_limit},
        )

        train_df = _coerce_cat_dtypes(X_train[self._feats].copy(), self.cat_features_)
        train_df[_TARGET] = y_train.values

        automl.fit_predict(train_df, roles=_build_roles(self.cat_features_, self.selected_features_), verbose=0)
        self._model = automl

        self.train_pred_ = np.array(automl.predict(train_df).data[:, 0])
        self.best_params_ = {
            'timeout': timeout, 'task': 'reg', 'loss': loss, 'metric': metric, 'cv': 5, 'cpu_limit': cpu_limit,
        }

        if X_valid is not None:
            valid_df = _coerce_cat_dtypes(X_valid[self._feats].copy(), self.cat_features_)
            self.valid_pred_ = np.array(automl.predict(valid_df).data[:, 0])
            logger.info('[LAMA Reg] Final MAE: %.3f', mean_absolute_error(y_valid, self.valid_pred_))
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        df = _coerce_cat_dtypes(X[self._feats].copy(), self.cat_features_)
        return np.array(self._model.predict(df).data[:, 0])


class LAMAClassifier(BaseModel):
    """LightAutoML (TabularAutoML) для классификации: бинарный и мультикласс в одном классе.

    n_optuna_trials используется для расчёта таймаута. Вероятности калибруются изотонической регрессией.

    Бинарный или мультикласс определяется автоматически по числу уникальных значений
    `y_train` (`self.n_classes_`), явно указывать не нужно — тот же контракт, что у
    `CatBoostClassifier`:

    - **Бинарный** (`n_classes_ == 2`): `predict_proba()` возвращает 1D-массив
      `P(y=1)`. Калибратор — `self.calibrator_` (`IsotonicRegression`).
    - **Мультикласс**: `predict_proba()` возвращает `(n, K)`-матрицу, строки
      нормированы к 1 (LightAutoML нумерует классы по возрастанию значения
      `y_train`, когда таргет уже целочисленный 0..K-1 — наш случай, см.
      `lightautoml.reader.base.Reader.check_class_target`). Калибраторы —
      `self.calibrators_`, список из `K` `IsotonicRegression` (по одной на
      класс, схема One-vs-Rest); бинарный `self.calibrator_` в этом случае
      остаётся `None`.

    Калибратор(ы) обучаются только если передана валидационная выборка; без
    неё `predict_proba()` возвращает сырые (некалиброванные) вероятности.

    model_settings['loss'] / model_settings['metric'] переопределяют лосс/метрику
    задачи LightAutoML напрямую (дефолт — `'logloss'`/`'auc'` для бинарного,
    `'crossentropy'`/`'auc'` для мультикласса; `'auc'` для мультикласса LightAutoML
    сам резолвит в ROC-AUC OvR). Допустимые значения — см.
    `lightautoml.tasks.base._valid_str_loss_names` / `_valid_str_metric_names`
    (ключи `'binary'`/`'multiclass'`).
    model_settings['cpu_limit'] переопределяет число процессов LightAutoML (default 4), см. LAMARegressor.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> LAMAClassifier:
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML
            from lightautoml.tasks import Task
        except ImportError as err:
            raise ImportError('LightAutoML not installed. Run: pip install lightautoml') from err
        if self.params is not None:
            raise ValueError(
                "LAMAClassifier не поддерживает явные params — LightAutoML управляет тюнингом "
                "самостоятельно. Передайте params=None и настройте таймаут через "
                "model_settings['timeout'] (по умолчанию n_optuna_trials * 60 сек)."
            )

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        self.n_classes_ = len(np.unique(np.asarray(y_train)))
        self.calibrator_ = None
        self.calibrators_ = None  # для мультикласса; бинарный использует self.calibrator_
        is_binary = self.n_classes_ == 2
        ms = self.model_settings

        timeout = int(ms.get('timeout', self.n_optuna_trials * 60))
        task_name, loss, metric = _resolve_cls_task_params(is_binary, ms)
        self._feats = self.selected_features_
        logger.info(
            '[LAMA Cls] timeout=%ds, task=%s, loss=%s, metric=%s, n_classes=%d',
            timeout, task_name, loss, metric, self.n_classes_,
        )

        cpu_limit = int(ms.get('cpu_limit', 4))
        automl = TabularAutoML(
            task=Task(task_name, loss=loss, metric=metric),
            timeout=timeout,
            cpu_limit=cpu_limit,
            reader_params={'cv': 5, 'random_state': 42, 'n_jobs': cpu_limit},
        )

        train_df = _coerce_cat_dtypes(X_train[self._feats].copy(), self.cat_features_)
        train_df[_TARGET] = y_train.values if hasattr(y_train, 'values') else y_train

        automl.fit_predict(train_df, roles=_build_roles(self.cat_features_, self.selected_features_), verbose=0)
        self._model = automl

        full_tr = automl.predict(train_df).data
        self.train_pred_ = full_tr[:, 0] if is_binary else full_tr
        self.best_params_ = {
            'timeout': timeout, 'task': task_name, 'loss': loss, 'metric': metric,
            'cv': 5, 'cpu_limit': cpu_limit,
        }

        if X_valid is not None:
            valid_df = _coerce_cat_dtypes(X_valid[self._feats].copy(), self.cat_features_)
            full_va = automl.predict(valid_df).data
            y_valid_arr = y_valid.values if hasattr(y_valid, 'values') else np.asarray(y_valid)
            if is_binary:
                self.valid_pred_ = full_va[:, 0]
                logger.info('[LAMA Cls] Final PR-AUC: %.3f', average_precision_score(y_valid, self.valid_pred_))
                self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid_arr.astype(int))
                logger.info('[LAMA Cls] Isotonic calibration fitted (n=%d)', len(self.valid_pred_))
            else:
                self.valid_pred_ = full_va
                roc = roc_auc_score(y_valid_arr, full_va, multi_class='ovr', average='macro')
                logger.info('[LAMA Cls] Final ROC-AUC macro OvR: %.3f', roc)
                self.calibrators_ = fit_multiclass_calibrators(full_va, y_valid_arr)
                logger.info('[LAMA Cls] Isotonic calibration fitted (%d calibrators)', self.n_classes_)

        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        df = _coerce_cat_dtypes(X[self._feats].copy(), self.cat_features_)
        raw = self._model.predict(df).data
        if self.n_classes_ == 2:
            score = raw[:, 0]
            return self.calibrator_.predict(score) if self.calibrator_ is not None else score
        if self.calibrators_ is not None:
            return apply_multiclass_calibrators(raw, self.calibrators_)
        return raw
