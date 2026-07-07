"""Базовый класс для всех адаптеров моделей."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

import numpy as np
import pandas as pd


def _to_pandas(df: Any) -> pd.DataFrame:
    """Конвертирует Polars DataFrame в pandas, pandas возвращает как есть."""
    if isinstance(df, pd.DataFrame):
        return df
    # Polars: проверяем наличие .to_pandas() без жёсткого импорта
    if hasattr(df, 'to_pandas'):
        # use_pyarrow_extension_array=False — нужны NumPy-backed массивы для sklearn/lgbm/catboost
        return df.to_pandas(use_pyarrow_extension_array=False)
    raise TypeError(
        f'Ожидается pandas.DataFrame или polars.DataFrame, получено {type(df).__name__!r}'
    )


def _to_numpy(series: Any) -> np.ndarray:
    """Конвертирует pandas Series, Polars Series или ndarray в numpy."""
    if isinstance(series, np.ndarray):
        return series
    if isinstance(series, pd.Series):
        return series.values
    if hasattr(series, 'to_numpy'):  # polars Series
        return series.to_numpy()
    raise TypeError(
        f'Ожидается pd.Series, pl.Series или np.ndarray, получено {type(series).__name__!r}'
    )

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Базовый sklearn-подобный класс для обёрток над моделями.

    После вызова fit() заполняются атрибуты:
        _model           — обученная «сырая» модель (lgb.LGBM*, CatBoost*, ...)
        best_params_     — словарь параметров финальной модели
        selected_features_ — список признаков, использованных при обучении
        cat_features_    — список категориальных признаков
        train_pred_      — предсказания на обучающей выборке (regression: float, cls: proba)
        valid_pred_      — предсказания на валидационной выборке (None если не передана)
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        n_optuna_trials: int = 50,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        """Args:
        params: Гиперпараметры модели. Если None — запускается Optuna.
        n_optuna_trials: Число trials Optuna (игнорируется если params задан).
        model_settings: Доп. настройки: baseline_col, reg_metric, cls_metric и т.п.

        """
        self.params = params
        self.n_optuna_trials = n_optuna_trials
        self.model_settings: dict[str, Any] = model_settings or {}

        self._model: Any = None
        self.best_params_: dict[str, Any] | None = None
        self.selected_features_: list[str] | None = None
        self.cat_features_: list[str] = []
        self.train_pred_: np.ndarray | None = None
        self.valid_pred_: np.ndarray | None = None
        self.calibrator_: Any = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> BaseModel:
        """Обучает модель. Возвращает self для method chaining.

        Args:
            X_train: Обучающая выборка.
            y_train: Целевая переменная.
            X_valid: Валидационная выборка (обязательна при params=None/Optuna).
            y_valid: Целевая переменная валидации.
            selected_features: Признаки для обучения. None → все столбцы X_train.
            cat_features: Категориальные признаки.

        """

    def predict(self, X: Any) -> np.ndarray:
        """Предсказывает значения для X (регрессия). Принимает pandas или polars DataFrame."""
        self._check_fitted()
        return self._predict_impl(_to_pandas(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Предсказывает вероятности класса 1 для X (классификация). Принимает pandas или polars DataFrame."""
        self._check_fitted()
        return self._predict_proba_impl(_to_pandas(X))

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f'{type(self).__name__} не реализует predict()')

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f'{type(self).__name__} не реализует predict_proba()')

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f'{type(self).__name__} не обучена. Вызовите .fit() перед .predict().'
            )

    def _resolve_features(
        self, X: pd.DataFrame, selected_features: list[str] | None
    ) -> list[str]:
        return selected_features if selected_features is not None else list(X.columns)

    @staticmethod
    def _coerce_inputs(
        X_train: Any,
        y_train: Any,
        X_valid: Any | None,
        y_valid: Any | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]:
        """Конвертирует Polars/numpy входы в pandas перед обучением."""
        X_tr = _to_pandas(X_train)
        y_tr = pd.Series(_to_numpy(y_train), name='target')
        X_va = _to_pandas(X_valid) if X_valid is not None else None
        y_va = pd.Series(_to_numpy(y_valid), name='target') if y_valid is not None else None
        return X_tr, y_tr, X_va, y_va
