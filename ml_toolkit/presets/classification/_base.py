"""Базовый класс для пресетов классификации.

Подклассы наследуют _coerce_inputs, _resolve_features и тонкий predict().
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from ml_toolkit.models._base import BaseModel


class BasePreset(BaseModel):
    """BaseModel с predict() через порог вместо _predict_impl."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:  # type: ignore[override]
        """Бинарная классификация по порогу вероятности."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f'{type(self).__name__} не обучена — вызовите .fit() первым.'
            )

    def save(self, path: str | Path) -> None:
        """Сериализует весь объект пресета (включая обученные подмодели) через pickle.

        Обученные пресеты держат произвольное число подмоделей в нестандартных
        местах (model1_/model2_ каскада, models_ ансамбля, meta_model_ стекинга,
        ...) — pickle всего `self` сохраняет их разом, без знания о внутренней
        структуре конкретного подкласса.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> BasePreset:
        """Загружает пресет, сохранённый через .save()."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f'Файл {path!r} содержит {type(obj).__name__}, ожидался {cls.__name__}.'
            )
        return obj
