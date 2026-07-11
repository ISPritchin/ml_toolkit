"""Базовый класс для пресетов регрессии.

В отличие от ml_toolkit.presets.classification._base.BasePreset, predict() не
переопределяется — BaseModel.predict() уже вызывает _predict_impl() напрямую и
возвращает непрерывные значения, что и требуется регрессии. Подклассы наследуют
только save/load и _check_fitted.
"""

from __future__ import annotations

from pathlib import Path
import pickle

from ml_toolkit.models._base import BaseModel


class BasePreset(BaseModel):
    """BaseModel с общим save/load для пресетов регрессии."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f'{type(self).__name__} не обучена — вызовите .fit() первым.'
            )

    def save(self, path: str | Path) -> None:
        """Сериализует весь объект пресета (включая обученные подмодели) через pickle.

        Обученные пресеты держат произвольное число подмоделей в нестандартных
        местах (models_ ансамбля, fold-моделей jackknife+ и т.п.) — pickle всего
        `self` сохраняет их разом, без знания о внутренней структуре конкретного
        подкласса.
        """
        with Path(path).open('wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> BasePreset:
        """Загружает пресет, сохранённый через .save()."""
        with Path(path).open('rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f'Файл {path!r} содержит {type(obj).__name__}, ожидался {cls.__name__}.'
            )
        return obj
