"""Базовый класс для пресетов регрессии.

В отличие от ml_toolkit.presets.classification._base.BasePreset, predict() не
переопределяется — BaseModel.predict() уже вызывает _predict_impl() напрямую и
возвращает непрерывные значения, что и требуется регрессии. save/load — тоже
общая реализация BaseModel (см. её докстринг), подклассы наследуют без
переопределения; здесь остаётся только _check_fitted.
"""

from __future__ import annotations

from ml_toolkit.models._base import BaseModel


class BasePreset(BaseModel):
    """BaseModel с сообщением об необученности, специфичным для пресетов."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f'{type(self).__name__} не обучена — вызовите .fit() первым.'
            )
