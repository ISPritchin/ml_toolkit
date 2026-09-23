"""Базовый класс для пресетов классификации.

Подклассы наследуют _coerce_inputs, _resolve_features, тонкий predict() и
save/load (общая реализация BaseModel — см. её докстринг).
"""

from __future__ import annotations

import numpy as np

from ml_toolkit.models._base import BaseModel, XInput


class BasePreset(BaseModel):
    """BaseModel с predict() через порог вместо _predict_impl."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def predict(self, X: XInput, threshold: float = 0.5) -> np.ndarray:  # type: ignore[override]
        """Бинарная классификация по порогу вероятности."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                f'{type(self).__name__} не обучена — вызовите .fit() первым.'
            )
