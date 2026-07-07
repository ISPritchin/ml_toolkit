"""Общие fixtures и helpers для тестов трансформеров."""

import numpy as np

from ml_toolkit.transformers import TRANSFORMERS
from ml_toolkit.transformers._windowing import compute_position_within_entity


def compute_entity_positions(n: int) -> np.ndarray:
    """Вычисляет позиции сущностей для n строк (одна сущность, отсортирована).

    Args:
        n: Количество строк.

    Returns:
        np.ndarray int64 с позициями [0, 1, 2, ..., n-1] для одной сущности.

    """
    return compute_position_within_entity(np.zeros(n, dtype=np.int64))


def run_transformer(
    transformer_name: str,
    values: list | np.ndarray,
    params: dict | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Запускает трансформер и возвращает (arrays, suffixes).

    Args:
        transformer_name: Ключ в TRANSFORMERS.
        values: Значения для обработки.
        params: Параметры трансформера (окна, лаги и т.п.). {} если None.

    Returns:
        Кортеж (arrays, suffixes) как если бы вызвали
        mod.compute(values, position, params).

    """
    mod = TRANSFORMERS[transformer_name]
    pos = compute_entity_positions(len(values))
    return mod.compute(np.array(values, dtype=np.float64), pos, params or {})


def get_feature_output(
    arrays: list[np.ndarray],
    suffixes: list[str],
    suffix: str,
) -> np.ndarray:
    """Извлекает массив фичи по её суффиксу.

    Args:
        arrays: Список массивов, возвращённых compute().
        suffixes: Список суффиксов, соответствующих arrays.
        suffix: Искомый суффикс (пустая строка для безсуффиксной фичи).

    Returns:
        Массив для данного суффикса.

    Raises:
        ValueError: Если суффикс не найден в suffixes.

    """
    try:
        idx = suffixes.index(suffix)
        return arrays[idx]
    except ValueError:
        raise ValueError(
            f"Суффикс '{suffix}' не найден. Доступные: {suffixes}"
        )


# Экспортируем для удобства импорта в тестах
__all__ = ['compute_entity_positions', 'get_feature_output', 'run_transformer']
