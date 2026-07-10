"""Общие утилиты для time-aware пресетов classification (WeightedBaggingByRecency,
TimeAwareValidationClassifier) — перевод ts_key в целочисленные периоды.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_periods(ts_key: pd.Series, period_unit: str) -> np.ndarray:
    """Переводит ts_key в целочисленные периоды (для агрегации по давности/окнам).

    datetime-подобный ts_key бинуется в периоды через `period_unit` (pandas
    frequency alias, например 'M' для месяца); уже числовой ts_key (например,
    номер месяца/квартала) используется как есть.
    """
    if pd.api.types.is_datetime64_any_dtype(ts_key):
        return ts_key.dt.to_period(period_unit).astype('int64').to_numpy(dtype=np.float64)
    return pd.to_numeric(ts_key).to_numpy(dtype=np.float64)
