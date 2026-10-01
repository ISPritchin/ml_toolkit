"""Тесты для ml_toolkit/model_evaluation/_regression.py.

Не покрывает каждый пресет по отдельности (это предсуществующий, нетронутый
код) — фокус на новой wape().
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.model_evaluation import REGRESSION_PRESETS


class TestWape:
    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 0.0, 5.0, 10.0])
        y_pred = np.array([1.1, 1.8, 0.5, 4.0, 9.0])
        expected = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
        assert REGRESSION_PRESETS['wape'](y_true, y_pred) == pytest.approx(expected)

    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert REGRESSION_PRESETS['wape'](y, y) == 0.0

    def test_all_zero_actuals_returns_zero_not_nan(self):
        """sum|y_true|=0 — деление на ноль; по конвенции (как в denom_floor-паттерне
        проекта) возвращаем 0.0, а не NaN/inf."""
        y_true = np.zeros(5)
        y_pred = np.array([1.0, -1.0, 2.0, 0.0, 0.5])
        assert REGRESSION_PRESETS['wape'](y_true, y_pred) == 0.0

    def test_less_sensitive_to_near_zero_actuals_than_mape(self):
        """WAPE агрегирует Σ|e|/Σ|y| — один y≈0 не взрывает метрику, в отличие от MAPE."""
        y_true = np.array([0.001, 100.0, 100.0, 100.0])
        y_pred = np.array([1.0, 100.0, 100.0, 100.0])
        wape = REGRESSION_PRESETS['wape'](y_true, y_pred)
        mape = REGRESSION_PRESETS['mape'](y_true, y_pred)
        assert wape < mape
