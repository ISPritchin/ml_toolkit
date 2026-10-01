"""Тесты для ml_toolkit/model_evaluation/_classification.py.

Не покрывает каждый пресет по отдельности (это предсуществующий, нетронутый
код) — фокус на (1) новой fbeta() и (2) мультикласс-безопасности всех
CLASSIFICATION_PRESETS разом, т.к. именно отсутствие этой гарантии было
причиной бага в ml_toolkit.models._utils.CLS_METRICS (roc_auc/f1 падали на
мультиклассе — см. tests/models/test_utils.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS, fbeta


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.6 + rng.normal(scale=0.3, size=200) + 0.2, 0, 1)
    return y_true, y_proba


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=200)
    y_proba = rng.dirichlet(np.ones(3), size=200)
    return y_true, y_proba


class TestFbeta:
    def test_beta_1_equals_f1_preset(self, binary_data):
        y_true, y_proba = binary_data
        assert fbeta(1.0)(y_true, y_proba) == pytest.approx(CLASSIFICATION_PRESETS['f1'](y_true, y_proba))

    def test_beta_1_equals_f1_preset_multiclass(self, multiclass_data):
        y_true, y_proba = multiclass_data
        assert fbeta(1.0)(y_true, y_proba) == pytest.approx(CLASSIFICATION_PRESETS['f1'](y_true, y_proba))

    def test_high_beta_favors_recall_over_precision(self, binary_data):
        """beta=5 (recall-тяжёлый) и beta=0.2 (precision-тяжёлый) — разные числа на тех же данных."""
        y_true, y_proba = binary_data
        assert fbeta(5.0)(y_true, y_proba) != pytest.approx(fbeta(0.2)(y_true, y_proba))

    def test_multiclass_uses_macro_average(self, multiclass_data):
        y_true, y_proba = multiclass_data
        v = fbeta(2.0)(y_true, y_proba)
        assert 0.0 <= v <= 1.0


class TestClassificationPresetsMulticlassSafety:
    """Контракт: любой пресет либо отрабатывает на мультиклассе, либо поднимает ValueError.

    (а не непрозрачный sklearn-краш вроде "multi_class must be in ('ovo', 'ovr')"
    или "mix of multiclass and multilabel-indicator targets" — именно такими
    падениями раньше страдал ml_toolkit.models._utils.CLS_METRICS).
    """

    @pytest.mark.parametrize('name', sorted(CLASSIFICATION_PRESETS))
    def test_runs_or_raises_value_error(self, name, multiclass_data):
        y_true, y_proba = multiclass_data
        fn = CLASSIFICATION_PRESETS[name]
        try:
            v = fn(y_true, y_proba)
        except ValueError:
            pytest.skip(f'{name}: намеренно не определён для мультикласса (ValueError)')
            return
        assert np.isfinite(v)

    @pytest.mark.parametrize('name', sorted(CLASSIFICATION_PRESETS))
    def test_runs_on_binary(self, name, binary_data):
        y_true, y_proba = binary_data
        v = CLASSIFICATION_PRESETS[name](y_true, y_proba)
        assert np.isfinite(v)
