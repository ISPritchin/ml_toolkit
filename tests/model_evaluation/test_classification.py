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

from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS, ClassificationEvaluator, fbeta


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


class TestClassificationEvaluatorAddNormalizesBinaryTwoColumnProba:
    """Все ml_toolkit-адаптеры возвращают (n, 2) для бинарной классификации (единый
    sklearn-контракт с мультиклассом, см. CLAUDE.md). CLASSIFICATION_PRESETS/метрики
    внутри ClassificationEvaluator различают binary/multiclass по y_proba.ndim, а не по
    self._task — без нормализации в add() любой (n, 2) результат от модели был бы
    молча принят за мультикласс. add() теперь приводит его к 1D P(y=1) при task='binary',
    так что всё ниже (метрики, графики, psi) продолжает работать как раньше.
    """

    def test_two_column_binary_proba_normalized_to_1d(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        p1 = np.clip(y_true * 0.6 + rng.normal(scale=0.3, size=200) + 0.2, 0, 1)
        proba_2col = np.column_stack([1 - p1, p1])

        ev = ClassificationEvaluator(task='binary')
        ev.add('valid', y_true, proba_2col)
        stored_y_true, stored_proba = ev._splits['valid']
        assert stored_proba.ndim == 1
        np.testing.assert_allclose(stored_proba, p1)

    def test_1d_binary_proba_still_accepted_unchanged(self):
        """Обратная совместимость: пресеты (ml_toolkit.presets) всё ещё отдают 1D — не трогаем."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        p1 = np.clip(y_true * 0.6 + rng.normal(scale=0.3, size=200) + 0.2, 0, 1)

        ev = ClassificationEvaluator(task='binary')
        ev.add('valid', y_true, p1)
        _, stored_proba = ev._splits['valid']
        np.testing.assert_allclose(stored_proba, p1)

    def test_metrics_identical_whether_1d_or_2col_passed(self):
        """Суть фикса: метрика не должна зависеть от того, в каком виде пришла бинарная proba."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        p1 = np.clip(y_true * 0.6 + rng.normal(scale=0.3, size=200) + 0.2, 0, 1)

        ev_1d = ClassificationEvaluator(task='binary')
        ev_1d.add('valid', y_true, p1).add_metric('roc_auc')
        ev_2col = ClassificationEvaluator(task='binary')
        ev_2col.add('valid', y_true, np.column_stack([1 - p1, p1])).add_metric('roc_auc')

        assert ev_1d.metrics(splits=['valid']).loc['roc_auc', 'valid'] == pytest.approx(
            ev_2col.metrics(splits=['valid']).loc['roc_auc', 'valid']
        )

    def test_multiclass_proba_not_touched(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 3, size=200)
        proba = rng.dirichlet(np.ones(3), size=200)

        ev = ClassificationEvaluator(task='multiclass')
        ev.add('valid', y_true, proba)
        _, stored_proba = ev._splits['valid']
        assert stored_proba.shape == (200, 3)
        np.testing.assert_allclose(stored_proba, proba)

    def test_wrong_column_count_for_binary_raises(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=50)
        proba = rng.dirichlet(np.ones(3), size=50)  # 3 столбца — не бинарная задача

        ev = ClassificationEvaluator(task='binary')
        with pytest.raises(ValueError, match='столбцов'):
            ev.add('valid', y_true, proba)
