"""Тесты для ml_toolkit/models/_utils.py — общих хелперов калибровки.

fit_multiclass_calibrators/apply_multiclass_calibrators — единственный код,
формирующий multiclass predict_proba и в CatBoostClassifier, и в LAMAClassifier
(оба импортируют эти же функции, см. _tabular/_boosting/_catboost.py и
_tabular/_automl/_lama.py) — поэтому контракт результата (форма (n, K), строки
нормированы к 1, порядок классов = sorted(unique(y))) гарантированно одинаков
у обоих адаптеров «по построению», без необходимости прогонять реальный LAMA
fit() (который в этом окружении не тестируется end-to-end — см. докстринг
test_lama.py про SIGSEGV в multiprocessing LightAutoML на macOS).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.models._utils import apply_multiclass_calibrators, fit_calibrator, fit_multiclass_calibrators


def _make_raw_proba(y: np.ndarray, n_classes: int, bias: float, noise: float, seed: int) -> np.ndarray:
    """Синтетическая 'сырая' матрица вероятностей — не привязана к конкретному адаптеру."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((len(y), n_classes))
    for k in range(n_classes):
        raw[:, k] = (y == k).astype(float) * 0.6 + rng.normal(scale=noise, size=len(y)) + bias
    raw = np.clip(raw, 1e-6, None)
    return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture
def multiclass_labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, size=500)


class TestApplyMulticlassCalibratorsContract:
    def test_output_shape_matches_input(self, multiclass_labels):
        y = multiclass_labels
        raw = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=1)
        cal = fit_multiclass_calibrators(raw, y)
        proba = apply_multiclass_calibrators(raw, cal)
        assert proba.shape == raw.shape

    def test_rows_sum_to_one(self, multiclass_labels):
        y = multiclass_labels
        raw = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=1)
        cal = fit_multiclass_calibrators(raw, y)
        proba = apply_multiclass_calibrators(raw, cal)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_values_in_unit_interval(self, multiclass_labels):
        y = multiclass_labels
        raw = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=1)
        cal = fit_multiclass_calibrators(raw, y)
        proba = apply_multiclass_calibrators(raw, cal)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_n_calibrators_equals_n_classes(self, multiclass_labels):
        y = multiclass_labels
        raw = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=1)
        cal = fit_multiclass_calibrators(raw, y)
        assert len(cal) == 3


class TestSameContractRegardlessOfSourceAdapter:
    """CatBoostClassifier и LAMAClassifier передают свои 'сырые' proba через ЭТИ ЖЕ функции.

    Симулируем два разных источника сырых вероятностей (разный шум/смещение —
    как если бы одна матрица пришла от CatBoost, другая от LightAutoML) и
    проверяем, что после калибровки контракт (форма, нормировка, согласие по
    argmax) идентичен — расхождение возможно только в сырых вероятностях ДО
    калибровки, не в контракте, который формируют общие функции.
    """

    def test_shape_and_normalization_identical_across_sources(self, multiclass_labels):
        y = multiclass_labels
        raw_a = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=1)   # 'catboost-like'
        raw_b = _make_raw_proba(y, 3, bias=0.1, noise=0.05, seed=2)   # 'lama-like'

        proba_a = apply_multiclass_calibrators(raw_a, fit_multiclass_calibrators(raw_a, y))
        proba_b = apply_multiclass_calibrators(raw_b, fit_multiclass_calibrators(raw_b, y))

        assert proba_a.shape == proba_b.shape == (len(y), 3)
        np.testing.assert_allclose(proba_a.sum(axis=1), 1.0)
        np.testing.assert_allclose(proba_b.sum(axis=1), 1.0)
        # Обе калибровки восстанавливают истинный класс на хорошо разделимых данных
        assert (proba_a.argmax(axis=1) == y).mean() > 0.9
        assert (proba_b.argmax(axis=1) == y).mean() > 0.9

    def test_binary_calibrator_contract_is_1d(self, multiclass_labels):
        """Бинарный путь (fit_calibrator) — тот же для CatBoost/LAMA, возвращает 1D P(y=1)."""
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=300)
        raw = np.clip((y.astype(float) * 0.6 + rng.normal(scale=0.05, size=300) + 0.2), 0, 1)
        calibrator = fit_calibrator(raw, y)
        proba = calibrator.predict(raw)
        assert proba.shape == (300,)
        assert np.all((proba >= 0) & (proba <= 1))
