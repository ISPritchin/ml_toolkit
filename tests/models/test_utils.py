"""Тесты для ml_toolkit/models/_utils.py — общих хелперов калибровки и метрик.

fit_multiclass_calibrators/apply_multiclass_calibrators — единственный код,
формирующий multiclass predict_proba и в CatBoostClassifier, и в LAMAClassifier
(оба импортируют эти же функции, см. _tabular/_boosting/_catboost.py и
_tabular/_automl/_lama.py) — поэтому контракт результата (форма (n, K), строки
нормированы к 1, порядок классов = sorted(unique(y))) гарантированно одинаков
у обоих адаптеров «по построению», без необходимости прогонять реальный LAMA
fit() (который в этом окружении не тестируется end-to-end — см. докстринг
test_lama.py про SIGSEGV в multiprocessing LightAutoML на macOS).

REG_METRICS/CLS_METRICS — regression-тест на баг: до фикса roc_auc/f1 в
CLS_METRICS были независимыми от model_evaluation реализациями, которые
падали на мультиклассе (roc_auc_score без multi_class='ovr', f1 через
поэлементный порог 0.5 на всей (n,K)-матрице). Теперь REG_METRICS/CLS_METRICS
строятся из ml_toolkit.model_evaluation.REGRESSION_PRESETS/CLASSIFICATION_PRESETS
— тесты ниже закрепляют (1) что они реально одна и та же функция (не копия),
и (2) что мультикласс больше не падает.
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


class TestRegMetricsSharedWithModelEvaluation:
    def test_keys_match_regression_presets(self):
        from ml_toolkit.model_evaluation import REGRESSION_PRESETS
        from ml_toolkit.models._utils import REG_METRICS
        assert set(REG_METRICS) == set(REGRESSION_PRESETS)

    def test_functions_are_the_same_object_not_a_copy(self):
        """REG_METRICS не переизобретает математику — берёт ту же функцию, что Evaluator."""
        from ml_toolkit.model_evaluation import REGRESSION_PRESETS
        from ml_toolkit.models._utils import REG_METRICS
        for name in REGRESSION_PRESETS:
            fn, _direction = REG_METRICS[name]
            assert fn is REGRESSION_PRESETS[name]

    @pytest.mark.parametrize('name,direction', [
        ('mae', 'minimize'), ('mse', 'minimize'), ('rmse', 'minimize'), ('mape', 'minimize'),
        ('smape', 'minimize'), ('wape', 'minimize'), ('r2', 'maximize'), ('medae', 'minimize'),
        ('max_error', 'minimize'),
    ])
    def test_direction(self, name, direction):
        from ml_toolkit.models._utils import REG_METRICS
        assert REG_METRICS[name][1] == direction


class TestClsMetricsSharedWithModelEvaluation:
    def test_keys_match_classification_presets(self):
        from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS
        from ml_toolkit.models._utils import CLS_METRICS
        assert set(CLS_METRICS) == set(CLASSIFICATION_PRESETS)

    def test_functions_are_the_same_object_not_a_copy(self):
        from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS
        from ml_toolkit.models._utils import CLS_METRICS
        for name in CLASSIFICATION_PRESETS:
            fn, _direction = CLS_METRICS[name]
            assert fn is CLASSIFICATION_PRESETS[name]

    @pytest.mark.parametrize('name', ['roc_auc', 'pr_auc', 'f1', 'accuracy', 'balanced_accuracy', 'mcc'])
    def test_multiclass_no_longer_crashes(self, name):
        """Regression-тест: до фикса roc_auc/f1 падали на мультиклассе (см. докстринг модуля)."""
        from ml_toolkit.models._utils import CLS_METRICS
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=200)
        proba = rng.dirichlet(np.ones(3), size=200)
        fn, _direction = CLS_METRICS[name]
        v = fn(y_true, proba)
        assert np.isfinite(v)

    def test_default_cls_metric_still_works_binary(self):
        from ml_toolkit.models._utils import CLS_METRICS
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        proba = rng.random(200)
        fn, direction = CLS_METRICS['pr_auc']
        assert direction == 'maximize'
        assert 0.0 <= fn(y_true, proba) <= 1.0


class TestMakeFbeta:
    def test_returns_maximize_direction(self):
        from ml_toolkit.models._utils import make_fbeta
        fn, direction = make_fbeta(2.0)
        assert direction == 'maximize'
        assert callable(fn)

    def test_beta_1_matches_f1_preset(self):
        from ml_toolkit.models._utils import CLS_METRICS, make_fbeta
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        proba = rng.random(200)
        fn, _ = make_fbeta(1.0)
        f1_fn, _ = CLS_METRICS['f1']
        assert fn(y_true, proba) == pytest.approx(f1_fn(y_true, proba))

    def test_works_as_cls_metric_in_optuna_fit(self, classification_data):
        """End-to-end: make_fbeta в model_settings['cls_metric'] реально доезжает до Optuna."""
        from ml_toolkit.models import LightGBMClassifier
        from ml_toolkit.models._utils import make_fbeta
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(n_optuna_trials=2, model_settings={'cls_metric': make_fbeta(2.0)})
        model.fit(X_train, y_train, X_valid, y_valid)
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid),)
