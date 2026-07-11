"""Тесты для KnowledgeDistillationPreset (ml_toolkit/presets/classification/high_pr_auc/knowledge_distillation.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_toolkit.presets.classification.high_pr_auc import (
    EasyEnsembleClassifier,
    KnowledgeDistillationPreset,
    MultiSeedBlend,
)
from ml_toolkit.presets.classification.high_pr_auc.knowledge_distillation import _soften
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


def _teacher():
    return EasyEnsembleClassifier(n_estimators=3, neg_ratio=3, base='catboost', base_params=BASE_PARAMS)


class TestKnowledgeDistillationPreset:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = KnowledgeDistillationPreset(teacher_preset=_teacher(), student_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.teacher_ is not None
        assert 0.0 <= model.teacher_score_ <= 1.0
        assert 0.0 <= model.student_score_ <= 1.0

    def test_student_is_single_small_model_not_ensemble(self, binary_data):
        """Ключевое свойство дистилляции: студент — одна CatBoost-модель, а не весь ансамбль учителя.

        Иначе теряется весь смысл — "один быстрый скорер".
        """
        X_train, y_train, X_valid, y_valid = binary_data
        model = KnowledgeDistillationPreset(teacher_preset=_teacher(), student_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        from catboost import CatBoostClassifier
        assert isinstance(model._model, CatBoostClassifier)
        assert len(model.teacher_.estimators_) == 3  # учитель внутри остался ансамблем

    def test_with_multiseedblend_teacher(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        teacher = MultiSeedBlend(n_seeds=2, base_params=BASE_PARAMS)
        model = KnowledgeDistillationPreset(teacher_preset=teacher, student_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_optuna_tuning(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = KnowledgeDistillationPreset(teacher_preset=_teacher(), n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'student_params' in model.best_params_

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match='temperature'):
            KnowledgeDistillationPreset(teacher_preset=_teacher(), temperature=0.0)


class TestSoften:
    def test_identity_at_temperature_one(self):
        p = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(_soften(p, 1.0), p, atol=1e-6)

    def test_pulls_toward_half_when_temperature_above_one(self):
        p = np.array([0.05, 0.95])
        soft = _soften(p, temperature=3.0)
        assert soft[0] > p[0]
        assert soft[1] < p[1]

    def test_preserves_order(self):
        p = np.array([0.1, 0.3, 0.6, 0.9])
        soft = _soften(p, temperature=2.5)
        assert np.all(np.diff(soft) > 0)

    def test_handles_extreme_probabilities(self):
        p = np.array([0.0, 1.0])
        soft = _soften(p, temperature=2.0)
        assert np.all(np.isfinite(soft))
        assert 0.0 <= soft[0] <= 1.0
        assert 0.0 <= soft[1] <= 1.0
