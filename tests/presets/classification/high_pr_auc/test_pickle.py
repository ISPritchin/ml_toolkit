"""Pickle-контракт BasePreset: fit() -> save() -> load() -> predict_proba тот же.

save()/load() — общая реализация BaseModel, наследуемая без переопределения каждым
пресетом (см. tests/models/test_pickle.py для полного описания контракта). Пакет
high_pr_auc содержит 40 классов — здесь покрыта репрезентативная выборка (13),
затрагивающая каждый архитектурно разный паттерн хранения состояния: множественные
подмодели (ensemble/stacking/bagging), каскад из двух стадий, обёртки над другим
пресетом (calibration/distillation), PU-learning, time-aware веса. Оставшиеся ~27 —
однотипные CatBoost+custom-loss пресеты (focal/ghm/ldam/tversky/poly/dice/...),
структурно идентичные уже проверенным в tests/presets/regression/test_pickle.py —
единственная найденная там реальная поломка (лямбда как атрибут экземпляра, в
AsymmetricCostRegressor) статически исключена по всему ml_toolkit.presets (grep
на `self\\.\\w+\\s*=\\s*lambda` / `self\\._loss_spec\\s*=` не находит других
вхождений после фикса).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_toolkit.presets.classification.high_pr_auc import (
    BaggingPUClassifier,
    CalibratedWrapper,
    CoTeachingClassifier,
    EasyEnsembleClassifier,
    FeatureBaggingEnsemble,
    HeterogeneousStacking,
    KnowledgeDistillationPreset,
    MultiSeedBlend,
    SelfTrainingBooster,
    SnapshotEnsembleClassifier,
    SubsampleStacking,
    TwoStageCascade,
    WeightedBaggingByRecency,
)
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS


def _assert_proba_pickle_roundtrip(model, X_valid, tmp_path) -> None:
    pred_before = model.predict_proba(X_valid)
    path = tmp_path / 'model.pkl'
    model.save(path)
    loaded = type(model).load(path)
    assert isinstance(loaded, type(model))
    pred_after = loaded.predict_proba(X_valid)
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-9, atol=1e-12)


class TestHighPrAucPresetsPickle:
    def test_two_stage_cascade(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = TwoStageCascade(recall_target=0.80, stage1_n_trials=2, stage2_n_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_subsample_stacking(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SubsampleStacking(n_base_models=2, n_folds=3, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_heterogeneous_stacking(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'logistic'], n_folds=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_easy_ensemble(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = EasyEnsembleClassifier(n_estimators=5, neg_ratio=5, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_self_training_booster(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SelfTrainingBooster(n_rounds=2, pseudo_weight=0.3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_co_teaching(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = CoTeachingClassifier(n_rounds=2, forget_rate=0.3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_knowledge_distillation(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        teacher = EasyEnsembleClassifier(n_estimators=3, neg_ratio=3, base='catboost', base_params=BASE_PARAMS)
        model = KnowledgeDistillationPreset(teacher_preset=teacher, student_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_calibrated_wrapper(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        base = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, base_params=BASE_PARAMS)
        model = CalibratedWrapper(base, method='isotonic')
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_bagging_pu(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BaggingPUClassifier(n_estimators=10, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_multi_seed_blend(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MultiSeedBlend(n_seeds=4, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_snapshot_ensemble(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SnapshotEnsembleClassifier(snapshot_fracs=[0.5, 1.0], base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_feature_bagging(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        model = FeatureBaggingEnsemble(n_estimators=4, feature_frac=0.6, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)

    def test_weighted_bagging_by_recency(self, binary_data, tmp_path):
        X_train, y_train, X_valid, y_valid = binary_data
        dates = pd.date_range('2025-01-01', periods=12, freq='MS')
        ts_key = pd.Series(dates[np.arange(len(X_train)) % 12])
        model = WeightedBaggingByRecency(n_estimators=5, halflife_periods=3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid, ts_key=ts_key)
        _assert_proba_pickle_roundtrip(model, X_valid, tmp_path)
