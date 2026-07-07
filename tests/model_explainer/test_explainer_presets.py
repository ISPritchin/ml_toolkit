"""Проверяет, что ModelExplainer (permutation importance, explain_row) работает на
КАЖДОМ пресете из ml_toolkit.presets.classification.high_pr_auc — не только на
"простых" пресетах с одной сырой моделью внутри, но и на ансамблях/обёртках, где
self._model — sentinel/tuple, а model_name_ не определяется как известный
tree/linear алгоритм. permutation — единственный метод, не зависящий от внутренней
структуры модели (работает через predict_proba/predict как чёрный ящик), поэтому
он должен отрабатывать без исключений для любого BaseModel-совместимого пресета.

multiclass_imbalance/ сюда не входит — там predict_proba возвращает матрицу
(n, n_classes), а ModelExplainer расcчитан на бинарную классификацию (скаляр на
строку в permutation/explain_row) — это отдельная, более крупная задача.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ml_toolkit.model_explainer import ModelExplainer
from ml_toolkit.models import CatBoostClassifier as RawCatBoostClassifier
from ml_toolkit.presets.classification import high_pr_auc as hpa
from tests.model_explainer.conftest import (
    FAST_CATBOOST_PARAMS,
    assert_valid_contribution,
    assert_valid_importance,
)

FAST = FAST_CATBOOST_PARAMS


def _fit(preset, X_train, y_train, X_valid, y_valid):
    preset.fit(X_train, y_train, X_valid, y_valid)
    return preset


def _greedy_ensemble_selection(X_train, y_train, X_valid, y_valid):
    library = []
    for seed in (1, 2):
        m = RawCatBoostClassifier(params={**FAST, 'random_seed': seed})
        m.fit(X_train, y_train, X_valid, y_valid)
        library.append(m)
    preset = hpa.GreedyForwardEnsembleSelection(model_library=library, max_members=2, n_bags=3)
    return _fit(preset, X_train, y_train, X_valid, y_valid)


# {имя класса: factory(X_train, y_train, X_valid, y_valid) -> обученный пресет}
PRESET_FACTORIES: dict[str, Callable[..., Any]] = {
    'BoostedEnsemble': lambda *d: _fit(hpa.BoostedEnsemble(base_params=FAST), *d),
    'PrecisionAtKClassifier': lambda *d: _fit(hpa.PrecisionAtKClassifier(k_fraction=0.2, n_optuna_trials=2), *d),
    'TwoStageCascade': lambda *d: _fit(hpa.TwoStageCascade(stage1_params=FAST, stage2_params=FAST), *d),
    'HardNegativeMiner': lambda *d: _fit(hpa.HardNegativeMiner(n_rounds=1, base_params=FAST), *d),
    'SubsampleStacking': lambda *d: _fit(hpa.SubsampleStacking(
        n_base_models=2, n_folds=2,
        base_configs=[{**FAST}, {**FAST, 'random_seed': 7}],
    ), *d),
    'EasyEnsembleClassifier': lambda *d: _fit(hpa.EasyEnsembleClassifier(
        n_estimators=3, neg_ratio=3, base='catboost', base_params=FAST,
    ), *d),
    'PULearningClassifier': lambda *d: _fit(hpa.PULearningClassifier(base_params=FAST), *d),
    'CalibratedWrapper': lambda *d: _fit(hpa.CalibratedWrapper(
        hpa.HardNegativeMiner(n_rounds=1, base_params=FAST), method='platt',
    ), *d),
    'ThresholdMovingCV': lambda *d: _fit(hpa.ThresholdMovingCV(
        hpa.HardNegativeMiner(n_rounds=1, base_params=FAST), optimize='f1',
    ), *d),
    # sampling_strategy=0.1 (дефолт) предполагает миноритарный класс < 10% —
    # classification_data даёт ~27%, поэтому нужен ratio выше текущего дисбаланса.
    'SyntheticOversamplingClassifier': lambda *d: _fit(hpa.SyntheticOversamplingClassifier(
        base='catboost', base_params=FAST, sampling_strategy=0.9,
    ), *d),
    'LambdaRankClassifier': lambda *d: _fit(hpa.LambdaRankClassifier(base_params=FAST), *d),
    'AsymmetricLossClassifier': lambda *d: _fit(hpa.AsymmetricLossClassifier(base_params=FAST), *d),
    'SelfTrainingBooster': lambda *d: _fit(hpa.SelfTrainingBooster(n_rounds=1, base_params=FAST), *d),
    'AnomalyBlendClassifier': lambda *d: _fit(hpa.AnomalyBlendClassifier(
        n_if_estimators=20, supervised_params=FAST,
    ), *d),
    'FeatureBaggingEnsemble': lambda *d: _fit(hpa.FeatureBaggingEnsemble(n_estimators=3, base_params=FAST), *d),
    'SnapshotEnsembleClassifier': lambda *d: _fit(hpa.SnapshotEnsembleClassifier(base_params=FAST), *d),
    'StabilitySelectionClassifier': lambda *d: _fit(hpa.StabilitySelectionClassifier(
        n_bootstrap=5, top_k=5, bootstrap_params=FAST, final_params=FAST,
    ), *d),
    'FocalLossClassifier': lambda *d: _fit(hpa.FocalLossClassifier(base_params=FAST), *d),
    'TverskyLossClassifier': lambda *d: _fit(hpa.TverskyLossClassifier(base_params=FAST), *d),
    'PolyLossClassifier': lambda *d: _fit(hpa.PolyLossClassifier(base_params=FAST), *d),
    'LDAMClassifier': lambda *d: _fit(hpa.LDAMClassifier(base_params=FAST), *d),
    'GHMLossClassifier': lambda *d: _fit(hpa.GHMLossClassifier(base_params=FAST), *d),
    'InfluenceBalancedLossClassifier': lambda *d: _fit(hpa.InfluenceBalancedLossClassifier(base_params=FAST), *d),
    'DiceLossClassifier': lambda *d: _fit(hpa.DiceLossClassifier(base_params=FAST), *d),
    'AsymmetricPolyLossClassifier': lambda *d: _fit(hpa.AsymmetricPolyLossClassifier(base_params=FAST), *d),
    'ConfidentLearningCleaner': lambda *d: _fit(hpa.ConfidentLearningCleaner(n_folds=2, base_params=FAST), *d),
    'CoTeachingClassifier': lambda *d: _fit(hpa.CoTeachingClassifier(n_rounds=2, base_params=FAST), *d),
    'BaggingPUClassifier': lambda *d: _fit(hpa.BaggingPUClassifier(n_estimators=3, base_params=FAST), *d),
    'SpyPUClassifier': lambda *d: _fit(hpa.SpyPUClassifier(base_params=FAST), *d),
    'ElkanNotoHoldoutPU': lambda *d: _fit(hpa.ElkanNotoHoldoutPU(base_params=FAST, n_bootstrap=10), *d),
    'NNPUClassifier': lambda *d: _fit(hpa.NNPUClassifier(class_prior=0.2, base_params=FAST), *d),
    'HeterogeneousStacking': lambda *d: _fit(hpa.HeterogeneousStacking(
        base_zoo=['catboost', 'lightgbm'], n_folds=2,
    ), *d),
    'MultiSeedBlend': lambda *d: _fit(hpa.MultiSeedBlend(n_seeds=2, base_params=FAST), *d),
    'GreedyForwardEnsembleSelection': _greedy_ensemble_selection,
    'DriftRobustClassifier': lambda *d: _fit(hpa.DriftRobustClassifier(base_params=FAST), *d),
    'AdversarialValidationWeighting': lambda *d: _fit(hpa.AdversarialValidationWeighting(base_params=FAST), *d),
}

# Каждый класс, экспортируемый из high_pr_auc, обязан иметь фабрику — иначе новый
# пресет молча выпадет из проверки "permutation работает для любого пресета".
_MISSING = set(hpa.__all__) - set(PRESET_FACTORIES)
assert not _MISSING, f'Нет фабрики ModelExplainer-теста для пресетов: {sorted(_MISSING)}'


@pytest.mark.parametrize('preset_name', sorted(PRESET_FACTORIES))
def test_permutation_importance_and_explain_row(preset_name, classification_data):
    X_train, y_train, X_valid, y_valid = classification_data
    factory = PRESET_FACTORIES[preset_name]
    preset = factory(X_train, y_train, X_valid, y_valid)

    explainer = ModelExplainer(preset, X_valid, y_valid, task='classification')

    imp = explainer.feature_importance(method='permutation', n_repeats=2)
    assert_valid_importance(imp, explainer.feature_names_)

    contrib = explainer.explain_row(X_valid.iloc[[0]])
    assert_valid_contribution(contrib, explainer.feature_names_)

    # 'auto' не должен требовать SHAP/gain там, где они недоступны — обязан
    # прозрачно упасть на permutation и не бросить исключение.
    imp_auto = explainer.feature_importance(method='auto', n_repeats=2)
    assert_valid_importance(imp_auto, explainer.feature_names_)
