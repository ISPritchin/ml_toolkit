"""Пакет отбора признаков.

Всегда доступно (без тяжёлых зависимостей):
    FeatureScreener              — многоступенчатый статистический пре-фильтр (polars).
    FeatureSelectionPipeline     — единый пайплайн: скрининг + AUC + drift (pandas).
    AdversarialDriftFilter       — adversarial validation drift-фильтр (pandas).
    compute_psi                  — Population Stability Index по признакам.

Discover-on-sample -> select -> artifact -> replay-on-full-data (см. discovery.py):
    FeatureSelectionArtifact               — persisted (product_cols, preset, accepted_cols).
    run_feature_discovery                  — сэмпл -> кандидаты -> отбор -> артефакт, одним вызовом.
    sample_entities                        — сэмплирование сущностей (со стратификацией по label).
    select_features_by_model_feedback      — скрининг + CatBoost importance.
    materialize_feature_selection_artifact — реплей артефакта на полном датасете.
    merge_feature_selection_artifacts      — объединение нескольких артефактов (union accepted_cols).

Доступно при наличии deap (catboost нужен только внутри make_catboost_scorer):
    select_features_genetic — генетический алгоритм отбора; принимает произвольный scorer.
    make_catboost_scorer    — фабрика CatBoost-скорера для select_features_genetic.
"""
from .discovery import (
    FeatureSelectionArtifact,
    materialize_feature_selection_artifact,
    merge_feature_selection_artifacts,
    run_feature_discovery,
    sample_entities,
    select_features_by_model_feedback,
)
from .drift_filter import AdversarialDriftFilter, compute_psi
from .pipeline import FeatureSelectionPipeline
from .screening import FeatureScreener

__all__ = [
    'AdversarialDriftFilter',
    'FeatureScreener',
    'FeatureSelectionArtifact',
    'FeatureSelectionPipeline',
    'compute_psi',
    'make_catboost_scorer',
    'materialize_feature_selection_artifact',
    'merge_feature_selection_artifacts',
    'run_feature_discovery',
    'sample_entities',
    'select_features_by_model_feedback',
    'select_features_genetic',
]

_LAZY = {'select_features_genetic', 'make_catboost_scorer'}


def __getattr__(name: str):
    if name in _LAZY:
        from . import genetic
        obj = getattr(genetic, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
