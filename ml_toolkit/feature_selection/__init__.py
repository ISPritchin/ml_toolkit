"""Пакет отбора признаков.

Всегда доступно (без тяжёлых зависимостей):
    FeatureScreener              — многоступенчатый статистический пре-фильтр (polars).
    FeatureSelectionPipeline     — единый пайплайн: скрининг + AUC + drift (pandas).
    AdversarialDriftFilter       — adversarial validation drift-фильтр (pandas).
    compute_psi                  — Population Stability Index по признакам.

Доступно при наличии deap (catboost нужен только внутри make_catboost_scorer):
    select_features_genetic — генетический алгоритм отбора; принимает произвольный scorer.
    make_catboost_scorer    — фабрика CatBoost-скорера для select_features_genetic.
"""
from .screening import FeatureScreener
from .pipeline import FeatureSelectionPipeline
from .drift_filter import AdversarialDriftFilter, compute_psi

__all__ = [
    "FeatureScreener",
    "FeatureSelectionPipeline",
    "AdversarialDriftFilter",
    "compute_psi",
    "select_features_genetic",
    "make_catboost_scorer",
]

_LAZY = {"select_features_genetic", "make_catboost_scorer"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import genetic
        obj = getattr(genetic, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
