"""Генетический алгоритм отбора признаков.

Экспортирует:
    select_features_genetic  — точка входа GA.
    make_catboost_scorer     — фабрика CatBoost-скорера.
    ScorerFn                 — тип scorer-callable.
"""
from ._core import ScorerFn, make_catboost_scorer, select_features_genetic

__all__ = ["select_features_genetic", "make_catboost_scorer", "ScorerFn"]
