"""Model dispatch layer.

    from ml_toolkit.models import LightGBMClassifier, CatBoostRegressor
    from ml_toolkit.models import RandomForestClassifier, XGBoostRegressor
    from ml_toolkit.models import LAMARegressor, TabMClassifier

    # Без Optuna (явные параметры):
    model = LightGBMClassifier(params={'n_estimators': 500, 'num_leaves': 31})
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_new)

    # С Optuna (params=None, X_valid обязателен):
    model = CatBoostRegressor(n_optuna_trials=50,
                               model_settings={'baseline_col': 'fee_nds_amount'})
    model.fit(X_train, y_train, X_valid, y_valid, selected_features=['a', 'b'])
    pred = model.predict(X_new)
    print(model.best_params_)

Атрибуты после fit():
    model.best_params_        — dict параметров финальной модели
    model.selected_features_  — признаки, которые попали в обучение
    model.train_pred_         — предсказания на train
    model.valid_pred_         — предсказания на valid (None если не передан)

Кастомные метрики (передаются через model_settings):
    reg_metric  — 'mae' (по умолч.) / 'rmse' / 'mape' / 'smape' / callable / (callable, direction)
    cls_metric  — 'pr_auc' (по умолч.) / 'roc_auc' / 'f1' / callable / (callable, direction)

Параметризованные метрики:
    from ml_toolkit.models._utils import make_precision_at_k, make_recall_at_k, make_quantile_loss

Кодирование категорий (model_settings['cat_encoder']):
    None / 'ordinal' → OrdinalEncoder (по умолч.)
    'onehot'         → OneHotEncoder

Lazy imports: класс модели загружается только при первом обращении к нему.
"""

from __future__ import annotations

import importlib
import logging

from ml_toolkit.model_evaluation import (
    ClassificationEvaluator,
    ModelEvaluator,
    RegressionEvaluator,
    f1_at_threshold,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)
from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._catboost_ranker import CatBoostRanker
from ml_toolkit.models._lightgbm import LightGBMClassifier, LightGBMRegressor
from ml_toolkit.models._lightgbm_ranker import LightGBMRanker
from ml_toolkit.models._xgboost_ranker import XGBoostRanker

# Все остальные классы моделей — lazy imports через __getattr__.
# Это позволяет импортировать ml_toolkit.models без установки optuna/torch/catboost и т.д.
# Класс загружается только при первом обращении к нему.
_LAZY_CLASSES: dict[str, tuple[str, str]] = {
    # CatBoost
    'CatBoostRegressor':             ('ml_toolkit.models._catboost',             'CatBoostRegressor'),
    'CatBoostClassifier':            ('ml_toolkit.models._catboost',             'CatBoostClassifier'),
    # XGBoost
    'XGBoostRegressor':              ('ml_toolkit.models._xgboost',              'XGBoostRegressor'),
    'XGBoostClassifier':             ('ml_toolkit.models._xgboost',              'XGBoostClassifier'),
    # sklearn Trees
    'DecisionTreeRegressor':         ('ml_toolkit.models._decision_tree',        'DecisionTreeRegressor'),
    'DecisionTreeClassifier':        ('ml_toolkit.models._decision_tree',        'DecisionTreeClassifier'),
    'RandomForestRegressor':         ('ml_toolkit.models._forest',               'RandomForestRegressor'),
    'RandomForestClassifier':        ('ml_toolkit.models._forest',               'RandomForestClassifier'),
    'ExtraTreesRegressor':           ('ml_toolkit.models._forest',               'ExtraTreesRegressor'),
    'ExtraTreesClassifier':          ('ml_toolkit.models._forest',               'ExtraTreesClassifier'),
    'HistGBMRegressor':              ('ml_toolkit.models._hist_gbm',             'HistGBMRegressor'),
    'HistGBMClassifier':             ('ml_toolkit.models._hist_gbm',             'HistGBMClassifier'),
    # Специализированные леса
    'QuantileForestRegressor':       ('ml_toolkit.models._quantile_forest',      'QuantileForestRegressor'),
    'QuantileForestClassifier':      ('ml_toolkit.models._quantile_forest',      'QuantileForestClassifier'),
    'ObliqueForestRegressor':        ('ml_toolkit.models._oblique_forest',       'ObliqueForestRegressor'),
    'ObliqueForestClassifier':       ('ml_toolkit.models._oblique_forest',       'ObliqueForestClassifier'),
    'MondrianForestRegressor':       ('ml_toolkit.models._mondrian',             'MondrianForestRegressor'),
    'MondrianForestClassifier':      ('ml_toolkit.models._mondrian',             'MondrianForestClassifier'),
    # Линейные
    'LinearRegressor':               ('ml_toolkit.models._linear',               'LinearRegressor'),
    'LinearClassifier':              ('ml_toolkit.models._linear',               'LinearClassifier'),
    # EBM / GAM
    'EBMRegressor':                  ('ml_toolkit.models._ebm',                  'EBMRegressor'),
    'EBMClassifier':                 ('ml_toolkit.models._ebm',                  'EBMClassifier'),
    'PyGAMRegressor':                ('ml_toolkit.models._gam',                  'PyGAMRegressor'),
    'PyGAMClassifier':               ('ml_toolkit.models._gam',                  'PyGAMClassifier'),
    'MARSRegressor':                 ('ml_toolkit.models._mars',                 'MARSRegressor'),
    'MARSClassifier':                ('ml_toolkit.models._mars',                 'MARSClassifier'),
    # Rule-based
    'RuleFitRegressor':              ('ml_toolkit.models._rulefit',              'RuleFitRegressor'),
    'RuleFitClassifier':             ('ml_toolkit.models._rulefit',              'RuleFitClassifier'),
    'IModelsRegressor':              ('ml_toolkit.models._imodels',              'IModelsRegressor'),
    'IModelsClassifier':             ('ml_toolkit.models._imodels',              'IModelsClassifier'),
    # Интерпретируемые деревья
    'LinearTreeRegressor':           ('ml_toolkit.models._linear_tree',          'LinearTreeRegressor'),
    'LinearTreeClassifier':          ('ml_toolkit.models._linear_tree',          'LinearTreeClassifier'),
    'InterpretableTreeRegressor':    ('ml_toolkit.models._interpretable_trees',  'InterpretableTreeRegressor'),
    'InterpretableTreeClassifier':   ('ml_toolkit.models._interpretable_trees',  'InterpretableTreeClassifier'),
    # Интерпретируемые нейронные
    'InterpretableNeuralRegressor':  ('ml_toolkit.models._interpretable_neural', 'InterpretableNeuralRegressor'),
    'InterpretableNeuralClassifier': ('ml_toolkit.models._interpretable_neural', 'InterpretableNeuralClassifier'),
    # AutoML
    'LAMARegressor':                 ('ml_toolkit.models._lama',                 'LAMARegressor'),
    'LAMAClassifier':                ('ml_toolkit.models._lama',                 'LAMAClassifier'),
    'TabMRegressor':                 ('ml_toolkit.models._tabm',                 'TabMRegressor'),
    'TabMClassifier':                ('ml_toolkit.models._tabm',                 'TabMClassifier'),
}


def __getattr__(name: str) -> type:
    if name in _LAZY_CLASSES:
        module_path, cls_name = _LAZY_CLASSES[name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        # Кешируем в namespace модуля, чтобы последующие обращения не вызывали __getattr__
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'ml_toolkit.models' has no attribute {name!r}")

logger = logging.getLogger(__name__)

__all__ = [
    # Базовый класс
    'BaseModel',
    # Gradient Boosting
    'LightGBMRegressor', 'LightGBMClassifier',
    'CatBoostRegressor', 'CatBoostClassifier',
    'XGBoostRegressor', 'XGBoostClassifier',
    # sklearn Trees & Forests
    'DecisionTreeRegressor', 'DecisionTreeClassifier',
    'RandomForestRegressor', 'RandomForestClassifier',
    'ExtraTreesRegressor', 'ExtraTreesClassifier',
    'HistGBMRegressor', 'HistGBMClassifier',
    # Specialized Forests (optional deps)
    'QuantileForestRegressor', 'QuantileForestClassifier',
    'ObliqueForestRegressor', 'ObliqueForestClassifier',
    'MondrianForestRegressor', 'MondrianForestClassifier',
    # Linear
    'LinearRegressor', 'LinearClassifier',
    # GAM / Additive
    'EBMRegressor', 'EBMClassifier',
    'PyGAMRegressor', 'PyGAMClassifier',
    'MARSRegressor', 'MARSClassifier',
    # Rule-based
    'RuleFitRegressor', 'RuleFitClassifier',
    'IModelsRegressor', 'IModelsClassifier',
    # Interpretable Trees
    'LinearTreeRegressor', 'LinearTreeClassifier',
    'InterpretableTreeRegressor', 'InterpretableTreeClassifier',
    # Interpretable Neural
    'InterpretableNeuralRegressor', 'InterpretableNeuralClassifier',
    # AutoML
    'LAMARegressor', 'LAMAClassifier',
    'TabMRegressor', 'TabMClassifier',
    # Ранжировщики
    'LightGBMRanker',
    'XGBoostRanker',
    'CatBoostRanker',
    # Оценка и визуализация
    'ClassificationEvaluator',
    'RegressionEvaluator',
    'ModelEvaluator',
    'precision_at_k',
    'recall_at_k',
    'lift_at_k',
    'f1_at_threshold',
]

# Линейные модели: все реализованы в одном модуле _linear.py
LINEAR_NAMES: frozenset[str] = frozenset({
    'ridge', 'elasticnet', 'huber', 'tweedie', 'quantile', 'bayesian_ridge',
})

# Древесные ансамбли (не бустинг)
FOREST_NAMES: frozenset[str] = frozenset({
    'random_forest', 'extra_trees', 'hist_gbm',
    'quantile_forest', 'oblique_forest', 'mondrian',
})

# GAM / аддитивные модели
GAM_NAMES: frozenset[str] = frozenset({'ebm', 'pygam', 'mars'})

# Rule-based интерпретируемые модели (imodels)
IMODELS_NAMES: frozenset[str] = frozenset({'figs', 'skope_rules', 'brl', 'ripper'})

# Деревья с линейными моделями и мягкими разбиениями
INTERPRETABLE_TREE_NAMES: frozenset[str] = frozenset({'soft_decision_tree', 'locally_linear_forest'})

# Нейросетевые интерпретируемые модели
INTERPRETABLE_NEURAL_NAMES: frozenset[str] = frozenset({'gaminet'})

# Ранжировщики (gradient boosting с ranking objectives)
RANKER_NAMES: frozenset[str] = frozenset({'lightgbm_ranker', 'xgboost_ranker', 'catboost_ranker'})

# LightGBM — один адаптер; boosting_type (gbdt/dart/goss) выбирается Optuna
LIGHTGBM_VARIANTS: frozenset[str] = frozenset({'lightgbm'})

# Деревья с линейными моделями в листьях
LINEAR_TREE_NAMES: frozenset[str] = frozenset({'linear_tree'})

# sklearn-деревья, у которых feature_importances_ доступна через Pipeline.named_steps
SKLEARN_TREE_NAMES: frozenset[str] = (FOREST_NAMES - {'mondrian'}) | {'decision_tree'}

# Все tree-based модели (используется в model_explainer/feature_importance.py для SHAP/gain)
ALL_TREE_NAMES: frozenset[str] = (
    frozenset({'catboost', 'xgboost'}) | LIGHTGBM_VARIANTS | SKLEARN_TREE_NAMES | RANKER_NAMES
)

