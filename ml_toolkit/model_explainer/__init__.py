"""Объяснение и интерпретация обученных моделей.

Основной API — класс ModelExplainer:
    from ml_toolkit.model_explainer import ModelExplainer

    explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
    imp  = explainer.feature_importance()        # pd.Series
    sv   = explainer.shap_values()               # np.ndarray (tree-модели)
    exp  = explainer.explain_row(X_row)          # pd.Series, вклад признаков
    fig  = explainer.plot_importance()
    fig  = explainer.plot_shap_beeswarm()
    fig  = explainer.plot_shap_waterfall()
    fig  = explainer.plot_partial_dependence()
    explainer.plot_intrinsic()                   # EBM/GAM/правила/tree
    paths = explainer.report('out/', prefix='lgbm_cls')

Низкоуровневые функции (backward-compat):
    from ml_toolkit.model_explainer import plot_feature_importance, plot_shap_individuals
    from ml_toolkit.model_explainer import plot_interpretable_extra, ALL_INTERPRETABLE
"""

from ml_toolkit.model_explainer.explainer import ModelExplainer
from ml_toolkit.model_explainer.feature_importance import (
    plot_feature_importance,
    plot_shap_individuals,
)
from ml_toolkit.model_explainer.intrinsic_visualization import (
    ALL_INTERPRETABLE,
    plot_interpretable_extra,
)

__all__ = [
    'ModelExplainer',
    'plot_feature_importance',
    'plot_shap_individuals',
    'plot_interpretable_extra',
    'ALL_INTERPRETABLE',
]
