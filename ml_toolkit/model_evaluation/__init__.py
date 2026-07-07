"""Model evaluation package.

Classification::

    from ml_toolkit.model_evaluation import ClassificationEvaluator, precision_at_k, lift_at_k

    ev = ClassificationEvaluator(task='binary')
    ev.add('valid', y_true, y_proba).add('test', y_true_t, y_proba_t)
    ev.add_default_metrics()
    ev.add_metric(precision_at_k(0.10), name='precision@10%')
    ev.add_metric(lift_at_k(0.10),      name='lift@10%')

    print(ev.metrics())
    ev.plot_roc(splits=['valid', 'test'])
    ev.report('cls_report.html')

Regression::

    from ml_toolkit.model_evaluation import RegressionEvaluator

    ev = RegressionEvaluator()
    ev.add('valid', y_true, y_pred).add('test', y_true_t, y_pred_t)
    ev.add_default_metrics()

    print(ev.metrics())
    ev.plot_actual_vs_predicted()
    ev.report('reg_report.html')

Available classification presets (add_metric str shorthand):
    roc_auc, pr_auc, log_loss, brier, ks, gini, mcc, ece,
    accuracy, balanced_accuracy, f1, precision, recall, cohen_kappa

Available regression presets:
    mae, mse, rmse, mape, smape, r2, medae, max_error
"""

from ._classification import (
    CLASSIFICATION_PRESETS,
    ClassificationEvaluator,
    ModelEvaluator,  # backward-compatible alias
    f1_at_threshold,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)
from ._comparison import (
    compare_models,
    plot_model_comparison,
    plot_model_delta,
    plot_model_heatmap,
)
from ._error_analysis import ErrorAnalyzer
from ._regression import REGRESSION_PRESETS, RegressionEvaluator

__all__ = [
    # Evaluators
    'ClassificationEvaluator',
    'RegressionEvaluator',
    'ModelEvaluator',          # alias
    # Error analysis
    'ErrorAnalyzer',
    # Factory functions (classification)
    'precision_at_k',
    'recall_at_k',
    'lift_at_k',
    'f1_at_threshold',
    # Preset dicts (for introspection)
    'CLASSIFICATION_PRESETS',
    'REGRESSION_PRESETS',
    # Multi-model comparison
    'compare_models',
    'plot_model_comparison',
    'plot_model_heatmap',
    'plot_model_delta',
]
