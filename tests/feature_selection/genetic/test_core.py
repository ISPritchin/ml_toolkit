"""Тесты `ml_toolkit.feature_selection.genetic._core.make_catboost_scorer`.

Фокус — на classification-ветке: бинарная классификация (обратная совместимость
со старым поведением) и multiclass (новая возможность, реестр метрик общий с
`ml_toolkit.model_evaluation.CLASSIFICATION_PRESETS`). Регрессионная ветка не
трогалась при расширении — покрыта одним sanity-тестом.

CatBoost обучается по-настоящему (не заглушка), но на крошечных синтетических
данных и с минимальным числом итераций, чтобы тесты оставались быстрыми.
"""

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.feature_selection import make_catboost_scorer
from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS

_MODEL_PARAMS = {'iterations': 25, 'verbose': False, 'random_seed': 0}

ALL_CLS_METRIC_NAMES = sorted(CLASSIFICATION_PRESETS)  # включая 'ks'
MULTICLASS_SAFE_METRIC_NAMES = sorted(set(CLASSIFICATION_PRESETS) - {'ks'})  # 'ks' - только бинарная


def _toy_frame(n: int, n_features: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({f'f{i}': rng.randn(n) for i in range(n_features)})


def _binary_labels(X: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series((X['f0'] + rng.randn(len(X)) * 0.3 > 0).astype(int), name='y')


def _multiclass_labels(X: pd.DataFrame, seed: int, n_classes: int = 3) -> pd.Series:
    rng = np.random.RandomState(seed)
    noisy = X['f0'] + rng.randn(len(X)) * 0.3
    return pd.Series(pd.cut(noisy, bins=n_classes, labels=range(n_classes)).astype(int), name='y')


@pytest.fixture
def binary_split():
    X = _toy_frame(n=140, n_features=4, seed=1)
    y = _binary_labels(X, seed=2)
    return X.iloc[:100], y.iloc[:100], X.iloc[100:], y.iloc[100:]


@pytest.fixture
def multiclass_split():
    X = _toy_frame(n=140, n_features=4, seed=3)
    y = _multiclass_labels(X, seed=4)
    return X.iloc[:100], y.iloc[:100], X.iloc[100:], y.iloc[100:]


@pytest.mark.parametrize('metric', ALL_CLS_METRIC_NAMES)
def test_binary_classification_all_metrics_return_finite_float(binary_split, metric):
    X_train, y_train, X_valid, y_valid = binary_split
    scorer = make_catboost_scorer(task='classification', metric=metric, model_params=_MODEL_PARAMS)
    score = scorer(X_train, y_train, X_valid, y_valid)
    assert isinstance(score, float)
    assert np.isfinite(score)


@pytest.mark.parametrize('metric', MULTICLASS_SAFE_METRIC_NAMES)
def test_multiclass_all_metrics_return_finite_float(multiclass_split, metric):
    X_train, y_train, X_valid, y_valid = multiclass_split
    scorer = make_catboost_scorer(task='classification', metric=metric, model_params=_MODEL_PARAMS)
    score = scorer(X_train, y_train, X_valid, y_valid)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_ks_raises_valueerror_for_multiclass(multiclass_split):
    X_train, y_train, X_valid, y_valid = multiclass_split
    scorer = make_catboost_scorer(task='classification', metric='ks', model_params=_MODEL_PARAMS)
    with pytest.raises(ValueError, match='binary classification'):
        scorer(X_train, y_train, X_valid, y_valid)


def test_unsupported_metric_name_raises_with_available_list(binary_split):
    X_train, y_train, X_valid, y_valid = binary_split
    scorer = make_catboost_scorer(task='classification', metric='not_a_real_metric', model_params=_MODEL_PARAMS)
    with pytest.raises(ValueError, match='not_a_real_metric'):
        scorer(X_train, y_train, X_valid, y_valid)


def test_logloss_alias_matches_log_loss(binary_split):
    """'logloss' - старое имя в genetic API, должно давать тот же результат, что и 'log_loss'."""
    X_train, y_train, X_valid, y_valid = binary_split
    old_name_score = make_catboost_scorer(
        task='classification', metric='logloss', model_params=_MODEL_PARAMS,
    )(X_train, y_train, X_valid, y_valid)
    new_name_score = make_catboost_scorer(
        task='classification', metric='log_loss', model_params=_MODEL_PARAMS,
    )(X_train, y_train, X_valid, y_valid)
    assert old_name_score == pytest.approx(new_name_score)


@pytest.mark.parametrize(
    'metric', ['pr_auc', 'roc_auc', 'f1', 'balanced_accuracy', 'accuracy', 'mcc', 'precision', 'recall'],
)
def test_higher_is_better_binary_metrics_are_negated(binary_split, metric):
    """"Выше - лучше" метрики инвертируются для минимизации - на разделимых данных должны быть <= 0."""
    X_train, y_train, X_valid, y_valid = binary_split
    scorer = make_catboost_scorer(task='classification', metric=metric, model_params=_MODEL_PARAMS)
    assert scorer(X_train, y_train, X_valid, y_valid) <= 0.0


@pytest.mark.parametrize('metric', ['logloss', 'brier', 'ece'])
def test_lower_is_better_binary_metrics_are_not_negated(binary_split, metric):
    X_train, y_train, X_valid, y_valid = binary_split
    scorer = make_catboost_scorer(task='classification', metric=metric, model_params=_MODEL_PARAMS)
    assert scorer(X_train, y_train, X_valid, y_valid) >= 0.0


def test_model_params_not_mutated_when_multiclass_defaults_loss_function(multiclass_split):
    """cb_params = dict(model_params) внутри scorer - переданный словарь не должен мутировать."""
    X_train, y_train, X_valid, y_valid = multiclass_split
    params = dict(_MODEL_PARAMS)
    scorer = make_catboost_scorer(task='classification', metric='accuracy', model_params=params)
    scorer(X_train, y_train, X_valid, y_valid)
    assert 'loss_function' not in params


def test_explicit_loss_function_is_not_overridden(multiclass_split):
    X_train, y_train, X_valid, y_valid = multiclass_split
    params = {**_MODEL_PARAMS, 'loss_function': 'MultiClass'}
    scorer = make_catboost_scorer(task='classification', metric='accuracy', model_params=params)
    score = scorer(X_train, y_train, X_valid, y_valid)
    assert np.isfinite(score)
    assert params['loss_function'] == 'MultiClass'  # не тронуто


def test_callable_metric_receives_full_proba_matrix_for_multiclass(multiclass_split):
    X_train, y_train, X_valid, y_valid = multiclass_split
    seen_shapes = []

    def spy_metric(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        seen_shapes.append(y_proba.shape)
        return 0.0

    scorer = make_catboost_scorer(task='classification', metric=spy_metric, model_params=_MODEL_PARAMS)
    scorer(X_train, y_train, X_valid, y_valid)
    assert len(seen_shapes) == 1
    assert seen_shapes[0] == (len(X_valid), 3)  # (n, n_classes), не срез [:, 1]


def test_callable_metric_receives_1d_proba_for_binary(binary_split):
    X_train, y_train, X_valid, y_valid = binary_split
    seen_shapes = []

    def spy_metric(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        seen_shapes.append(y_proba.shape)
        return 0.0

    scorer = make_catboost_scorer(task='classification', metric=spy_metric, model_params=_MODEL_PARAMS)
    scorer(X_train, y_train, X_valid, y_valid)
    assert seen_shapes[0] == (len(X_valid),)


def test_regression_still_works_after_refactor():
    rng = np.random.RandomState(5)
    n = 120
    X = pd.DataFrame({f'f{i}': rng.randn(n) for i in range(4)})
    y = pd.Series(X['f0'] * 2 + rng.randn(n) * 0.1, name='y')
    X_train, y_train, X_valid, y_valid = X.iloc[:90], y.iloc[:90], X.iloc[90:], y.iloc[90:]

    scorer = make_catboost_scorer(task='regression', metric='mae', model_params=_MODEL_PARAMS)
    score = scorer(X_train, y_train, X_valid, y_valid)
    assert isinstance(score, float)
    assert score >= 0.0
