"""Проверяет, что все способы объяснения ModelExplainer работают на базовых моделях
(ml_toolkit.models), по одной модели на каждую ветку поведения explainer'а:

- CatBoostClassifier          — tree + SHAP (ветка catboost в _compute_shap)
- LightGBMClassifier          — tree + SHAP (ветка lightgbm в _compute_shap)
- RandomForestClassifier      — tree + SHAP (sklearn-ветка _compute_shap, Pipeline)
- DecisionTreeClassifier      — tree + SHAP + intrinsic (структура дерева)
- LinearClassifier              — |coef| importance, explain_row через coef, без SHAP/intrinsic
- InterpretableTreeRegressor     — intrinsic без SHAP (permutation — единственный importance),
  model_settings={'name': 'locally_linear_forest'} — единственный вариант из ALL_INTERPRETABLE,
  не требующий необязательных пакетов (torch/lineartree/imodels/interpret/pygam/pyearth)
- CatBoostRegressor              — та же матрица методов в task='regression'

xgboost/mondrian не входят в матрицу: пакеты xgboost/scikit-garden не входят в
зависимости проекта и не установлены в этом окружении — для mondrian отсутствие
SHAP проверяется отдельно как чистая проверка на уровне множеств (без обучения).
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ml_toolkit.model_explainer import ModelExplainer
from ml_toolkit.model_explainer.explainer import _SHAP_SUPPORTED, _TREE_NAMES
from ml_toolkit.models import (
    CatBoostClassifier, CatBoostRegressor, DecisionTreeClassifier,
    InterpretableTreeRegressor, LightGBMClassifier, LinearClassifier,
    RandomForestClassifier,
)
from tests.model_explainer.conftest import assert_valid_contribution, assert_valid_importance

FAST_CB = {'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'verbose': 0, 'random_seed': 42}


def _run_full_battery(explainer: ModelExplainer, X_valid: pd.DataFrame, feats: list[str], tmp_path) -> None:
    """Прогоняет весь публичный API ModelExplainer и проверяет базовые инварианты."""
    imp_auto = explainer.feature_importance(method='auto')
    assert_valid_importance(imp_auto, feats)

    contrib = explainer.explain_row(X_valid.iloc[[0]])
    assert_valid_contribution(contrib, feats)

    if explainer.supports_shap_:
        sv = explainer.shap_values(max_samples=50)
        assert sv.shape == (min(50, len(X_valid)), len(feats))

        imp_shap = explainer.feature_importance(method='shap')
        assert_valid_importance(imp_shap, feats)

        fig = explainer.plot_shap_beeswarm(max_samples=50)
        plt.close(fig)
        fig = explainer.plot_shap_waterfall(n_show=2)
        plt.close(fig)
    else:
        with pytest.raises(ValueError):
            explainer.shap_values()

    imp_perm = explainer.feature_importance(method='permutation', n_repeats=2)
    assert_valid_importance(imp_perm, feats)

    fig = explainer.plot_importance()
    plt.close(fig)
    fig = explainer.plot_partial_dependence(top_n=3)
    plt.close(fig)

    if explainer.supports_intrinsic_:
        assert explainer.plot_intrinsic(save_path=tmp_path / 'intrinsic.png') is True
        assert (tmp_path / 'intrinsic.png').exists()
    else:
        assert explainer.plot_intrinsic() is False

    saved = explainer.report(tmp_path / 'report', prefix='m')
    assert len(saved) > 0
    for p in saved:
        assert p.exists()


class TestCatBoostClassifier:
    def test_full_battery(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
        assert explainer.model_name_ == 'catboost'
        assert explainer.supports_shap_ is True
        assert explainer.supports_intrinsic_ is False
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)

        imp_gain = explainer.feature_importance(method='gain')
        assert_valid_importance(imp_gain, explainer.feature_names_)


class TestLightGBMClassifier:
    def test_full_battery(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(params={'n_estimators': 40, 'max_depth': 3, 'verbose': -1})
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
        assert explainer.model_name_ == 'lightgbm'
        assert explainer.supports_shap_ is True
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)


class TestRandomForestClassifier:
    def test_full_battery(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = RandomForestClassifier(params={'n_estimators': 30, 'max_depth': 4, 'random_state': 42})
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
        assert explainer.model_name_ == 'random_forest'
        assert explainer.supports_shap_ is True
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)


class TestDecisionTreeClassifier:
    def test_full_battery(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = DecisionTreeClassifier(params={'max_depth': 3, 'random_state': 42})
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
        assert explainer.model_name_ == 'decision_tree'
        assert explainer.supports_shap_ is True
        assert explainer.supports_intrinsic_ is True
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)


class TestLinearClassifier:
    def test_full_battery(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='classification')
        assert explainer.model_name_ == 'ridge'
        assert explainer.supports_shap_ is False
        assert explainer.supports_intrinsic_ is False
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)

        imp_coef = explainer.feature_importance(method='coef')
        assert_valid_importance(imp_coef, explainer.feature_names_)

        contrib = explainer.explain_row(X_valid.iloc[[0]])
        assert contrib.name == 'coef_contribution'


class TestInterpretableTreeRegressorLocallyLinearForest:
    def test_full_battery(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(
            params={'n_estimators': 30, 'max_depth': 4, 'n_neighbors': 20, 'ridge_alpha': 1.0},
            model_settings={'name': 'locally_linear_forest'},
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='regression')
        assert explainer.model_name_ == 'locally_linear_forest'
        assert explainer.supports_shap_ is False
        assert explainer.supports_intrinsic_ is True
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)

        contrib = explainer.explain_row(X_valid.iloc[[0]])
        assert contrib.name == 'local_contribution'


class TestCatBoostRegressor:
    def test_full_battery(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer = ModelExplainer(model, X_valid, y_valid, task='regression')
        assert explainer.model_name_ == 'catboost'
        assert explainer.supports_shap_ is True
        _run_full_battery(explainer, X_valid, explainer.feature_names_, tmp_path)


def test_mondrian_excluded_from_shap_support():
    """Mondrian — tree, но без SHAP (пакет scikit-garden не входит в зависимости
    проекта и недоступен в этом окружении, поэтому проверяем множества напрямую,
    без обучения реальной модели)."""
    assert 'mondrian' in _TREE_NAMES
    assert 'mondrian' not in _SHAP_SUPPORTED


class TestPermutationImportanceVerbose:
    """verbose=False (по умолчанию) — тихо; verbose=True — tqdm-прогресс, тот же
    результат (только для permutation — единственного метода с итеративным циклом)."""

    def test_verbose_toggles_progress_bar_without_changing_result(self, classification_data, capfd):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)

        explainer_quiet = ModelExplainer(model, X_valid, y_valid, task='classification')
        capfd.readouterr()
        imp_quiet = explainer_quiet.feature_importance(method='permutation', n_repeats=2, verbose=False)
        assert 'permutation importance' not in capfd.readouterr().err

        explainer_verbose = ModelExplainer(model, X_valid, y_valid, task='classification')
        imp_verbose = explainer_verbose.feature_importance(method='permutation', n_repeats=2, verbose=True)
        assert 'permutation importance' in capfd.readouterr().err

        pd.testing.assert_series_equal(imp_quiet, imp_verbose)
