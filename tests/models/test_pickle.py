"""Pickle-контракт BaseModel: fit() -> save() -> load() -> predict тот же, что до сохранения.

save()/load() — общая реализация BaseModel (ml_toolkit/models/_base.py), наследуемая
без переопределения каждым адаптером (`_tabular/_{family}/*`) и каждым пресетом
(`ml_toolkit.presets.classification`/`regression`, см. tests/presets/test_pickle.py) —
поэтому тест здесь один универсальный (assert_pickle_roundtrip в conftest.py), а не
per-family реализация save/load.

Пакеты, не входящие в обязательные зависимости проекта (xgboost, quantile_forest,
sktree, skgarden, interpret, pygam, pyearth, imodels, lineartree, torch), пропускаются
через pytest.importorskip — как и в их собственных test_*.py файлах; код готов
прогнаться, как только пакет появится в окружении.

LAMAClassifier/LAMARegressor намеренно НЕ фитятся здесь даже при наличии lightautoml —
см. докстринг tests/models/_tabular/_automl/test_lama.py про SIGSEGV в multiprocessing
LightAutoML на macOS, крашащий весь процесс pytest. Для них — только структурная
проверка, что save/load унаследованы (не требует fit()).
"""

from __future__ import annotations

import pytest

from tests.models.conftest import assert_pickle_roundtrip

# ── Boosting ─────────────────────────────────────────────────────────────────

from ml_toolkit.models._tabular._boosting._catboost import CatBoostClassifier, CatBoostRegressor
from ml_toolkit.models._tabular._boosting._catboost_ranker import CatBoostRanker
from ml_toolkit.models._tabular._boosting._lightgbm import LightGBMClassifier, LightGBMRegressor
from ml_toolkit.models._tabular._boosting._lightgbm_ranker import LightGBMRanker
from tests.models._tabular._boosting.test_catboost import FAST_CB
from tests.models._tabular._boosting.test_lightgbm import FAST_LGB
from tests.models._tabular._boosting.test_rankers import FAST_CB_RANK, FAST_LGB_RANK


class TestBoostingPickle:
    def test_catboost_regressor(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_catboost_classifier(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_catboost_ranker(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(params=FAST_CB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_lightgbm_regressor(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_lightgbm_classifier(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_lightgbm_ranker(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(params=FAST_LGB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_xgboost_regressor(self, regression_data, tmp_path):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._tabular._boosting._xgboost import XGBoostRegressor
        from tests.models._tabular._boosting.test_xgboost import FAST_XGB
        X_train, y_train, X_valid, y_valid = regression_data
        model = XGBoostRegressor(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_xgboost_classifier(self, classification_data, tmp_path):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._tabular._boosting._xgboost import XGBoostClassifier
        from tests.models._tabular._boosting.test_xgboost import FAST_XGB
        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostClassifier(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_xgboost_ranker(self, classification_data, tmp_path):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._tabular._boosting._xgboost_ranker import XGBoostRanker
        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostRanker(params={'n_estimators': 40, 'max_depth': 3})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)


# ── Forests ──────────────────────────────────────────────────────────────────

from ml_toolkit.models._tabular._forests._decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ml_toolkit.models._tabular._forests._forest import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from ml_toolkit.models._tabular._forests._hist_gbm import HistGBMClassifier, HistGBMRegressor
from tests.models._tabular._forests.test_decision_tree import FAST_PARAMS as DT_PARAMS
from tests.models._tabular._forests.test_forest import FAST_PARAMS as FOREST_PARAMS
from tests.models._tabular._forests.test_hist_gbm import FAST_PARAMS as HISTGBM_PARAMS


class TestForestsPickle:
    def test_decision_tree_regressor(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = DecisionTreeRegressor(params=DT_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_decision_tree_classifier(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = DecisionTreeClassifier(params=DT_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    @pytest.mark.parametrize('RegClass', [RandomForestRegressor, ExtraTreesRegressor])
    def test_forest_regressors(self, RegClass, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = RegClass(params=FOREST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    @pytest.mark.parametrize('ClsClass', [RandomForestClassifier, ExtraTreesClassifier])
    def test_forest_classifiers(self, ClsClass, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = ClsClass(params=FOREST_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_hist_gbm_regressor(self, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = HistGBMRegressor(params=HISTGBM_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_hist_gbm_classifier(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = HistGBMClassifier(params=HISTGBM_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_quantile_forest_regressor(self, regression_data, tmp_path):
        pytest.importorskip('quantile_forest')
        from ml_toolkit.models._tabular._forests._quantile_forest import QuantileForestRegressor
        from tests.models._tabular._forests.test_quantile_forest import FAST_REG_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = QuantileForestRegressor(params=FAST_REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_quantile_forest_classifier(self, classification_data, tmp_path):
        pytest.importorskip('quantile_forest')
        from ml_toolkit.models._tabular._forests._quantile_forest import QuantileForestClassifier
        from tests.models._tabular._forests.test_quantile_forest import FAST_CLS_PARAMS
        X_train, y_train, X_valid, y_valid = classification_data
        model = QuantileForestClassifier(params=FAST_CLS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_oblique_forest_regressor(self, regression_data, tmp_path):
        pytest.importorskip('sktree')
        from ml_toolkit.models._tabular._forests._oblique_forest import ObliqueForestRegressor
        from tests.models._tabular._forests.test_oblique_forest import FAST_PARAMS as OF_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = ObliqueForestRegressor(params=OF_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_mondrian_forest_regressor(self, regression_data, tmp_path):
        pytest.importorskip('skgarden')
        from ml_toolkit.models._tabular._forests._mondrian import MondrianForestRegressor
        from tests.models._tabular._forests.test_mondrian import FAST_PARAMS as MF_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = MondrianForestRegressor(params=MF_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)


# ── Interpretable ────────────────────────────────────────────────────────────

from ml_toolkit.models._tabular._interpretable._linear import LinearClassifier, LinearRegressor


class TestInterpretablePickle:
    @pytest.mark.parametrize('name', ['ridge', 'elasticnet', 'huber', 'quantile', 'bayesian_ridge'])
    def test_linear_regressor(self, name, regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={}, model_settings={'name': name})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_linear_regressor_tweedie(self, positive_regression_data, tmp_path):
        X_train, y_train, X_valid, y_valid = positive_regression_data
        model = LinearRegressor(params={}, model_settings={'name': 'tweedie'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_linear_classifier(self, classification_data, tmp_path):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_locally_linear_forest_regressor(self, regression_data, tmp_path):
        """Единственный вариант InterpretableTree*, не требующий torch (см. модуль)."""
        from ml_toolkit.models._tabular._interpretable._interpretable_trees import InterpretableTreeRegressor
        X_train, y_train, X_valid, y_valid = regression_data
        params = {'n_estimators': 20, 'max_depth': 4, 'n_neighbors': 20, 'ridge_alpha': 1.0, 'random_state': 42}
        model = InterpretableTreeRegressor(params=params, model_settings={'name': 'locally_linear_forest'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_soft_decision_tree_regressor(self, regression_data, tmp_path):
        pytest.importorskip('torch')
        from ml_toolkit.models._tabular._interpretable._interpretable_trees import InterpretableTreeRegressor
        from tests.models._tabular._interpretable.test_interpretable_trees import SDT_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableTreeRegressor(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_soft_decision_tree_classifier(self, classification_data, tmp_path):
        pytest.importorskip('torch')
        from ml_toolkit.models._tabular._interpretable._interpretable_trees import InterpretableTreeClassifier
        from tests.models._tabular._interpretable.test_interpretable_trees import SDT_PARAMS
        X_train, y_train, X_valid, y_valid = classification_data
        model = InterpretableTreeClassifier(params=SDT_PARAMS, model_settings={'name': 'soft_decision_tree'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_interpretable_neural_regressor(self, regression_data, tmp_path):
        pytest.importorskip('torch')
        from ml_toolkit.models._tabular._interpretable._interpretable_neural import InterpretableNeuralRegressor
        from tests.models._tabular._interpretable.test_interpretable_neural import REG_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = InterpretableNeuralRegressor(params=REG_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_ebm_regressor(self, regression_data, tmp_path):
        pytest.importorskip('interpret')
        from ml_toolkit.models._tabular._interpretable._ebm import EBMRegressor
        from tests.models._tabular._interpretable.test_ebm import FAST_PARAMS as EBM_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = EBMRegressor(params=EBM_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_ebm_classifier(self, classification_data, tmp_path):
        pytest.importorskip('interpret')
        from ml_toolkit.models._tabular._interpretable._ebm import EBMClassifier
        from tests.models._tabular._interpretable.test_ebm import FAST_PARAMS as EBM_PARAMS
        X_train, y_train, X_valid, y_valid = classification_data
        model = EBMClassifier(params=EBM_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path, proba=True)

    def test_pygam_regressor(self, regression_data, tmp_path):
        pytest.importorskip('pygam')
        from ml_toolkit.models._tabular._interpretable._gam import PyGAMRegressor
        from tests.models._tabular._interpretable.test_gam import FAST_PARAMS as GAM_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = PyGAMRegressor(params=GAM_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_mars_regressor(self, regression_data, tmp_path):
        pytest.importorskip('pyearth')
        from ml_toolkit.models._tabular._interpretable._mars import MARSRegressor
        from tests.models._tabular._interpretable.test_mars import FAST_REG_PARAMS as MARS_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = MARSRegressor(params=MARS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_rulefit_regressor(self, regression_data, tmp_path):
        pytest.importorskip('imodels')
        from ml_toolkit.models._tabular._interpretable._rulefit import RuleFitRegressor
        from tests.models._tabular._interpretable.test_rulefit import FAST_PARAMS as RULEFIT_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = RuleFitRegressor(params=RULEFIT_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_imodels_regressor_figs(self, regression_data, tmp_path):
        pytest.importorskip('imodels')
        from ml_toolkit.models._tabular._interpretable._imodels import IModelsRegressor
        from tests.models._tabular._interpretable.test_imodels import FIGS_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = IModelsRegressor(params=FIGS_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_linear_tree_regressor(self, regression_data, tmp_path):
        pytest.importorskip('lineartree')
        from ml_toolkit.models._tabular._interpretable._linear_tree import LinearTreeRegressor
        from tests.models._tabular._interpretable.test_linear_tree import FAST_PARAMS as LT_PARAMS
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearTreeRegressor(params=LT_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)


# ── AutoML ───────────────────────────────────────────────────────────────────

class TestAutoMLPickle:
    def test_tabm_regressor(self, regression_data, tmp_path):
        pytest.importorskip('torch')
        pytest.importorskip('tabm')
        from ml_toolkit.models._tabular._automl._tabm import TabMRegressor
        from tests.models._tabular._automl.test_tabm import FAST_PARAMS as TABM_PARAMS, FAST_SETTINGS
        X_train, y_train, X_valid, y_valid = regression_data
        model = TabMRegressor(params=TABM_PARAMS, model_settings=FAST_SETTINGS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_pickle_roundtrip(model, X_valid, tmp_path)

    def test_lama_has_save_load_without_fitting(self):
        """LAMA не фитится здесь (см. докстринг модуля) — только структурная проверка.

        save/load должны быть унаследованы от BaseModel без необходимости
        реально обучать модель (что здесь и проверяем — импорт не требует
        lightautoml, т.к. он лениво импортируется только внутри fit()).
        """
        from ml_toolkit.models._tabular._automl._lama import LAMAClassifier, LAMARegressor

        for cls in (LAMARegressor, LAMAClassifier):
            assert hasattr(cls, 'save')
            assert hasattr(cls, 'load')
