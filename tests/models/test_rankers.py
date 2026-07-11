"""Тесты для ранжировщиков: CatBoostRanker, LightGBMRanker, XGBoostRanker.

ml_toolkit/models/_catboost_ranker.py, _lightgbm_ranker.py, _xgboost_ranker.py.

Все три адаптера решают бинарную классификацию через ranking objective (вся выборка —
одна группа или несколько групп заданного размера); predict_proba() возвращает
скоры, калиброванные изотонической регрессией на валидации. XGBoostRanker пропускается,
если пакет xgboost не установлен (importorskip) — код готов к среде, где он есть.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml_toolkit.models._catboost_ranker import CatBoostRanker
from ml_toolkit.models._lightgbm_ranker import LightGBMRanker
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_proba

FAST_CB_RANK = {'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'loss_function': 'YetiRank', 'verbose': False}
FAST_LGB_RANK = {'n_estimators': 40, 'num_leaves': 7, 'max_depth': 3, 'objective': 'lambdarank', 'verbose': -1}


class TestCatBoostRanker:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(params=FAST_CB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(params=FAST_CB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_predict_without_calibrator_still_bounded(self, classification_data):
        """Без X_valid калибратора нет — скор нормируется min-max в [0, 1] вручную."""
        X_train, y_train, _, _ = classification_data
        model = CatBoostRanker(params=FAST_CB_RANK)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        pred = model.predict_proba(X_train)
        assert np.all((pred >= 0) & (pred <= 1))

    def test_requires_valid_for_optuna(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = CatBoostRanker(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_group_size_splits_into_multiple_groups(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(params=FAST_CB_RANK, model_settings={'group_size': 50})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['loss_function'] == 'YetiRank'

    def test_rank_objective_setting(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(
            params={**FAST_CB_RANK, 'loss_function': 'QuerySoftMax'},
            model_settings={'rank_objective': 'QuerySoftMax'},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_custom_cls_metric_used_by_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostRanker(n_optuna_trials=2, model_settings={'cls_metric': (roc_auc_score, 'maximize')})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = CatBoostRanker(params=FAST_CB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_proba(model, X_valid)


class TestLightGBMRanker:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(params=FAST_LGB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(params=FAST_LGB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_requires_valid_for_optuna(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = LightGBMRanker(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_group_size_splits_into_multiple_groups(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(params=FAST_LGB_RANK, model_settings={'group_size': 50})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_predict_without_calibrator_still_bounded(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = LightGBMRanker(params=FAST_LGB_RANK)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        pred = model.predict_proba(X_train)
        assert np.all((pred >= 0) & (pred <= 1))

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['objective'] == 'lambdarank'

    def test_rank_objective_setting(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMRanker(
            params={**FAST_LGB_RANK, 'objective': 'rank_xendcg'},
            model_settings={'rank_objective': 'rank_xendcg'},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = LightGBMRanker(params=FAST_LGB_RANK)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_proba(model, X_valid)


class TestXGBoostRanker:
    def test_fit_predict_proba(self, classification_data):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._xgboost_ranker import XGBoostRanker

        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostRanker(params={
            'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.2, 'objective': 'rank:ndcg',
        })
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_predict_without_calibrator_still_bounded(self, classification_data):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._xgboost_ranker import XGBoostRanker

        X_train, y_train, _, _ = classification_data
        model = XGBoostRanker(params={
            'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.2, 'objective': 'rank:ndcg',
        })
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        pred = model.predict_proba(X_train)
        assert np.all((pred >= 0) & (pred <= 1))

    def test_fit_with_optuna(self, classification_data):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._xgboost_ranker import XGBoostRanker

        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostRanker(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['objective'] == 'rank:ndcg'

    def test_rank_objective_setting(self, classification_data):
        pytest.importorskip('xgboost')
        from ml_toolkit.models._xgboost_ranker import XGBoostRanker

        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostRanker(
            params={'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.2, 'objective': 'rank:pairwise'},
            model_settings={'rank_objective': 'rank:pairwise'},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features(self, classification_data_multi_cat):
        """XGBoostRanker не поддерживает category/object dtype нативно — _to_float() кодирует любую колонку.

        Object/category колонку в числовые коды (cat.codes) независимо от cat_features,
        в отличие от XGBoostClassifier/Regressor (enable_categorical=True). Тест фиксирует
        текущее поведение: обучение и предсказание работают, а не падают на нескольких
        категориальных признаках разной кардинальности.
        """
        pytest.importorskip('xgboost')
        from ml_toolkit.models._xgboost_ranker import XGBoostRanker

        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = XGBoostRanker(params={
            'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.2, 'objective': 'rank:ndcg',
        })
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_proba(model, X_valid)
