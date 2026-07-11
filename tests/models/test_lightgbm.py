"""Тесты для LightGBMRegressor/LightGBMClassifier (ml_toolkit/models/_lightgbm.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.models._lightgbm import LightGBMClassifier, LightGBMRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_LGB = {'n_estimators': 40, 'max_depth': 3, 'num_leaves': 7, 'verbose': -1}


class TestLightGBMRegressorExplicitParams:
    def test_fit_predict(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_LGB

    def test_fit_without_valid(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = LightGBMRegressor(params=FAST_LGB)
        model.fit(X_train, y_train)
        assert model.valid_pred_ is None
        assert_valid_predictions(model, X_train)


class TestLightGBMRegressorOptuna:
    def test_requires_valid_set(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = LightGBMRegressor(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    @pytest.mark.slow
    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 'n_estimators' in model.best_params_


class TestLightGBMRegressorBaseline:
    def test_predict_adds_baseline_back(self, regression_data):
        """LightGBM residual learning: обучается на (y - baseline), predict() прибавляет baseline обратно вручную.

        В отличие от CatBoost, где это делает Pool нативно.
        """
        rng = np.random.default_rng(5)
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['baseline'] = y_train.to_numpy() + rng.normal(scale=0.05, size=len(y_train))
        X_valid['baseline'] = y_valid.to_numpy() + rng.normal(scale=0.05, size=len(y_valid))

        feats = ['f0', 'f1', 'f2', 'f3', 'f4']
        model = LightGBMRegressor(params=FAST_LGB, model_settings={'baseline_col': 'baseline'})
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)

        pred_with_baseline = model.predict(X_valid)
        err_with_baseline = np.abs(pred_with_baseline - y_valid.to_numpy()).mean()

        X_valid_no_baseline = X_valid.drop(columns=['baseline'])
        pred_without_baseline = model.predict(X_valid_no_baseline)
        err_without_baseline = np.abs(pred_without_baseline - y_valid.to_numpy()).mean()

        assert err_with_baseline < 0.5
        assert err_without_baseline > err_with_baseline * 5


class TestLightGBMClassifierExplicitParams:
    def test_fit_predict_proba(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_when_valid_passed(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_no_calibrator_without_valid(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None


class TestLightGBMClassifierOptuna:
    def test_requires_valid_set(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = LightGBMClassifier(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    @pytest.mark.slow
    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LightGBMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'n_estimators' in model.best_params_


class TestLightGBMCatFeatures:
    def test_categorical_feature_used_natively_classifier(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert model.cat_features_ == ['cat_col']
        assert_valid_proba(model, X_valid)

    def test_categorical_feature_used_natively_regressor(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = LightGBMRegressor(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert model.cat_features_ == ['cat_col']
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_classifier(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_regressor(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = LightGBMRegressor(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_predictions(model, X_valid)


class TestLightGBMRegressorCustomMetric:
    @pytest.mark.slow
    def test_callable_reg_metric_used_by_optuna(self, regression_data):
        from sklearn.metrics import mean_squared_error

        def rmse(y_true, y_pred):
            return float(mean_squared_error(y_true, y_pred) ** 0.5)

        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=2, model_settings={'reg_metric': (rmse, 'minimize')})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_unknown_named_reg_metric_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=2, model_settings={'reg_metric': 'not_a_metric'})
        with pytest.raises(ValueError, match='reg_metric'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestLightGBMBoostingType:
    @pytest.mark.parametrize('boosting_type', ['gbdt', 'dart', 'goss'])
    def test_explicit_boosting_type(self, boosting_type, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        params = {'n_estimators': 40, 'max_depth': 3, 'num_leaves': 7, 'verbose': -1,
                   'boosting_type': boosting_type}
        model = LightGBMRegressor(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_param_space_may_omit_boosting_type_defaults_to_gbdt(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data

        def my_space(trial):
            return {'n_estimators': trial.suggest_int('n_estimators', 30, 60, step=10)}

        model = LightGBMRegressor(n_optuna_trials=2, model_settings={'param_space': my_space})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_.get('boosting_type', 'gbdt') == 'gbdt'


class TestLightGBMMulticlass:
    @pytest.fixture
    def multiclass_data(self):
        rng = np.random.default_rng(13)
        n_train, n_valid = 200, 60
        cols = [f'f{i}' for i in range(5)]
        X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
        y_train = pd.Series(rng.integers(0, 3, size=n_train))
        X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
        y_valid = pd.Series(rng.integers(0, 3, size=n_valid))
        return X_train, y_train, X_valid, y_valid

    def test_explicit_params_fit_predict_proba_shape(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LightGBMClassifier(params={**FAST_LGB, 'objective': 'multiclass', 'num_class': 3})
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.n_classes_ == 3
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_explicit_params_infers_multiclass_without_objective(self, multiclass_data):
        """LightGBM сам определяет multiclass objective по количеству классов в y, если он не задан явно.

        Адаптер это не форсирует в explicit-params ветке.
        """
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LightGBMClassifier(params=FAST_LGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.n_classes_ == 3
        assert model.predict_proba(X_valid).shape == (len(X_valid), 3)

    @pytest.mark.slow
    def test_optuna_fit_predict_proba_shape(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LightGBMClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.n_classes_ == 3
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert 'num_class' in model.best_params_
        assert model.best_params_['objective'] == 'multiclass'

    def test_multiclass_calibrators_fitted_with_valid(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LightGBMClassifier(params={**FAST_LGB, 'objective': 'multiclass', 'num_class': 3})
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.calibrators_ is not None
        assert len(model.calibrators_) == 3
        assert model.calibrator_ is None

    def test_no_multiclass_calibrators_without_valid(self, multiclass_data):
        X_train, y_train, _, _ = multiclass_data
        model = LightGBMClassifier(params={**FAST_LGB, 'objective': 'multiclass', 'num_class': 3})
        model.fit(X_train, y_train)
        assert model.calibrators_ is None
        proba = model.predict_proba(X_train)
        assert proba.shape == (len(X_train), 3)

    @pytest.mark.slow
    def test_undersample_majority_uses_balance_fraction(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = LightGBMClassifier(n_optuna_trials=2, model_settings={'undersample_majority': True})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.predict_proba(X_valid).shape == (len(X_valid), 3)


class TestLightGBMUndersampleMajority:
    @pytest.mark.slow
    def test_default_false_trains_on_full_data(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._lightgbm'):
            model = LightGBMClassifier(n_optuna_trials=2)
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert any('undersample_majority=False' in r.message for r in caplog.records)

    def test_enabled_trains_on_subsample(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._lightgbm'):
            model = LightGBMClassifier(n_optuna_trials=2, model_settings={'undersample_majority': True})
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert not any('undersample_majority=False' in r.message for r in caplog.records)


class TestLightGBMParamSpace:
    def test_custom_param_space_overrides_default(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data

        def my_space(trial):
            return {'n_estimators': trial.suggest_int('n_estimators', 20, 40, step=10)}

        model = LightGBMClassifier(n_optuna_trials=2, model_settings={'param_space': my_space})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 20 <= model.best_params_['n_estimators'] <= 40


class TestLightGBMOptunaPruner:
    @pytest.mark.slow
    def test_named_pruner_alias(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=3, model_settings={'optuna_pruner': 'hyperband'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_unknown_pruner_alias_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=2, model_settings={'optuna_pruner': 'not_a_pruner'})
        with pytest.raises(ValueError, match='optuna_pruner'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestLightGBMOptunaTimeout:
    @pytest.mark.slow
    def test_timeout_stops_study_early(self, regression_data):
        import time

        X_train, y_train, X_valid, y_valid = regression_data
        model = LightGBMRegressor(n_optuna_trials=200, model_settings={'optuna_timeout': 1})
        start = time.monotonic()
        model.fit(X_train, y_train, X_valid, y_valid)
        elapsed = time.monotonic() - start

        assert_valid_predictions(model, X_valid)
        assert elapsed < 15


class TestLightGBMOptunaVerbose:
    """См. TestCatBoostOptunaVerbose в test_catboost.py — capfd не используется намеренно.

    Из-за гонки между асинхронным flush optuna-логов и readouterr(); caplog синхронен.
    """

    @pytest.mark.slow
    def test_verbose_false_forces_warning_during_fit(self, regression_data, caplog):
        import logging

        import optuna

        X_train, y_train, X_valid, y_valid = regression_data
        optuna.logging.set_verbosity(optuna.logging.INFO)

        with caplog.at_level(logging.INFO, logger='optuna'):
            model = LightGBMRegressor(n_optuna_trials=2, model_settings={'optuna_verbose': False})
            model.fit(X_train, y_train, X_valid, y_valid)

        assert not any('Trial' in r.message for r in caplog.records)

    @pytest.mark.slow
    def test_verbose_true_does_not_suppress_optuna_logs(self, regression_data, caplog):
        import logging

        import optuna

        X_train, y_train, X_valid, y_valid = regression_data
        optuna.logging.set_verbosity(optuna.logging.INFO)

        with caplog.at_level(logging.INFO, logger='optuna'):
            model = LightGBMRegressor(n_optuna_trials=2, model_settings={'optuna_verbose': True})
            model.fit(X_train, y_train, X_valid, y_valid)

        assert any('Trial' in r.message for r in caplog.records)

