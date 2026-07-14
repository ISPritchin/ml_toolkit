"""Тесты для CatBoostRegressor/CatBoostClassifier (ml_toolkit/models/_tabular/_boosting/_catboost.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.models._tabular._boosting._catboost import CatBoostClassifier, CatBoostRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba

FAST_CB = {'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'verbose': 0, 'random_seed': 42}


class TestCatBoostRegressorExplicitParams:
    def test_fit_predict(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ == FAST_CB

    def test_fit_without_valid(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train)
        assert model.valid_pred_ is None
        assert_valid_predictions(model, X_train)


class TestCatBoostRegressorOptuna:
    def test_requires_valid_set(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = CatBoostRegressor(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert model.best_params_ is not None
        assert 'iterations' in model.best_params_


class TestCatBoostRegressorBaseline:
    def test_predict_uses_baseline_from_input(self, regression_data):
        """baseline_col передаётся Pool(baseline=...) на train И на predict.

        Trees учат только residual относительно baseline. Если на predict() baseline-колонку убрать,
        то её вклад в предсказание пропадёт и итоговый скор станет намного хуже,
        доказывая, что predict() реально читает baseline из X, а не игнорирует его.
        """
        rng = np.random.default_rng(5)
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        # baseline — почти точный предиктор таргета: residual тривиален для 40 итераций
        X_train['baseline'] = y_train.to_numpy() + rng.normal(scale=0.05, size=len(y_train))
        X_valid['baseline'] = y_valid.to_numpy() + rng.normal(scale=0.05, size=len(y_valid))

        feats = ['f0', 'f1', 'f2', 'f3', 'f4']
        model = CatBoostRegressor(params=FAST_CB, model_settings={'baseline_col': 'baseline'})
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)

        pred_with_baseline = model.predict(X_valid)
        err_with_baseline = np.abs(pred_with_baseline - y_valid.to_numpy()).mean()

        X_valid_no_baseline = X_valid.drop(columns=['baseline'])
        pred_without_baseline = model.predict(X_valid_no_baseline)
        err_without_baseline = np.abs(pred_without_baseline - y_valid.to_numpy()).mean()

        assert err_with_baseline < 0.5
        assert err_without_baseline > err_with_baseline * 5


class TestCatBoostClassifierExplicitParams:
    def test_fit_predict_proba(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_when_valid_passed(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_no_calibrator_without_valid(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        # Без калибратора predict_proba должен по-прежнему возвращать валидные вероятности
        assert_valid_proba(model, X_train)


class TestCatBoostClassifierOptuna:
    def test_requires_valid_set(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = CatBoostClassifier(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_ is not None


class TestCatBoostClassifierCustomMetric:
    def test_callable_cls_metric_used_by_optuna(self, classification_data):
        """cls_metric как callable — сюда попадает roc_auc_score вместо PR-AUC по умолчанию."""
        from sklearn.metrics import roc_auc_score

        X_train, y_train, X_valid, y_valid = classification_data
        model = CatBoostClassifier(
            n_optuna_trials=2,
            model_settings={'cls_metric': (roc_auc_score, 'maximize')},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)


class TestCatBoostCatFeatures:
    def test_categorical_feature_used_natively_classifier(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert model.cat_features_ == ['cat_col']
        assert_valid_proba(model, X_valid)

    def test_categorical_feature_used_natively_regressor(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert model.cat_features_ == ['cat_col']
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_classifier(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = CatBoostClassifier(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_regressor(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = CatBoostRegressor(params=FAST_CB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        assert model.cat_features_ == MULTI_CAT_FEATURES
        assert_valid_predictions(model, X_valid)


class TestCatBoostRegressorCustomMetric:
    def test_callable_reg_metric_used_by_optuna(self, regression_data):
        """reg_metric как callable — сюда попадает RMSE вместо MAE по умолчанию."""
        from sklearn.metrics import mean_squared_error

        def rmse(y_true, y_pred):
            return float(mean_squared_error(y_true, y_pred) ** 0.5)

        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(
            n_optuna_trials=2,
            model_settings={'reg_metric': (rmse, 'minimize')},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_named_reg_metric_preset(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'reg_metric': 'rmse'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_unknown_named_reg_metric_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'reg_metric': 'not_a_metric'})
        with pytest.raises(ValueError, match='reg_metric'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestCatBoostMulticlass:
    @pytest.fixture
    def multiclass_data(self):
        rng = np.random.default_rng(11)
        n_train, n_valid = 300, 90
        cols = [f'f{i}' for i in range(5)]
        X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
        y_train = pd.Series(rng.integers(0, 3, size=n_train))
        X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
        y_valid = pd.Series(rng.integers(0, 3, size=n_valid))
        return X_train, y_train, X_valid, y_valid

    def test_fit_predict_proba_shape(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        params = {**FAST_CB, 'loss_function': 'MultiClass', 'eval_metric': 'AUC'}
        model = CatBoostClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.n_classes_ == 3
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid), 3)
        # Строки — распределение вероятностей по классам
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_multiclass_calibrators_fitted_with_valid(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        params = {**FAST_CB, 'loss_function': 'MultiClass', 'eval_metric': 'AUC'}
        model = CatBoostClassifier(params=params)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.calibrators_ is not None
        assert len(model.calibrators_) == 3
        assert model.calibrator_ is None  # бинарный слот не используется в мультиклассе

    def test_no_multiclass_calibrators_without_valid(self, multiclass_data):
        X_train, y_train, _, _ = multiclass_data
        params = {**FAST_CB, 'loss_function': 'MultiClass', 'eval_metric': 'AUC'}
        model = CatBoostClassifier(params=params)
        model.fit(X_train, y_train)
        assert model.calibrators_ is None
        proba = model.predict_proba(X_train)
        assert proba.shape == (len(X_train), 3)


class TestCatBoostUndersampleMajority:
    def test_default_false_trains_on_full_data(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._tabular._boosting._catboost'):
            model = CatBoostClassifier(n_optuna_trials=2)
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert any('undersample_majority=False' in r.message for r in caplog.records)

    def test_enabled_trains_on_subsample(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._tabular._boosting._catboost'):
            model = CatBoostClassifier(n_optuna_trials=2, model_settings={'undersample_majority': True})
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert not any('undersample_majority=False' in r.message for r in caplog.records)


class TestCatBoostParamSpace:
    def test_custom_param_space_overrides_default(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data

        def my_space(trial):
            return {
                'iterations': trial.suggest_int('iterations', 20, 40, step=10),
                'max_depth': 3,
            }

        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'param_space': my_space})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 20 <= model.best_params_['iterations'] <= 40
        assert model.best_params_['max_depth'] == 3


class TestCatBoostOptunaPruner:
    def test_named_pruner_alias(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=3, model_settings={'optuna_pruner': 'hyperband'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_none_pruner_alias_disables_pruning(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'optuna_pruner': 'none'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_unknown_pruner_alias_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'optuna_pruner': 'not_a_pruner'})
        with pytest.raises(ValueError, match='optuna_pruner'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestCatBoostOptunaTimeout:
    @pytest.mark.slow
    def test_timeout_stops_study_early(self, regression_data):
        import time

        X_train, y_train, X_valid, y_valid = regression_data
        model = CatBoostRegressor(
            n_optuna_trials=200, model_settings={'optuna_timeout': 1},
        )
        start = time.monotonic()
        model.fit(X_train, y_train, X_valid, y_valid)
        elapsed = time.monotonic() - start

        assert_valid_predictions(model, X_valid)
        # 200 триалов без лимита заняли бы намного дольше пары секунд
        assert elapsed < 15


class TestCatBoostOptunaVerbose:
    """capfd намеренно не используется: логи optuna пишутся через logging-хендлер.

    Хендлер буферизует/флашит асинхронно относительно вызовов capfd.readouterr() — на практике это
    приводило к тому, что строки Trial N finished оказывались в чужом окне захвата. caplog
    перехватывает LogRecord синхронно в момент emit(), без этой гонки.
    """

    def test_verbose_false_forces_warning_during_fit(self, regression_data, caplog):
        import logging

        import optuna

        X_train, y_train, X_valid, y_valid = regression_data
        optuna.logging.set_verbosity(optuna.logging.INFO)

        with caplog.at_level(logging.INFO, logger='optuna'):
            model = CatBoostRegressor(n_optuna_trials=2, model_settings={'optuna_verbose': False})
            model.fit(X_train, y_train, X_valid, y_valid)

        assert not any('Trial' in r.message for r in caplog.records)

    def test_verbose_true_does_not_suppress_optuna_logs(self, regression_data, caplog):
        import logging

        import optuna

        X_train, y_train, X_valid, y_valid = regression_data
        optuna.logging.set_verbosity(optuna.logging.INFO)

        with caplog.at_level(logging.INFO, logger='optuna'):
            model = CatBoostRegressor(n_optuna_trials=2, model_settings={'optuna_verbose': True})
            model.fit(X_train, y_train, X_valid, y_valid)

        assert any('Trial' in r.message for r in caplog.records)

    def test_verbosity_restored_after_fit(self, regression_data):
        import optuna

        X_train, y_train, X_valid, y_valid = regression_data
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

        model = CatBoostRegressor(n_optuna_trials=2, model_settings={'optuna_verbose': False})
        model.fit(X_train, y_train, X_valid, y_valid)

        # fit() приглушает Optuna на время тюнинга, но обязан вернуть исходный уровень
        assert optuna.logging.get_verbosity() == optuna.logging.DEBUG

