"""Тесты для XGBoostRegressor/XGBoostClassifier (ml_toolkit/models/_xgboost.py).

Пакет xgboost не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен (как и test_rankers.py::TestXGBoostRanker).
Код написан и рассчитан на прогон, как только xgboost появится в окружении/CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xgboost = pytest.importorskip('xgboost')

from ml_toolkit.models import train_classification_model, train_regression_model  # noqa: E402
from ml_toolkit.models._xgboost import XGBoostClassifier, XGBoostRegressor  # noqa: E402
from tests.models.conftest import assert_valid_predictions, assert_valid_proba  # noqa: E402

FAST_XGB = {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.2}


class TestXGBoostRegressorExplicitParams:
    def test_fit_predict(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = XGBoostRegressor(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        # enable_categorical добавлен адаптером поверх переданных params
        assert model.best_params_['enable_categorical'] is False

    def test_fit_without_valid(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = XGBoostRegressor(params=FAST_XGB)
        model.fit(X_train, y_train)
        assert model.valid_pred_ is None
        assert_valid_predictions(model, X_train)


class TestXGBoostRegressorBaseline:
    def test_predict_adds_baseline_back(self, regression_data):
        """XGBoost residual learning: обучается на (y - baseline), predict() прибавляет
        baseline обратно вручную — тот же контракт, что у LightGBMRegressor.
        """
        rng = np.random.default_rng(5)
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['baseline'] = y_train.to_numpy() + rng.normal(scale=0.05, size=len(y_train))
        X_valid['baseline'] = y_valid.to_numpy() + rng.normal(scale=0.05, size=len(y_valid))

        feats = ['f0', 'f1', 'f2', 'f3', 'f4']
        model = XGBoostRegressor(params=FAST_XGB, model_settings={'baseline_col': 'baseline'})
        model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)

        pred_with_baseline = model.predict(X_valid)
        err_with_baseline = np.abs(pred_with_baseline - y_valid.to_numpy()).mean()

        X_valid_no_baseline = X_valid.drop(columns=['baseline'])
        pred_without_baseline = model.predict(X_valid_no_baseline)
        err_without_baseline = np.abs(pred_without_baseline - y_valid.to_numpy()).mean()

        assert err_with_baseline < 0.5
        assert err_without_baseline > err_with_baseline * 5

    def test_baseline_combines_with_postprocess_fn_and_optuna(self, regression_data):
        """См. TestCatBoostRegressorBaseline в test_catboost.py — тот же трёхсторонний
        тест (baseline_col + postprocess_fn + Optuna) для XGBoost.
        """
        rng = np.random.default_rng(5)
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['baseline'] = y_train.to_numpy() + rng.normal(scale=0.05, size=len(y_train))
        X_valid['baseline'] = y_valid.to_numpy() + rng.normal(scale=0.05, size=len(y_valid))
        feats = ['f0', 'f1', 'f2', 'f3', 'f4']
        shift = 1000.0

        def add_shift(_X, pred):
            return pred + shift

        raw_model, train_pred, valid_pred, infer_pred, best_params = train_regression_model(
            name='xgboost', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=feats, cat_features=[],
            model_settings={'baseline_col': 'baseline'}, n_optuna_trials=2,
            postprocess_fn=add_shift,
        )

        assert np.all(infer_pred > shift / 2)
        assert np.all(train_pred > shift / 2)
        assert np.all(valid_pred > shift / 2)

        mae_after_removing_shift = np.abs((valid_pred - shift) - y_valid.to_numpy()).mean()
        assert mae_after_removing_shift < 0.5


class TestXGBoostRegressorOptuna:
    def test_requires_valid_set(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = XGBoostRegressor(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = XGBoostRegressor(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 'n_estimators' in model.best_params_
        assert model.best_params_['enable_categorical'] is False


class TestXGBoostRegressorCustomMetric:
    def test_callable_reg_metric(self, regression_data):
        from sklearn.metrics import mean_squared_error

        def rmse(y_true, y_pred):
            return float(mean_squared_error(y_true, y_pred) ** 0.5)

        X_train, y_train, X_valid, y_valid = regression_data
        model = XGBoostRegressor(n_optuna_trials=2, model_settings={'reg_metric': (rmse, 'minimize')})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)


class TestXGBoostRegressorCatFeatures:
    def test_categorical_feature_enables_native_support(self, regression_data):
        """До фикса: explicit params + cat_features падал в XGBoost, т.к. enable_categorical
        не форсировался вне Optuna-ветки, а _prep() уже выставляет dtype='category'.
        """
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = XGBoostRegressor(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert model.best_params_['enable_categorical'] is True
        assert_valid_predictions(model, X_valid)


class TestXGBoostClassifierExplicitParams:
    def test_fit_predict_proba(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostClassifier(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_['enable_categorical'] is False

    def test_calibrator_fitted_when_valid_passed(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostClassifier(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_no_calibrator_without_valid(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = XGBoostClassifier(params=FAST_XGB)
        model.fit(X_train, y_train)
        assert model.calibrator_ is None
        assert_valid_proba(model, X_train)


class TestXGBoostClassifierOptuna:
    def test_requires_valid_set(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = XGBoostClassifier(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = XGBoostClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert model.best_params_ is not None


class TestXGBoostMulticlass:
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
        model = XGBoostClassifier(params={**FAST_XGB, 'objective': 'multi:softprob', 'num_class': 3})
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.n_classes_ == 3
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_explicit_params_infers_multiclass_without_objective(self, multiclass_data):
        """XGBoost сам определяет multi:softprob по количеству классов в y, если
        objective не задан явно — как и LightGBM, адаптер это не форсирует в
        explicit-params ветке.
        """
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = XGBoostClassifier(params=FAST_XGB)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.n_classes_ == 3
        assert model.predict_proba(X_valid).shape == (len(X_valid), 3)

    def test_optuna_fit_predict_proba_shape(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = XGBoostClassifier(n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.n_classes_ == 3
        proba = model.predict_proba(X_valid)
        assert proba.shape == (len(X_valid), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert model.best_params_['objective'] == 'multi:softprob'
        assert model.best_params_['num_class'] == 3

    def test_multiclass_calibrators_fitted_with_valid(self, multiclass_data):
        X_train, y_train, X_valid, y_valid = multiclass_data
        model = XGBoostClassifier(params={**FAST_XGB, 'objective': 'multi:softprob', 'num_class': 3})
        model.fit(X_train, y_train, X_valid, y_valid)

        assert model.calibrators_ is not None
        assert len(model.calibrators_) == 3
        assert model.calibrator_ is None

    def test_no_multiclass_calibrators_without_valid(self, multiclass_data):
        X_train, y_train, _, _ = multiclass_data
        model = XGBoostClassifier(params={**FAST_XGB, 'objective': 'multi:softprob', 'num_class': 3})
        model.fit(X_train, y_train)
        assert model.calibrators_ is None
        proba = model.predict_proba(X_train)
        assert proba.shape == (len(X_train), 3)


class TestXGBoostUndersampleMajority:
    def test_default_false_trains_on_full_data(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._xgboost'):
            model = XGBoostClassifier(n_optuna_trials=2)
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert any('undersample_majority=False' in r.message for r in caplog.records)

    def test_enabled_trains_on_subsample(self, classification_data, caplog):
        import logging
        X_train, y_train, X_valid, y_valid = classification_data
        with caplog.at_level(logging.INFO, logger='ml_toolkit.models._xgboost'):
            model = XGBoostClassifier(n_optuna_trials=2, model_settings={'undersample_majority': True})
            model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert not any('undersample_majority=False' in r.message for r in caplog.records)


class TestXGBoostParamSpace:
    def test_custom_param_space_overrides_default(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data

        def my_space(trial):
            return {'n_estimators': trial.suggest_int('n_estimators', 20, 40, step=10)}

        model = XGBoostRegressor(n_optuna_trials=2, model_settings={'param_space': my_space})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 20 <= model.best_params_['n_estimators'] <= 40


class TestXGBoostOptunaPruner:
    def test_unknown_pruner_alias_raises(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = XGBoostRegressor(n_optuna_trials=2, model_settings={'optuna_pruner': 'not_a_pruner'})
        with pytest.raises(ValueError, match='optuna_pruner'):
            model.fit(X_train, y_train, X_valid, y_valid)


class TestXGBoostFunctionalAPI:
    def test_train_regression_model_postprocess_applied_to_infer_pred(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        shift = 1_000_000.0

        def add_shift(_X, pred):
            return pred + shift

        raw_model, train_pred, valid_pred, infer_pred, best_params = train_regression_model(
            name='xgboost', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            model_settings={}, n_optuna_trials=2,
            postprocess_fn=add_shift,
        )
        assert np.all(infer_pred > shift / 2)
        assert np.all(train_pred > shift / 2)
        assert np.all(valid_pred > shift / 2)

    def test_train_classification_model(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        raw_model, train_proba, val_proba, infer_proba, best_params = train_classification_model(
            name='xgboost', X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
            X_inference=X_valid, selected_features=list(X_train.columns), cat_features=[],
            n_optuna_trials=2, model_settings={},
        )
        assert raw_model is not None
        assert np.all((infer_proba >= 0) & (infer_proba <= 1))
