"""Тесты для LinearRegressor/LinearClassifier (ml_toolkit/models/_linear.py)."""

from __future__ import annotations

import pytest

from ml_toolkit.models._linear import LinearClassifier, LinearRegressor
from tests.models.conftest import MULTI_CAT_FEATURES, assert_valid_predictions, assert_valid_proba


class TestLinearRegressorTypes:
    @pytest.mark.parametrize('name', ['ridge', 'elasticnet', 'huber', 'quantile'])
    def test_fit_predict_explicit_params(self, name, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={}, model_settings={'name': name})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_explicit_params_beyond_tuned_keys_are_honored(self, regression_data):
        """Регрессия бага: _make_regressor читал только alpha/l1_ratio/epsilon/power через
        params.get(...) — любой другой валидный sklearn-параметр (fit_intercept, max_iter)
        молча отбрасывался вместо применения.
        """
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={'fit_intercept': False}, model_settings={'name': 'ridge'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model._model.fit_intercept is False
        assert model.best_params_['fit_intercept'] is False

    def test_quantile_param_was_previously_hardcoded_to_median(self, regression_data):
        """QuantileRegressor: quantile был всегда захардкожен в 0.5 в конструкторе адаптера,
        независимо от того, что передано в params — единственный параметр, ради которого
        вообще существует QuantileRegressor, был недоступен для настройки.
        """
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={'quantile': 0.9}, model_settings={'name': 'quantile'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model._model.quantile == 0.9
        assert_valid_predictions(model, X_valid)

    def test_tweedie_requires_positive_target(self, positive_regression_data):
        X_train, y_train, X_valid, y_valid = positive_regression_data
        model = LinearRegressor(params={}, model_settings={'name': 'tweedie'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)

    def test_bayesian_ridge_skips_optuna_even_without_params(self, regression_data):
        """bayesian_ridge самонастраивается — не требует X_valid даже при params=None."""
        X_train, y_train, _, _ = regression_data
        model = LinearRegressor(model_settings={'name': 'bayesian_ridge'})
        model.fit(X_train, y_train)
        assert_valid_predictions(model, X_train)

    def test_unknown_name_raises(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = LinearRegressor(params={}, model_settings={'name': 'not_a_real_model'})
        with pytest.raises(ValueError, match='Unknown linear regression type'):
            model.fit(X_train, y_train)

    def test_default_name_is_ridge(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={})
        model.fit(X_train, y_train, X_valid, y_valid)
        from sklearn.linear_model import Ridge
        assert isinstance(model._model, Ridge)

    def test_requires_valid_for_optuna(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = LinearRegressor(n_optuna_trials=2, model_settings={'name': 'ridge'})
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(n_optuna_trials=3, model_settings={'name': 'ridge'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_predictions(model, X_valid)
        assert 'alpha' in model.best_params_

    def test_categorical_features_excluded(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        X_train['cat_col'] = 'x'
        X_valid['cat_col'] = 'x'
        model = LinearRegressor(params={}, model_settings={'name': 'ridge'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert 'cat_col' not in model._num_feats_
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_excluded_by_default(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = LinearRegressor(params={}, model_settings={'name': 'ridge'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_predictions(model, X_valid)

    def test_multiple_categorical_features_included_with_onehot(self, regression_data_multi_cat):
        X_train, y_train, X_valid, y_valid = regression_data_multi_cat
        model = LinearRegressor(params={}, model_settings={'name': 'ridge', 'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
            assert any(f.startswith(f'{col}_') for f in model._num_feats_)
        assert_valid_predictions(model, X_valid)

    def test_no_hardcoded_baseline_col_default(self, regression_data):
        """model_settings без 'baseline_col' не должен подмешивать никакой столбец
        по умолчанию — ml_toolkit не хардкодит имена колонок бизнес-задач.
        """
        X_train, y_train, X_valid, y_valid = regression_data
        model = LinearRegressor(params={}, model_settings={'name': 'ridge'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'fee_nds_amount' not in model._num_feats_
        assert set(model._num_feats_) == set(X_train.columns)


class TestLinearClassifier:
    def test_fit_predict_proba_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_calibrator_fitted_with_valid(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.calibrator_ is not None

    def test_explicit_class_weight_does_not_raise(self, classification_data):
        """Регрессия бага: LogisticRegression(**self.params, class_weight='balanced')
        падал с TypeError('multiple values for keyword argument'), если params уже
        содержал 'class_weight' — естественный сценарий (например, params скопирован
        из best_params_ предыдущего запуска).
        """
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0, 'class_weight': 'balanced'})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)

    def test_explicit_class_weight_none_is_honored(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0, 'class_weight': None})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] is None
        assert model._model.class_weight is None

    def test_default_is_balanced_when_not_specified(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.best_params_['class_weight'] == 'balanced'

    def test_requires_valid_for_optuna(self, classification_data):
        X_train, y_train, _, _ = classification_data
        model = LinearClassifier(n_optuna_trials=2)
        with pytest.raises(ValueError, match='X_valid'):
            model.fit(X_train, y_train)

    def test_fit_with_optuna(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LinearClassifier(n_optuna_trials=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert 'C' in model.best_params_

    def test_categorical_features_excluded(self, classification_data_with_cat):
        X_train, y_train, X_valid, y_valid = classification_data_with_cat
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=['cat_col'])
        assert 'cat_col' not in model._num_feats_
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_excluded_by_default(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = LinearClassifier(params={'C': 1.0})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
        assert_valid_proba(model, X_valid)

    def test_multiple_categorical_features_included_with_onehot(self, classification_data_multi_cat):
        X_train, y_train, X_valid, y_valid = classification_data_multi_cat
        model = LinearClassifier(params={'C': 1.0}, model_settings={'cat_encoder': 'onehot'})
        model.fit(X_train, y_train, X_valid, y_valid, cat_features=MULTI_CAT_FEATURES)
        for col in MULTI_CAT_FEATURES:
            assert col not in model._num_feats_
            assert any(f.startswith(f'{col}_') for f in model._num_feats_)
        assert_valid_proba(model, X_valid)

