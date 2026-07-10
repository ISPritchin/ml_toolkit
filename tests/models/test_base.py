"""Тесты для контракта BaseModel (ml_toolkit/models/_base.py).

Используется минимальный конкретный подкласс — сами адаптеры (CatBoost, LightGBM,
...) наследуют этот контракт, но их тестировать здесь избыточно (это делают
test_catboost.py/test_lightgbm.py/...). Здесь проверяется только то, что даёт
BaseModel сам по себе: конвертация входа, состояние "не обучена", resolve_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml_toolkit.models._base import BaseModel, _to_numpy, _to_pandas


class _DummyModel(BaseModel):
    """Тривиальный адаптер: predict возвращает сумму признаков + 1 (детерминированно)."""

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, selected_features=None, cat_features=None):
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = cat_features or []
        self._model = 'fitted'
        self.train_pred_ = X_train[self.selected_features_].sum(axis=1).to_numpy()
        if X_valid is not None:
            self.valid_pred_ = X_valid[self.selected_features_].sum(axis=1).to_numpy()
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.selected_features_].sum(axis=1).to_numpy()


class TestCheckFitted:
    def test_predict_before_fit_raises(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0, 2.0]})
        with pytest.raises(RuntimeError, match='не обучена'):
            model.predict(X)

    def test_predict_after_fit_works(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        y = pd.Series([1.0, 2.0])
        model.fit(X, y)
        pred = model.predict(X)
        np.testing.assert_allclose(pred, [4.0, 6.0])


class TestPredictProbaNotImplemented:
    def test_regressor_like_model_has_no_predict_proba(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0, 2.0]})
        y = pd.Series([1.0, 2.0])
        model.fit(X, y)
        with pytest.raises(NotImplementedError):
            model.predict_proba(X)


class TestResolveFeatures:
    def test_none_selected_features_uses_all_columns(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
        y = pd.Series([1.0])
        model.fit(X, y)
        assert model.selected_features_ == ['a', 'b', 'c']

    def test_explicit_selected_features_subset(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0], 'b': [2.0], 'c': [3.0]})
        y = pd.Series([1.0])
        model.fit(X, y, selected_features=['a', 'c'])
        assert model.selected_features_ == ['a', 'c']


class TestPolarsInput:
    def test_fit_accepts_polars_dataframe(self):
        model = _DummyModel()
        X = pl.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        y = pl.Series('y', [1.0, 2.0, 3.0])
        model.fit(X, y)
        assert model.selected_features_ == ['a', 'b']
        np.testing.assert_allclose(model.train_pred_, [5.0, 7.0, 9.0])

    def test_predict_accepts_polars_dataframe(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        y = pd.Series([1.0, 2.0])
        model.fit(X, y)
        X_new = pl.DataFrame({'a': [10.0], 'b': [20.0]})
        pred = model.predict(X_new)
        np.testing.assert_allclose(pred, [30.0])

    def test_valid_split_populates_valid_pred(self):
        model = _DummyModel()
        X_train = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        y_train = pd.Series([1.0, 2.0])
        X_valid = pd.DataFrame({'a': [5.0], 'b': [6.0]})
        y_valid = pd.Series([7.0])
        model.fit(X_train, y_train, X_valid, y_valid)
        np.testing.assert_allclose(model.valid_pred_, [11.0])

    def test_no_valid_split_leaves_valid_pred_none(self):
        model = _DummyModel()
        X = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        y = pd.Series([1.0])
        model.fit(X, y)
        assert model.valid_pred_ is None


class TestConversionHelpers:
    def test_to_pandas_passthrough(self):
        df = pd.DataFrame({'a': [1, 2]})
        assert _to_pandas(df) is df

    def test_to_pandas_converts_polars(self):
        df = pl.DataFrame({'a': [1, 2]})
        out = _to_pandas(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out['a']) == [1, 2]

    def test_to_pandas_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            _to_pandas([1, 2, 3])

    def test_to_numpy_from_pandas_series(self):
        s = pd.Series([1, 2, 3])
        out = _to_numpy(s)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1, 2, 3])

    def test_to_numpy_from_polars_series(self):
        s = pl.Series('y', [1, 2, 3])
        out = _to_numpy(s)
        np.testing.assert_array_equal(out, [1, 2, 3])

    def test_to_numpy_from_ndarray_passthrough(self):
        arr = np.array([1, 2, 3])
        assert _to_numpy(arr) is arr

    def test_to_numpy_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            _to_numpy('not a series')
