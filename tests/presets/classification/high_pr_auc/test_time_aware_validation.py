"""Тесты для TimeAwareValidationClassifier (ml_toolkit/presets/classification/high_pr_auc/time_aware_validation.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification.high_pr_auc import TimeAwareValidationClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS


def _daily_ts_key(n: int, start: str = '2024-01-01') -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq='D'))


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(1)
    n = 1200
    cols = [f'f{i}' for i in range(5)]
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    y = pd.Series((rng.random(n) < 0.2).astype(int))
    ts_key = _daily_ts_key(n)
    return X, y, ts_key


class TestTimeAwareValidationClassifier:
    def test_fit_predict(self, panel_data):
        X, y, ts_key = panel_data
        model = TimeAwareValidationClassifier(
            n_windows=4, embargo_periods=2, period_unit='D', base_params=BASE_PARAMS,
        )
        model.fit(X, y, ts_key=ts_key)
        proba = model.predict_proba(X.iloc[:10])
        assert proba.shape == (10,)
        assert np.all((proba >= 0) & (proba <= 1))
        assert len(model.estimators_) == 4
        assert len(model.window_scores_) == 4
        assert 0.0 <= model.oof_score_ <= 1.0

    def test_embargo_gap_respected(self, panel_data):
        X, y, ts_key = panel_data
        embargo = 5
        model = TimeAwareValidationClassifier(
            n_windows=3, embargo_periods=embargo, period_unit='D', base_params=BASE_PARAMS,
        )
        model.fit(X, y, ts_key=ts_key)
        for bounds in model.window_bounds_:
            gap = bounds['val_start'] - bounds['train_end']
            assert gap > embargo, f'gap={gap} должен быть строго больше embargo={embargo}'

    def test_windows_expand(self, panel_data):
        """Каждое следующее окно должно иметь train не меньше предыдущего (expanding window)."""
        X, y, ts_key = panel_data
        model = TimeAwareValidationClassifier(
            n_windows=4, embargo_periods=1, period_unit='D', base_params=BASE_PARAMS,
        )
        model.fit(X, y, ts_key=ts_key)
        n_trains = [b['n_train'] for b in model.window_bounds_]
        assert n_trains == sorted(n_trains)

    def test_oof_predictions_cover_non_first_block(self, panel_data):
        X, y, ts_key = panel_data
        model = TimeAwareValidationClassifier(
            n_windows=4, embargo_periods=1, period_unit='D', base_params=BASE_PARAMS,
        )
        model.fit(X, y, ts_key=ts_key)
        total_val = sum(b['n_val'] for b in model.window_bounds_)
        assert len(model.valid_pred_) == total_val

    def test_too_few_periods_raises(self):
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=['f0', 'f1', 'f2'])
        y = pd.Series((rng.random(n) < 0.3).astype(int))
        ts_key = pd.Series(['2024-01-01'] * n)  # один-единственный период
        model = TimeAwareValidationClassifier(n_windows=5, base_params=BASE_PARAMS)
        with pytest.raises(ValueError, match='Недостаточно уникальных периодов'):
            model.fit(X, y, ts_key=pd.to_datetime(ts_key))

    def test_ts_key_length_mismatch_raises(self, panel_data):
        X, y, ts_key = panel_data
        model = TimeAwareValidationClassifier(n_windows=3, base_params=BASE_PARAMS)
        with pytest.raises(ValueError, match='ts_key'):
            model.fit(X, y, ts_key=ts_key.iloc[:-1])

    def test_invalid_n_windows_raises(self):
        with pytest.raises(ValueError, match='n_windows'):
            TimeAwareValidationClassifier(n_windows=1)

    def test_invalid_embargo_raises(self):
        with pytest.raises(ValueError, match='embargo_periods'):
            TimeAwareValidationClassifier(embargo_periods=-1)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match='base'):
            TimeAwareValidationClassifier(base='xgboost')

    @pytest.mark.parametrize('base', ['lightgbm', 'catboost'])
    def test_both_bases(self, panel_data, base):
        X, y, ts_key = panel_data
        model = TimeAwareValidationClassifier(
            n_windows=3, embargo_periods=1, period_unit='D', base=base, base_params=BASE_PARAMS,
        )
        model.fit(X, y, ts_key=ts_key)
        assert model.final_estimator_ is not None
