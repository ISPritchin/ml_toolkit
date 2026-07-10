"""Тесты для WeightedBaggingByRecency
(ml_toolkit/presets/classification/high_pr_auc/weighted_bagging_recency.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification._time_utils import compute_periods
from ml_toolkit.presets.classification.high_pr_auc import WeightedBaggingByRecency
from ml_toolkit.presets.classification.high_pr_auc.weighted_bagging_recency import (
    _recency_weights,
)
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


def _monthly_ts_key(n: int, n_months: int = 12) -> pd.Series:
    """Даты, равномерно раскиданные по n_months месяцам — старые вперемешку со свежими."""
    dates = pd.date_range('2025-01-01', periods=n_months, freq='MS')
    idx = np.arange(n) % n_months
    return pd.Series(dates[idx])


class TestWeightedBaggingByRecency:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        ts_key = _monthly_ts_key(len(X_train))
        model = WeightedBaggingByRecency(
            n_estimators=5, halflife_periods=3, base_params=BASE_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid, ts_key=ts_key)
        assert_valid_proba(model, X_valid)
        assert len(model.estimators_) == 5
        assert len(model.estimator_scores_) == 5

    def test_ts_key_length_mismatch_raises(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        bad_ts_key = _monthly_ts_key(len(X_train) - 1)
        model = WeightedBaggingByRecency(n_estimators=2, base_params=BASE_PARAMS)
        with pytest.raises(ValueError, match='ts_key'):
            model.fit(X_train, y_train, X_valid, y_valid, ts_key=bad_ts_key)

    def test_invalid_halflife_raises(self):
        with pytest.raises(ValueError, match='halflife_periods'):
            WeightedBaggingByRecency(halflife_periods=0)

    def test_invalid_sample_frac_raises(self):
        with pytest.raises(ValueError, match='sample_frac'):
            WeightedBaggingByRecency(sample_frac=0.0)

    def test_estimators_are_diverse(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        ts_key = _monthly_ts_key(len(X_train))
        model = WeightedBaggingByRecency(
            n_estimators=5, halflife_periods=3, base_params=BASE_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid, ts_key=ts_key)

        X_va_feats = X_valid[model.selected_features_]
        raw_scores = [model._predict_one(est, X_va_feats) for est in model.estimators_]

        n = len(raw_scores)
        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(raw_scores[i], raw_scores[j]), (
                    f'estimators {i} and {j} produce identical predictions — no diversity'
                )

    def test_tuning_sample_independent_from_first_estimator(self, binary_data):
        """Регрессионный тест на паттерн из EasyEnsembleClassifier: подвыборка
        тюнинга (rng0) не должна совпадать с подвыборкой estimator'а #0.
        """
        X_train, y_train, X_valid, y_valid = binary_data
        ts_key = _monthly_ts_key(len(X_train))
        model = WeightedBaggingByRecency(
            n_estimators=3, halflife_periods=3, n_optuna_trials=3, base_params=None,
        )
        model.fit(X_train, y_train, X_valid, y_valid, ts_key=ts_key)

        n_train = len(X_train)
        periods = compute_periods(ts_key.reset_index(drop=True), model.period_unit)
        weights = _recency_weights(periods, model.halflife_periods)
        n_sample = max(1, int(round(model.sample_frac * n_train)))

        tune_seq, est0_seq = np.random.SeedSequence(model.random_seed).spawn(model.n_estimators + 1)[:2]
        tune_idx = np.random.default_rng(tune_seq).choice(n_train, size=n_sample, replace=True, p=weights)
        est0_idx = np.random.default_rng(est0_seq).choice(n_train, size=n_sample, replace=True, p=weights)
        assert not np.array_equal(tune_idx, est0_idx)


class TestRecencyWeights:
    def test_weights_sum_to_one(self):
        periods = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        w = _recency_weights(periods, halflife_periods=2.0)
        assert w.sum() == pytest.approx(1.0)

    def test_most_recent_period_has_highest_weight(self):
        periods = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        w = _recency_weights(periods, halflife_periods=2.0)
        assert np.argmax(w) == np.argmax(periods)

    def test_halflife_halves_weight_ratio(self):
        periods = np.array([0.0, 10.0], dtype=np.float64)
        w = _recency_weights(periods, halflife_periods=10.0)
        # age=10 у первой строки, halflife=10 -> вес ровно вдвое меньше самой свежей
        assert w[0] / w[1] == pytest.approx(0.5)

    def test_compute_periods_datetime(self):
        ts = pd.Series(pd.to_datetime(['2025-01-15', '2025-02-01', '2025-03-20']))
        periods = compute_periods(ts, 'M')
        assert periods[1] - periods[0] == pytest.approx(1.0)
        assert periods[2] - periods[1] == pytest.approx(1.0)

    def test_compute_periods_numeric_passthrough(self):
        ts = pd.Series([1, 3, 5])
        periods = compute_periods(ts, 'M')
        np.testing.assert_array_equal(periods, [1.0, 3.0, 5.0])
