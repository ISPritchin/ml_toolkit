import numpy as np
import pytest

from ml_toolkit.transformers import (
    autocorr,
    burstiness,
    direction_flag,
    extreme_events,
    growth_quality,
    inactive_streak,
    lag1_diff,
    lag_comparison,
    lifecycle_phase,
    log_slope,
    nonlinearity,
    plateau,
    recency,
    recovery_dynamics,
    regime_change,
    rolling_cv,
    slope,
    streak,
    tenure,
    total_variation,
    trend_consistency,
    zero_clustering,
)
from ml_toolkit.transformers._windowing import compute_position_within_entity


def _get(arrays, suffixes, name):
    return arrays[suffixes.index(name)]


def test_compute_position_within_entity_resets_per_group():
    entity_codes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    assert compute_position_within_entity(entity_codes).tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]


def test_trend_slope_detects_growth_and_decline():
    growing = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pos = compute_position_within_entity(np.zeros(7, dtype=np.int64))

    arrs, sfxs = slope.compute(growing, pos, {"windows": [6, 12]})
    slope_w6 = _get(arrs, sfxs, "w6")
    assert slope_w6[-1] > 0.9

    arrs, sfxs = direction_flag.compute(growing, pos, {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == 1.0

    arrs, sfxs = streak.compute(growing, pos, {})
    assert _get(arrs, sfxs, "up")[-1] == 6.0
    assert _get(arrs, sfxs, "down")[-1] == 0.0

    declining = growing[::-1].copy()
    arrs2, sfxs2 = slope.compute(declining, pos, {"windows": [6, 12]})
    assert _get(arrs2, sfxs2, "w6")[-1] < -0.9

    arrs3, sfxs3 = direction_flag.compute(declining, pos, {"windows": [6]})
    assert _get(arrs3, sfxs3, "w6")[-1] == -1.0

    arrs4, sfxs4 = streak.compute(declining, pos, {})
    assert _get(arrs4, sfxs4, "down")[-1] == 6.0


def test_volatility_cv_higher_for_spiky_series():
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    stable = np.full(12, 100.0)
    spiky = np.array([100, 100, 100, 100, 100, 800, 100, 100, 100, 100, 100, 100], dtype=float)

    arrs_s, sfxs = rolling_cv.compute(stable, pos, {"windows": [12]})
    arrs_p, _ = rolling_cv.compute(spiky, pos, {"windows": [12]})
    assert _get(arrs_s, sfxs, "w12")[-1] == 0.0
    assert _get(arrs_p, sfxs, "w12")[-1] > 0.0


def test_tenure_activity_tracks_dormancy():
    values = np.array([10, 10, 10, 10, 10, 10, 0, 0, 0, 0], dtype=float)
    pos = compute_position_within_entity(np.zeros(10, dtype=np.int64))

    arrs, sfxs = tenure.compute(values, pos, {})
    assert _get(arrs, sfxs, "first_active_flag")[0] == 1.0
    assert _get(arrs, sfxs, "tenure_months")[5] == 6.0

    arrs, sfxs = recency.compute(values, pos, {})
    assert _get(arrs, sfxs, "recency_gap")[-1] == 4.0

    arrs, sfxs = inactive_streak.compute(values, pos, {})
    assert _get(arrs, sfxs, "current")[-1] == 4.0
    assert _get(arrs, sfxs, "max")[-1] == 4.0


def test_dynamics_diff_and_log_diff_signs():
    values = np.array([100.0, 50.0, 200.0])
    pos = compute_position_within_entity(np.zeros(3, dtype=np.int64))
    arrs, sfxs = lag1_diff.compute(values, pos, {})
    assert _get(arrs, sfxs, "diff")[1] == -50.0
    assert _get(arrs, sfxs, "diff")[2] == 150.0
    assert _get(arrs, sfxs, "log_diff")[1] < 0
    assert _get(arrs, sfxs, "log_diff")[2] > 0


def test_plateau_constant_series_is_fully_flat():
    values = np.full(12, 100.0)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = plateau.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "flat_share_w12")[-1] == 1.0
    assert _get(arrs, sfxs, "current_flat_streak")[-1] == 11.0


def test_extreme_events_spike_detected():
    values = np.full(12, 10.0)
    values[6] = 1000.0
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = extreme_events.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "spike_count_w12")[-1] >= 1.0


def test_smoothness_constant_series_zero_variation():
    values = np.full(12, 50.0)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = total_variation.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "w12")[-1] == 0.0


def test_lag_comparison_yoy():
    values = np.array([10.0] * 12 + [20.0] * 12)
    pos = compute_position_within_entity(np.zeros(24, dtype=np.int64))
    arrs, sfxs = lag_comparison.compute(values, pos, {})
    assert abs(_get(arrs, sfxs, "lag12_ratio")[-1] - 1.0) < 1e-6


def test_nonlinearity_u_shaped():
    values = np.array([10.0, 8.0, 4.0, 2.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 8.0, 10.0])
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = nonlinearity.compute(values, pos, {"windows": [6, 12]})
    assert _get(arrs, sfxs, "quad_proxy_w12")[-1] > 0.0


def test_regime_change_detects_level_shift():
    values = np.array([0.0] * 3 + [100.0] * 9)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = regime_change.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "magnitude_w12")[-1] > 2.0
    assert _get(arrs, sfxs, "flag_w12")[-1] == 1.0


def test_autocorrelation_lag1_positive_for_trend():
    values = np.arange(1.0, 25.0)
    pos = compute_position_within_entity(np.zeros(24, dtype=np.int64))
    arrs, sfxs = autocorr.compute(values, pos, {})
    assert _get(arrs, sfxs, "lag1")[-1] > 0.9


def test_autocorrelation_lag1_negative_for_oscillating():
    n = 20
    values = np.array([100.0 if i % 2 == 0 else 0.0 for i in range(n)])
    pos = compute_position_within_entity(np.zeros(n, dtype=np.int64))
    arrs, sfxs = autocorr.compute(values, pos, {})
    assert _get(arrs, sfxs, "lag1")[-1] < 0


def test_growth_quality_single_spike_is_inorganic():
    values = np.array([10.0] * 6 + [1000.0] + [10.0] * 5)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = growth_quality.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "best_share_w12")[-1] > 0.9
    assert _get(arrs, sfxs, "organic_w12")[-1] < 0.1


def test_log_growth_slope_positive_for_exponential_growth():
    values = np.array([10.0 * (1.1 ** i) for i in range(12)])
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = log_slope.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "w12")[-1] > 0.0


def test_lifecycle_new_peak_detection():
    values = np.arange(1.0, 13.0)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = lifecycle_phase.compute(values, pos, {"windows": [12]})
    # каждое значение — новый максимум в монотонно растущем ряду, включая позицию 0
    assert _get(arrs, sfxs, "is_new_peak").sum() == 12.0


def test_recovery_dynamics_completeness():
    values = np.array([100.0, 80.0, 50.0, 20.0, 50.0, 80.0, 100.0] + [100.0] * 5)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = recovery_dynamics.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "completeness_w12")[-1] > 0.95


def test_zero_clustering_max_run_detects_consecutive_zeros():
    values = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = zero_clustering.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "max_zero_run_w12")[-1] == 5.0


def test_trend_consistency_high_for_clean_trend():
    values = np.arange(1.0, 13.0)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = trend_consistency.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "dir_consistency_w12")[-1] > 0.9


def test_burstiness_calm_share_all_zeros():
    values = np.zeros(12, dtype=float)
    pos = compute_position_within_entity(np.zeros(12, dtype=np.int64))
    arrs, sfxs = burstiness.compute(values, pos, {"windows": [12]})
    assert _get(arrs, sfxs, "calm_share_w12")[-1] == 1.0
