"""Tests for ml_toolkit/transformers/_windowing.py's dilation support.

dilation=1 (the default) must reproduce every primitive's pre-existing contiguous-window
behaviour exactly; dilation>1 must read strided points [row_idx, row_idx-d, ..., row_idx-(w-1)*d].
"""

import numpy as np
import pytest

from ml_toolkit.transformers._windowing import (
    compute_window_mean_and_std,
    compute_window_min_and_max,
    compute_window_sorted_buffer,
    compute_window_sum,
    fit_linear_trend_slope,
    resolve_window_size,
    sorted_median,
)


class TestResolveWindowSize:
    def test_dilation_1_matches_old_contiguous_behaviour(self):
        assert resolve_window_size(0, 6) == 1
        assert resolve_window_size(5, 6) == 6
        assert resolve_window_size(100, 6) == 6
        assert resolve_window_size(0, 6, dilation=1) == 1
        assert resolve_window_size(5, 6, dilation=1) == 6

    def test_dilation_2_halves_available_points(self):
        # position 5 -> 6 rows of history (0..5); every 2nd row reachable -> 3 points (0,2,4 back from 5 -> idx 5,3,1)
        assert resolve_window_size(5, 6, dilation=2) == 3
        # position 4 -> 5 rows of history; dilation=2 -> floor(4/2)+1 = 3 points reachable
        assert resolve_window_size(4, 6, dilation=2) == 3
        # requested window smaller than what's available -> still capped by requested
        assert resolve_window_size(100, 3, dilation=2) == 3

    def test_dilation_3_at_entity_start(self):
        assert resolve_window_size(0, 6, dilation=3) == 1
        assert resolve_window_size(2, 6, dilation=3) == 1
        assert resolve_window_size(3, 6, dilation=3) == 2


class TestFitLinearTrendSlope:
    def test_dilation_1_matches_known_example(self):
        # from slope.py's own docstring example
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert fit_linear_trend_slope(values, 4, 5) == pytest.approx(10.0)
        assert fit_linear_trend_slope(values, 4, 5, dilation=1) == pytest.approx(10.0)

    def test_dilation_2_reads_every_other_point(self):
        # values at positions 0..8; row_idx=8, window_size=3, dilation=2 -> reads idx 4,6,8 -> [50,70,90]
        values = np.array([10.0, 999.0, 30.0, 999.0, 50.0, 999.0, 70.0, 999.0, 90.0])
        slope = fit_linear_trend_slope(values, 8, 3, dilation=2)
        # perfectly linear (50,70,90), step of 1 index unit -> slope 20/point
        assert slope == pytest.approx(20.0)

    def test_window_size_below_2_returns_zero_regardless_of_dilation(self):
        values = np.array([1.0, 2.0, 3.0])
        assert fit_linear_trend_slope(values, 2, 1, dilation=5) == 0.0


class TestComputeWindowMeanAndStd:
    def test_dilation_1_matches_known_example(self):
        # from rolling_std.py's test: [10,10,10,10,10,40] w=6 -> mean=15, std=sqrt(125)
        values = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 40.0])
        mean, std = compute_window_mean_and_std(values, 5, 6)
        assert mean == pytest.approx(15.0)
        assert std == pytest.approx(125.0**0.5)

    def test_dilation_2_reads_strided_points(self):
        # row_idx=7, ws=3, dilation=2 -> idx 3,5,7 -> [40,999,60] ignoring the interleaved noise
        values = np.array([0, 0, 0, 40.0, 999.0, 999.0, 999.0, 60.0])
        values[5] = 50.0  # idx 5 participates
        mean, _ = compute_window_mean_and_std(values, 7, 3, dilation=2)
        assert mean == pytest.approx((40.0 + 50.0 + 60.0) / 3)


class TestComputeWindowSum:
    def test_dilation_1_matches_plain_sum(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_window_sum(values, 3, 4) == pytest.approx(10.0)

    def test_dilation_2(self):
        values = np.array([1.0, 999.0, 3.0, 999.0, 5.0])
        # row_idx=4, ws=3, dilation=2 -> idx 0,2,4 -> [1,3,5]
        assert compute_window_sum(values, 4, 3, dilation=2) == pytest.approx(9.0)


class TestComputeWindowMinAndMax:
    def test_dilation_1_matches_plain_min_max(self):
        values = np.array([5.0, 1.0, 9.0, 3.0])
        lo, hi = compute_window_min_and_max(values, 3, 4)
        assert (lo, hi) == (1.0, 9.0)

    def test_dilation_2_ignores_interleaved_extremes(self):
        # interleaved values at odd indices are the extremes but must be skipped by dilation=2
        values = np.array([5.0, -100.0, 9.0, 200.0, 3.0])
        lo, hi = compute_window_min_and_max(values, 4, 3, dilation=2)
        # row_idx=4, ws=3, dilation=2 -> idx 0,2,4 -> [5,9,3]
        assert (lo, hi) == (3.0, 9.0)


class TestComputeWindowSortedBufferAndMedian:
    def test_dilation_2_median_over_strided_points(self):
        values = np.array([10.0, 999.0, 30.0, 999.0, 20.0])
        # row_idx=4, ws=3, dilation=2 -> idx 0,2,4 -> [10,30,20] -> sorted [10,20,30] -> median 20
        buf = compute_window_sorted_buffer(values, 4, 3, dilation=2)
        assert sorted_median(buf, 3) == pytest.approx(20.0)
