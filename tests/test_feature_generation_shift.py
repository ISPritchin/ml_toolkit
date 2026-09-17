"""Tests for the `shift` reserved param key in ml_toolkit.feature_generation.

`shift` carries an already-computed feature value from one row of an entity onto
another row of the SAME entity, with zero changes to any transformer kernel (same
orchestration-layer trick as `segment`). shift>0 moves a value from the past forward
onto a later row (safe: the source value never used anything past its own original
row). shift<0 moves a value from the future backward onto an earlier row (a
deliberate, documented leakage risk -- tested here for correctness, not endorsed).
"""

import math

import polars as pl
import pytest

from ml_toolkit.feature_generation import generate_feature_groups_df

WINDOW_MEAN_W1 = {'window_mean': {'windows': [1]}}


def _two_entity_df(entity_col: str = 'entity_id') -> pl.DataFrame:
    """Entity 1: growing 10..120 (step 10). Entity 2: declining 120..10 (step -10)."""
    months = list(range(1, 13))
    growing = [10.0 * i for i in months]
    declining = [10.0 * (13 - i) for i in months]
    return pl.DataFrame({
        entity_col: [1] * len(months) + [2] * len(months),
        'ts_key': months + months,
        'value': growing + declining,
    })


def _col_values(out: pl.DataFrame, entity_id: int, col: str) -> list[float]:
    return out.filter(pl.col('entity_id') == entity_id).sort('ts_key')[col].to_list()


def test_shift_positive_carries_past_value_forward_within_entity():
    df = _two_entity_df()
    out = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[('value', {'window_mean': {'windows': [1], 'shift': 3}})],
    )
    col = 'value__window_mean__w1__shift3'
    assert col in out.columns

    unshifted_e1 = [10.0 * i for i in range(1, 13)]  # 10..120
    expected_e1 = [math.nan, math.nan, math.nan] + unshifted_e1[:-3]
    got_e1 = _col_values(out, 1, col)
    for g, e in zip(got_e1, expected_e1, strict=True):
        if math.isnan(e):
            assert math.isnan(g)
        else:
            assert g == pytest.approx(e)


def test_shift_positive_does_not_leak_across_entity_boundary():
    # entity 2 starts right after entity 1 in the sorted base parquet -- a buggy
    # implementation using raw array indices without entity-boundary checks would
    # pull entity 1's tail values into entity 2's first few rows.
    df = _two_entity_df()
    out = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[('value', {'window_mean': {'windows': [1], 'shift': 3}})],
    )
    col = 'value__window_mean__w1__shift3'
    got_e2 = _col_values(out, 2, col)
    # first 3 rows of entity 2 must be NaN (not entity 1's last values 100/110/120)
    assert all(math.isnan(v) for v in got_e2[:3])
    unshifted_e2 = [10.0 * (13 - i) for i in range(1, 13)]  # 120..10
    for g, e in zip(got_e2[3:], unshifted_e2[:-3], strict=True):
        assert g == pytest.approx(e)


def test_shift_negative_lead_carries_future_value_backward_within_entity():
    df = _two_entity_df()
    out = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[('value', {'window_mean': {'windows': [1], 'shift': -3}})],
    )
    col = 'value__window_mean__w1__lead3'
    assert col in out.columns

    unshifted_e1 = [10.0 * i for i in range(1, 13)]
    expected_e1 = unshifted_e1[3:] + [math.nan, math.nan, math.nan]
    got_e1 = _col_values(out, 1, col)
    for g, e in zip(got_e1, expected_e1, strict=True):
        if math.isnan(e):
            assert math.isnan(g)
        else:
            assert g == pytest.approx(e)


def test_shift_zero_is_a_no_op_matching_unshifted_output():
    df = _two_entity_df()
    out_plain = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key', feature_spec=[('value', WINDOW_MEAN_W1)],
    )
    out_shift0 = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[('value', {'window_mean': {'windows': [1], 'shift': 0}})],
    )
    assert 'value__window_mean__w1' in out_plain.columns
    assert 'value__window_mean__w1' in out_shift0.columns
    assert out_plain['value__window_mean__w1'].to_list() == out_shift0['value__window_mean__w1'].to_list()


def test_fill_nan_covers_shift_created_boundary_nan():
    df = _two_entity_df()
    out = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[('value', {'window_mean': {'windows': [1], 'shift': 3, 'fill_nan': -999.0}})],
    )
    col = 'value__window_mean__w1__shift3'
    got_e1 = _col_values(out, 1, col)
    assert got_e1[:3] == [-999.0, -999.0, -999.0]
    assert not any(math.isnan(v) for v in got_e1)


def test_same_transformer_with_and_without_shift_are_distinct_candidates_not_a_conflict():
    df = _two_entity_df()
    out = generate_feature_groups_df(
        df, entity_column_name='entity_id', ts_column_name='ts_key',
        feature_spec=[
            ('value', {'window_mean': {'windows': [1]}}),
            ('value', {'window_mean': {'windows': [1], 'shift': 3}}),
        ],
    )
    assert 'value__window_mean__w1' in out.columns
    assert 'value__window_mean__w1__shift3' in out.columns
    # the two columns actually differ (shift really did something, not silently deduped away)
    assert out['value__window_mean__w1'].to_list() != out['value__window_mean__w1__shift3'].fill_nan(-1).to_list()
