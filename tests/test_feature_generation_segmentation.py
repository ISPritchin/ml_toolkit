"""Тесты сегментации (`segment:`/`segments:`) в `generate_feature_groups`.

Проверяется путь: именованный конфиг сегментации в пресете -> подмена
position_within_entity на position_within_segment для сегментированного
трансформера -> NaN в строках вне сегмента -> суффикс колонки кодирует конфиг
-> Phase B (корреляционный фильтр) не падает на NaN.
"""

import math

import numpy as np
import polars as pl
import pytest

from ml_toolkit.feature_generation import generate_feature_groups

# Пример из обсуждения: два периода активности, разрыв из >=2 нулей подряд.
_WORKED_VALUES = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 4, 3, 5, 2]


def _single_entity_df(entity_col: str = 'entity_id', values: list | None = None) -> pl.DataFrame:
    values = values if values is not None else _WORKED_VALUES
    return pl.DataFrame({
        entity_col: [1] * len(values),
        'ts_key': list(range(1, len(values) + 1)),
        'value': [float(v) for v in values],
    })


def test_segmented_window_mean_nan_in_gap_and_correct_values(tmp_path):
    df = _single_entity_df()
    preset = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3], 'segment': 'short_gap'},
    }

    accepted = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset)],
        out_path=tmp_path / 'out.parquet',
    )

    col = 'value__window_mean__w3__seg-zerogap2'
    assert accepted == [col]

    out = pl.read_parquet(tmp_path / 'out.parquet')
    got = out[col].to_numpy()

    expected = [
        1.0, 1.5, 2.0, 3.0, 4.0, 3.0, 5.0 / 3,
        np.nan, np.nan, np.nan, np.nan,
        4.0, 3.5, 4.0, 10.0 / 3,
    ]
    for i, exp in enumerate(expected):
        if math.isnan(exp):
            assert math.isnan(got[i]), f'row {i}: expected NaN, got {got[i]}'
        else:
            assert got[i] == pytest.approx(exp, abs=1e-4), f'row {i}: {got[i]} != {exp}'


def test_segment_gap_indicator_marks_gap_rows(tmp_path):
    df = _single_entity_df()
    preset = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'segment_gap': {'segment': 'short_gap'},
    }

    generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset)],
        out_path=tmp_path / 'out.parquet',
    )

    out = pl.read_parquet(tmp_path / 'out.parquet')
    got = out['value__segment_gap__seg-zerogap2'].to_numpy()
    expected = [0.0] * 7 + [1.0] * 4 + [0.0] * 4
    assert got.tolist() == pytest.approx(expected)


def test_same_transformer_different_segments_coexist(tmp_path):
    df = _single_entity_df()
    preset = {
        'segments': {
            'gap1': {'strategy': 'zero_gap', 'gap_threshold': 1},
            'gap2': {'strategy': 'zero_gap', 'gap_threshold': 2},
        },
        'window_mean': {'windows': [3], 'segment': 'gap1'},
    }
    # Второй вариант того же трансформера с другим сегментом — через отдельную
    # группу feature_spec, называющую ту же колонку.
    preset2 = {
        'segments': {'gap2': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3], 'segment': 'gap2'},
    }

    accepted = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset), ('value', preset2)],
        out_path=tmp_path / 'out.parquet',
    )

    assert set(accepted) == {
        'value__window_mean__w3__seg-zerogap1',
        'value__window_mean__w3__seg-zerogap2',
    }
    out = pl.read_parquet(tmp_path / 'out.parquet')
    # gap_threshold=1 экономнее на grace period -> должен исключать больше строк NaN,
    # чем gap_threshold=2 (более щедрый разрыв => меньше исключённых точек).
    n_nan_gap1 = out['value__window_mean__w3__seg-zerogap1'].is_nan().sum()
    n_nan_gap2 = out['value__window_mean__w3__seg-zerogap2'].is_nan().sum()
    assert n_nan_gap1 > n_nan_gap2


def test_same_transformer_same_segment_conflicting_other_params_raises(tmp_path):
    df = _single_entity_df()
    preset_a = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3], 'segment': 'short_gap'},
    }
    preset_b = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [6], 'segment': 'short_gap'},  # другие windows - конфликт
    }

    with pytest.raises(ValueError, match='разными параметрами'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[('value', preset_a), ('value', preset_b)],
            out_path=tmp_path / 'out.parquet',
        )


def test_unsegmented_transformer_unaffected_by_segments_section(tmp_path):
    # Наличие 'segments:' в пресете не должно ломать трансформеры без 'segment'.
    df = _single_entity_df()
    preset = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3]},  # без ссылки на segment
    }
    accepted = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset)],
        out_path=tmp_path / 'out.parquet',
    )
    assert accepted == ['value__window_mean__w3']
    out = pl.read_parquet(tmp_path / 'out.parquet')
    assert out['value__window_mean__w3'].is_nan().sum() == 0


def test_unknown_named_segment_reference_raises(tmp_path):
    df = _single_entity_df()
    preset = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3], 'segment': 'does_not_exist'},
    }
    with pytest.raises(ValueError, match='does_not_exist'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[('value', preset)],
            out_path=tmp_path / 'out.parquet',
        )


def test_mask_strategy_end_to_end(tmp_path):
    values = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0]
    is_active = [True, True, True, True, False, False]
    df = pl.DataFrame({
        'entity_id': [1] * len(values),
        'ts_key': list(range(1, len(values) + 1)),
        'value': values,
        'is_active': is_active,
    })
    preset = {
        'segments': {
            'mask_gap': {'strategy': 'mask', 'gap_threshold': 1, 'mask_column': 'is_active'},
        },
        'window_mean': {'windows': [2], 'segment': 'mask_gap'},
    }

    generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset)],
        out_path=tmp_path / 'out.parquet',
    )

    out = pl.read_parquet(tmp_path / 'out.parquet')
    got = out['value__window_mean__w2__seg-mask-is_activeg1'].to_numpy()
    # is_active=[T,T,T,T,F,F] -> is_trigger=[F,F,F,F,T,T], gap_threshold=1:
    # idx4 - grace period (в сегменте), idx5 - исключена (NaN).
    assert not math.isnan(got[4])
    assert math.isnan(got[5])


def test_correlation_filter_does_not_crash_on_nan_candidates(tmp_path):
    df = _single_entity_df()
    preset = {
        'segments': {'short_gap': {'strategy': 'zero_gap', 'gap_threshold': 2}},
        'window_mean': {'windows': [3], 'segment': 'short_gap'},
        'window_median': {'windows': [3]},  # обычный, без NaN
    }
    accepted = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', preset)],
        out_path=tmp_path / 'out.parquet',
        corr_threshold=0.9,
    )
    assert len(accepted) >= 1
    out = pl.read_parquet(tmp_path / 'out.parquet')
    assert out.height == len(_WORKED_VALUES)


def test_minimum_with_segments_preset_from_disk(tmp_path):
    # Два entity по 10 записей: id=1 без разрывов (контроль), id=2 начинается
    # с нулей - ряд ещё не начался (zero_gap исключает ведущие нули целиком,
    # без grace period gap_threshold, см. _segmentation.py).
    entity1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    entity2 = [0, 0, 0, 40, 50, 60, 70, 80, 90, 100]
    df = pl.DataFrame({
        'entity_id': [1] * 10 + [2] * 10,
        'ts_key': list(range(1, 11)) + list(range(1, 11)),
        'value': [float(v) for v in entity1 + entity2],
    })

    accepted = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('value', 'minimum_with_segments')],  # реальный файл с диска
        out_path=tmp_path / 'out.parquet',
    )

    assert set(accepted) == {
        'value__rolling_sum__w3__seg-zerogap2',
        'value__segment_gap__seg-zerogap2',
    }

    out = pl.read_parquet(tmp_path / 'out.parquet').sort(['entity_id', 'ts_key'])
    e1 = out.filter(pl.col('entity_id') == 1)
    e2 = out.filter(pl.col('entity_id') == 2)

    # id=1: без триггеров - сегментированная сумма совпадает с обычным rolling_sum.
    assert e1['value__rolling_sum__w3__seg-zerogap2'].to_numpy().tolist() == pytest.approx(
        [10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0]
    )
    assert e1['value__segment_gap__seg-zerogap2'].to_numpy().tolist() == pytest.approx([0.0] * 10)

    # id=2: все 3 ведущих нуля - ряд ещё не начался (zero_gap исключает ведущие
    # триггеры целиком, без grace period gap_threshold, см. _segmentation.py) ->
    # NaN. С 40 начинается первый настоящий сегмент, position=0.
    got_sum = e2['value__rolling_sum__w3__seg-zerogap2'].to_numpy()
    expected_sum = [np.nan, np.nan, np.nan, 40.0, 90.0, 150.0, 180.0, 210.0, 240.0, 270.0]
    for i, exp in enumerate(expected_sum):
        if math.isnan(exp):
            assert math.isnan(got_sum[i]), f'row {i}: expected NaN, got {got_sum[i]}'
        else:
            assert got_sum[i] == pytest.approx(exp), f'row {i}: {got_sum[i]} != {exp}'

    got_gap = e2['value__segment_gap__seg-zerogap2'].to_numpy()
    assert got_gap.tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
