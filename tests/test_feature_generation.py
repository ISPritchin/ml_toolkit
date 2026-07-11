"""Тесты генерического движка `ml_toolkit.feature_generation` — без привязки к задаче.

В отличие от `tasks/auto_kkp_classification/tests/test_feature_generation_e2e.py`
(бизнес-обёртка над двумя конкретными датасетами cltv_subset/holding), здесь
проверяется, что `select_features`/`apply_selected_features` (uniform-API) и
`generate_feature_groups`/`apply_feature_groups` (групповой API — разные
трансформеры и разные пресеты для разных колонок, `polars.selectors`) работают
с любым одним датасетом произвольной формы — колонки-сущности в двух тестах
здесь намеренно называются по-разному (`entity_id` и `group_id`), чтобы
показать, что движок не завязан на конкретную предметную область.
"""

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

from ml_toolkit.feature_generation import (
    apply_feature_groups,
    apply_selected_features,
    generate_feature_groups,
    select_features,
)

SLOPE_DEFAULT = {'slope': {'windows': [6, 12, 24]}}
STREAK = {'streak': {}}


def _growth_and_decline_df(entity_col: str) -> pl.DataFrame:
    months = list(range(1, 13))
    growing = [10.0 * i for i in months]
    declining = [10.0 * (13 - i) for i in months]
    return pl.DataFrame({
        entity_col: [1] * len(months) + [2] * len(months),
        'ts_key': months + months,
        'value': growing + declining,
    })


def _two_product_df(entity_col: str) -> pl.DataFrame:
    months = list(range(1, 13))
    growing = [10.0 * i for i in months]
    declining = [10.0 * (13 - i) for i in months]
    return pl.DataFrame({
        entity_col: [1] * len(months) + [2] * len(months),
        'ts_key': months + months,
        'trans_a': growing + declining,
        'trans_b': declining + growing,
    })


def _duplicate_product_df(entity_col: str) -> pl.DataFrame:
    months = list(range(1, 13))
    growing = [10.0 * i for i in months]
    declining = [10.0 * (13 - i) for i in months]
    values = growing + declining
    return pl.DataFrame({
        entity_col: [1] * len(months) + [2] * len(months),
        'ts_key': months + months,
        'trans_a': values,
        'trans_b': values,  # точная копия - фичи по ней идеально коррелируют с trans_a
    })


def test_select_features_writes_output_and_returns_accepted_cols(tmp_path):
    df = _growth_and_decline_df('entity_id')

    accepted_cols = select_features(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        product_cols=['value'],
        out_path=tmp_path / 'out.parquet',
        corr_threshold=None,  # без фильтра - детерминированный список кандидатов
        transformer_names=['slope'],
        preset='minimum',
    )

    assert accepted_cols == ['value__slope__w6', 'value__slope__w12', 'value__slope__w24']

    out = pl.read_parquet(tmp_path / 'out.parquet')
    assert set(out.columns) == {'entity_id', 'ts_key', 'value', *accepted_cols}
    for col in accepted_cols:
        assert np.isfinite(out[col].to_numpy()).all()

    last_growing = out.filter(pl.col('entity_id') == 1).sort('ts_key').tail(1)
    last_declining = out.filter(pl.col('entity_id') == 2).sort('ts_key').tail(1)
    assert last_growing['value__slope__w6'].item() > 0
    assert last_declining['value__slope__w6'].item() < 0


def test_apply_selected_features_reuses_accepted_cols_on_differently_named_entity(tmp_path):
    source_df = _growth_and_decline_df('entity_id')
    accepted_cols = select_features(
        source_df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        product_cols=['value'],
        out_path=tmp_path / 'source.parquet',
        corr_threshold=None,
        transformer_names=['slope'],
        preset='minimum',
    )

    # Второй датасет с другим именем колонки-сущности - движок не должен об этом
    # ничего "знать": apply_selected_features принимает entity_column_name как
    # обычный параметр, а не догадывается по названию.
    other_df = _growth_and_decline_df('group_id')

    apply_selected_features(
        other_df,
        entity_column_name='group_id',
        ts_column_name='ts_key',
        product_cols=['value'],
        accepted_cols=accepted_cols,
        out_path=tmp_path / 'applied.parquet',
        transformer_names=['slope'],
        preset='minimum',
    )

    applied = pl.read_parquet(tmp_path / 'applied.parquet')
    assert set(applied.columns) == {'group_id', 'ts_key', 'value', *accepted_cols}
    for col in accepted_cols:
        assert np.isfinite(applied[col].to_numpy()).all()


def test_generate_feature_groups_applies_different_transformers_per_column(tmp_path):
    df = _two_product_df('entity_id')

    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[
            ('trans_a', SLOPE_DEFAULT),
            ('trans_b', STREAK),
        ],
        out_path=tmp_path / 'out.parquet',
    )

    assert all(c.startswith('trans_a__slope') or c.startswith('trans_b__streak') for c in result_cols)
    assert any(c.startswith('trans_a__slope') for c in result_cols)
    assert any(c.startswith('trans_b__streak') for c in result_cols)
    # trans_a не должна получить streak, а trans_b - slope
    assert not any(c.startswith('trans_a__streak') or c.startswith('trans_b__slope') for c in result_cols)

    out = pl.read_parquet(tmp_path / 'out.parquet')
    assert set(out.columns) == {'entity_id', 'ts_key', 'trans_a', 'trans_b', *result_cols}


def test_generate_feature_groups_supports_polars_selectors(tmp_path):
    df = _two_product_df('entity_id')

    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[(cs.starts_with('trans_'), SLOPE_DEFAULT)],
        out_path=tmp_path / 'out.parquet',
    )

    assert set(result_cols) == {
        'trans_a__slope__w6', 'trans_a__slope__w12', 'trans_a__slope__w24',
        'trans_b__slope__w6', 'trans_b__slope__w12', 'trans_b__slope__w24',
    }


def test_generate_feature_groups_dedupes_overlapping_requests(tmp_path):
    df = _two_product_df('entity_id')

    # "trans_a" запрошена явно строкой и повторно накрыта селектором - одна и та
    # же пара (колонка, трансформер) с одинаковыми параметрами не должна
    # задвоиться в выходной схеме.
    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[
            ('trans_a', SLOPE_DEFAULT),
            (cs.starts_with('trans_'), SLOPE_DEFAULT),
        ],
        out_path=tmp_path / 'out.parquet',
    )

    assert len(result_cols) == len(set(result_cols))
    assert result_cols.count('trans_a__slope__w6') == 1
    assert set(result_cols) == {
        'trans_a__slope__w6', 'trans_a__slope__w12', 'trans_a__slope__w24',
        'trans_b__slope__w6', 'trans_b__slope__w12', 'trans_b__slope__w24',
    }


def test_generate_feature_groups_correlation_filter_is_opt_in(tmp_path):
    df = _duplicate_product_df('entity_id')

    no_filter_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[(cs.starts_with('trans_'), SLOPE_DEFAULT)],
        out_path=tmp_path / 'no_filter.parquet',
    )
    # По умолчанию (corr_threshold=None) фильтр не запускается - обе колонки
    # (идентичные и потому идеально коррелирующие) сохраняются целиком.
    assert len(no_filter_cols) == 6

    filtered_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[(cs.starts_with('trans_'), SLOPE_DEFAULT)],
        out_path=tmp_path / 'filtered.parquet',
        corr_threshold=0.9,
    )
    # С явным порогом фильтр включается и отбрасывает дублирующие кандидаты
    # (не только между trans_a/trans_b, но и между окнами внутри одной колонки -
    # на этом синтетическом датасете они тоже сильно коррелируют).
    assert len(filtered_cols) < len(no_filter_cols)


def test_generate_feature_groups_empty_preset_dict_is_passthrough(tmp_path):
    df = _two_product_df('entity_id')

    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('trans_a', SLOPE_DEFAULT), ('trans_b', {})],
        out_path=tmp_path / 'out.parquet',
    )

    assert all(c.startswith('trans_a__') for c in result_cols)

    out = pl.read_parquet(tmp_path / 'out.parquet')
    assert 'trans_b' in out.columns
    assert not any(c.startswith('trans_b__') for c in out.columns)


def test_generate_feature_groups_empty_spec_raises(tmp_path):
    df = _two_product_df('entity_id')

    with pytest.raises(ValueError, match='feature_spec пуст'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[],
            out_path=tmp_path / 'out.parquet',
        )


def test_generate_feature_groups_missing_preset_raises(tmp_path):
    df = _two_product_df('entity_id')

    # Автоматического пресета по умолчанию нет - None вторым элементом запрещён.
    with pytest.raises(ValueError, match='пресет обязателен'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[('trans_a', None)],
            out_path=tmp_path / 'out.parquet',
        )


def test_generate_feature_groups_unknown_column_raises(tmp_path):
    df = _two_product_df('entity_id')

    with pytest.raises(ValueError, match='отсутствующие в df'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[('does_not_exist', SLOPE_DEFAULT)],
            out_path=tmp_path / 'out.parquet',
        )


def test_generate_feature_groups_unknown_transformer_name_raises(tmp_path):
    df = _two_product_df('entity_id')

    with pytest.raises(ValueError, match='Неизвестный трансформер'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[('trans_a', {'not_a_real_transformer': {}})],
            out_path=tmp_path / 'out.parquet',
        )


def test_apply_feature_groups_reuses_result_cols_on_differently_named_entity(tmp_path):
    source_df = _two_product_df('entity_id')
    feature_spec = [('trans_a', SLOPE_DEFAULT), ('trans_b', STREAK)]

    result_cols = generate_feature_groups(
        source_df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=feature_spec,
        out_path=tmp_path / 'source.parquet',
    )

    other_df = _two_product_df('group_id')
    apply_feature_groups(
        other_df,
        entity_column_name='group_id',
        ts_column_name='ts_key',
        feature_spec=feature_spec,
        accepted_cols=result_cols,
        out_path=tmp_path / 'applied.parquet',
    )

    applied = pl.read_parquet(tmp_path / 'applied.parquet')
    assert set(applied.columns) == {'group_id', 'ts_key', 'trans_a', 'trans_b', *result_cols}
    for col in result_cols:
        assert np.isfinite(applied[col].to_numpy()).all()


def test_generate_feature_groups_per_group_own_params(tmp_path):
    df = _two_product_df('entity_id')

    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[
            ('trans_a', {'slope': {'windows': [3, 6]}}),  # свои параметры
            ('trans_b', SLOPE_DEFAULT),                     # другие параметры
        ],
        out_path=tmp_path / 'out.parquet',
    )

    assert set(result_cols) == {
        'trans_a__slope__w3', 'trans_a__slope__w6',
        'trans_b__slope__w6', 'trans_b__slope__w12', 'trans_b__slope__w24',
    }


def test_generate_feature_groups_accepts_named_preset_from_disk(tmp_path):
    df = _two_product_df('entity_id')

    # "minimum" - реальный пресет из ml_toolkit/transformers/presets/,
    # применяется целиком (все трансформеры файла, здесь - только slope).
    result_cols = generate_feature_groups(
        df,
        entity_column_name='entity_id',
        ts_column_name='ts_key',
        feature_spec=[('trans_a', 'minimum')],
        out_path=tmp_path / 'out.parquet',
    )

    assert result_cols == ['trans_a__slope__w6', 'trans_a__slope__w12', 'trans_a__slope__w24']


def test_generate_feature_groups_conflicting_presets_for_same_column_raises(tmp_path):
    df = _two_product_df('entity_id')

    with pytest.raises(ValueError, match='разными параметрами'):
        generate_feature_groups(
            df,
            entity_column_name='entity_id',
            ts_column_name='ts_key',
            feature_spec=[
                ('trans_a', {'slope': {'windows': [3, 6]}}),
                (cs.starts_with('trans_'), SLOPE_DEFAULT),  # другие параметры - конфликт на trans_a
            ],
            out_path=tmp_path / 'out.parquet',
        )
