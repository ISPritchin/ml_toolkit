import numpy as np
import pytest

from ml_toolkit.transformers._segmentation import (
    _run_length_segment,
    _trigger_relative_gap,
    _trigger_zero_gap,
    compute_segment_position,
    segment_suffix_fragment,
    validate_segment_config,
)
from ml_toolkit.transformers._windowing import compute_position_within_entity


def test_worked_example_two_segments():
    # Пример из обсуждения: два периода активности, разрыв из 2+ нулей подряд.
    values = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 4, 3, 5, 2], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(len(values), dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)

    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=2)

    expected_in_segment = np.array(
        [True] * 7 + [False] * 4 + [True] * 4, dtype=bool
    )
    assert np.array_equal(in_segment, expected_in_segment)

    expected_pos_seg = np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 1, 2, 3], dtype=np.int64)
    assert np.array_equal(pos_seg, expected_pos_seg)


def test_gap_threshold_is_a_grace_period_not_immediate_exclusion():
    # gap_threshold=N: первые N подряд идущих триггеров ещё считаются частью
    # сегмента (grace period, подтверждающий, что это настоящий разрыв, а не
    # случайный нулевой месяц) — исключаются только строки СВЕРХ этого окна.
    # Одиночный триггер длиной ровно 1 при gap_threshold=1 целиком поглощается
    # grace period -> ни одна строка не исключается. Порог всё же был "нащупан"
    # внутренним состоянием (run_length достиг gap_threshold), поэтому строка
    # сразу после триггера трактуется как старт нового сегмента (position=0),
    # а не продолжение прежнего.
    values = np.array([1, 0, 1, 1], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(4, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=1)
    assert in_segment.tolist() == [True, True, True, True]
    assert pos_seg.tolist() == [0, 1, 0, 1]


def test_gap_threshold_one_excludes_rows_beyond_grace_period():
    # Разрыв длиной 2 при gap_threshold=1: первая нулевая строка — grace period
    # (остаётся в сегменте), вторая — уже исключена.
    values = np.array([1, 0, 0, 1], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(4, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=1)
    assert in_segment.tolist() == [True, True, False, True]
    assert pos_seg.tolist() == [0, 1, 0, 0]


def test_entity_boundary_resets_segmentation_state():
    # Два клиента подряд: разрыв первого клиента не должен просачиваться во второго.
    values = np.array([1, 0, 0, 0, 5, 6, 7], dtype=np.float64)
    entity_codes = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    position = compute_position_within_entity(entity_codes)
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=2)

    # клиент 0: idx0 (v=1) и первые 2 нуля (grace period, gap_threshold=2) — в
    # сегменте; idx3 (третий подряд ноль) исключается.
    assert in_segment[:4].tolist() == [True, True, True, False]
    # клиент 1: свежий сегмент с позиции 0, независимо от состояния клиента 0
    assert in_segment[4:].tolist() == [True, True, True]
    assert pos_seg[4:].tolist() == [0, 1, 2]


def test_no_triggers_single_segment():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    position = compute_position_within_entity(np.zeros(4, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=2)
    assert in_segment.all()
    assert pos_seg.tolist() == [0, 1, 2, 3]


def test_all_triggers_whole_series_is_gap_after_threshold():
    values = np.zeros(5)
    position = compute_position_within_entity(np.zeros(5, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(is_trigger, position, gap_threshold=2)
    # idx0: run=1<2 -> in segment; idx1: run=2>=2 -> still in segment (threshold row);
    # idx2..4: in gap
    assert in_segment.tolist() == [True, True, False, False, False]


def test_exclude_leading_triggers_excludes_entire_leading_run_regardless_of_threshold():
    # Ряд ещё не начался: ведущие нули не получают grace period gap_threshold,
    # исключаются целиком (даже если их меньше gap_threshold).
    values = np.array([0, 0, 0, 4, 5, 6], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(6, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(
        is_trigger, position, gap_threshold=2, exclude_leading_triggers=True
    )
    assert in_segment.tolist() == [False, False, False, True, True, True]
    assert pos_seg.tolist() == [0, 0, 0, 0, 1, 2]


def test_exclude_leading_triggers_single_leading_zero_still_excluded():
    # Даже один ведущий ноль исключается - "ещё не начался" не завязано на счётчик.
    values = np.array([0, 4, 5, 6], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(4, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(
        is_trigger, position, gap_threshold=5, exclude_leading_triggers=True
    )
    assert in_segment.tolist() == [False, True, True, True]
    assert pos_seg.tolist() == [0, 0, 1, 2]


def test_exclude_leading_triggers_all_zero_entity_never_starts():
    # Ряд из одних нулей - сегмент так и не открывается ни разу.
    values = np.zeros(5)
    position = compute_position_within_entity(np.zeros(5, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(
        is_trigger, position, gap_threshold=2, exclude_leading_triggers=True
    )
    assert in_segment.tolist() == [False] * 5


def test_exclude_leading_triggers_mid_series_gap_still_uses_grace_period():
    # Ведущий пробег отличается от обычного пробега В СЕРЕДИНЕ ряда: как
    # только ряд стартовал, дальнейшие разрывы снова проходят grace period.
    values = np.array([0, 0, 5, 6, 0, 0, 0, 7], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(8, dtype=np.int64))
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(
        is_trigger, position, gap_threshold=2, exclude_leading_triggers=True
    )
    # idx0-1: ведущие нули - исключены целиком (не grace period).
    # idx2-3: первый настоящий сегмент.
    # idx4-5: обычный grace period (gap_threshold=2) - ещё в сегменте.
    # idx6: третий подряд ноль - разрыв.
    # idx7: новый сегмент.
    assert in_segment.tolist() == [False, False, True, True, True, True, False, True]


def test_exclude_leading_triggers_does_not_leak_across_entities():
    # Первая сущность стартует сразу, вторая - с ведущих нулей; состояние
    # "ещё не начался" сбрасывается на границе entity независимо от первой.
    values = np.array([5, 6, 0, 0, 7, 8], dtype=np.float64)
    entity_codes = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    position = compute_position_within_entity(entity_codes)
    is_trigger = _trigger_zero_gap(values)
    pos_seg, in_segment = _run_length_segment(
        is_trigger, position, gap_threshold=5, exclude_leading_triggers=True
    )
    assert in_segment.tolist() == [True, True, False, False, True, True]
    assert pos_seg[4:].tolist() == [0, 1]


def test_compute_segment_position_zero_gap_excludes_leading_zeros():
    # compute_segment_position с strategy='zero_gap' автоматически включает
    # exclude_leading_triggers - вызывающему коду не нужно об этом заботиться.
    values = np.array([0, 0, 40, 50, 60], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(5, dtype=np.int64))
    pos_seg, in_segment = compute_segment_position(
        values, position, 'zero_gap', {'gap_threshold': 2}
    )
    assert in_segment.tolist() == [False, False, True, True, True]


def test_compute_segment_position_relative_gap_keeps_grace_period_for_early_triggers():
    # relative_gap НЕ получает exclude_leading_triggers - это специфично для
    # zero_gap (см. _segmentation.py докстринг): ранние триггер-строки всё
    # равно проходят обычный grace period gap_threshold, а не исключаются
    # безусловно.
    values = np.array([10, 1, 1, 10, 10, 10], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(6, dtype=np.int64))
    pos_seg, in_segment = compute_segment_position(
        values, position, 'relative_gap',
        {'gap_threshold': 2, 'reference_window': 3, 'relative_threshold': 0.5},
    )
    # idx1,2 триггерят relative_gap (падение ниже 0.5*reference), но остаются
    # в сегменте - grace period, не «ряд ещё не начался».
    assert in_segment[1]
    assert in_segment[2]


def test_trigger_relative_gap_detects_drop():
    # Плавный уровень ~10, затем резкое падение ниже relative_threshold*mean
    values = np.array([10, 10, 10, 10, 1, 1], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(6, dtype=np.int64))
    is_trigger = _trigger_relative_gap(values, position, reference_window=4, relative_threshold=0.5)
    assert not is_trigger[0]  # первая точка: reference = сама себя (ws=1) -> not < 0.5*10
    assert not is_trigger[3]  # всё ещё уровень ~10
    assert is_trigger[4]      # 1 < 0.5 * mean(10,10,10,1)=7.75


def test_compute_segment_position_zero_gap_matches_worked_example():
    values = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 4, 3, 5, 2], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(len(values), dtype=np.int64))
    pos_seg, in_segment = compute_segment_position(
        values, position, 'zero_gap', {'gap_threshold': 2}
    )
    assert in_segment.tolist() == [True] * 7 + [False] * 4 + [True] * 4


def test_compute_segment_position_mask_strategy():
    # mask False длиной 3 при gap_threshold=2: первые 2 (grace period) ещё в
    # сегменте, 3-я исключается; mask=True сразу закрывает разрыв.
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    position = compute_position_within_entity(np.zeros(6, dtype=np.int64))
    external_mask = np.array([True, True, False, False, False, True])
    pos_seg, in_segment = compute_segment_position(
        values, position, 'mask', {'gap_threshold': 2}, external_mask=external_mask
    )
    assert in_segment.tolist() == [True, True, True, True, False, True]
    assert pos_seg.tolist() == [0, 1, 2, 3, 0, 0]


def test_compute_segment_position_mask_without_external_mask_raises():
    values = np.array([1.0, 2.0])
    position = compute_position_within_entity(np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match='external_mask'):
        compute_segment_position(values, position, 'mask', {'gap_threshold': 1})


def test_compute_segment_position_unknown_strategy_raises():
    values = np.array([1.0, 2.0])
    position = compute_position_within_entity(np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match='неизвестная стратегия'):
        compute_segment_position(values, position, 'bogus', {'gap_threshold': 1})


def test_compute_segment_position_gap_threshold_below_one_raises():
    values = np.array([1.0, 2.0])
    position = compute_position_within_entity(np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match='gap_threshold'):
        compute_segment_position(values, position, 'zero_gap', {'gap_threshold': 0})


def test_validate_segment_config_missing_keys_raises():
    with pytest.raises(ValueError, match='reference_window'):
        validate_segment_config({'strategy': 'relative_gap', 'gap_threshold': 2}, context='test')


def test_validate_segment_config_unknown_strategy_raises():
    with pytest.raises(ValueError, match='неизвестная стратегия'):
        validate_segment_config({'strategy': 'bogus', 'gap_threshold': 1}, context='test')


def test_validate_segment_config_ok_passthrough():
    cfg = {'strategy': 'zero_gap', 'gap_threshold': 2}
    assert validate_segment_config(cfg, context='test') is cfg


def test_segment_suffix_fragment_zero_gap():
    assert segment_suffix_fragment({'strategy': 'zero_gap', 'gap_threshold': 2}) == 'seg-zerogap2'


def test_segment_suffix_fragment_relative_gap():
    frag = segment_suffix_fragment(
        {'strategy': 'relative_gap', 'gap_threshold': 2, 'reference_window': 6, 'relative_threshold': 0.3}
    )
    assert frag == 'seg-relgap0p3r6g2'


def test_segment_suffix_fragment_mask():
    frag = segment_suffix_fragment({'strategy': 'mask', 'gap_threshold': 3, 'mask_column': 'is_active'})
    assert frag == 'seg-mask-is_activeg3'


def test_segment_suffix_fragment_distinguishes_different_configs():
    a = segment_suffix_fragment({'strategy': 'zero_gap', 'gap_threshold': 2})
    b = segment_suffix_fragment({'strategy': 'zero_gap', 'gap_threshold': 3})
    assert a != b
