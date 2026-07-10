import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params):
    return run_transformer('segment_gap', values, params)


def _get(arrays, suffixes):
    return get_feature_output(arrays, suffixes, '')


def test_worked_example_zero_gap():
    values = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 4, 3, 5, 2]
    arrs, sfxs = _run(values, {'segment': {'strategy': 'zero_gap', 'gap_threshold': 2}})
    flag = _get(arrs, sfxs)
    expected = [0.0] * 7 + [1.0] * 4 + [0.0] * 4
    assert flag.tolist() == pytest.approx(expected)


def test_no_gap_all_zero_flags():
    values = [1, 2, 3, 4, 5]
    arrs, sfxs = _run(values, {'segment': {'strategy': 'zero_gap', 'gap_threshold': 2}})
    flag = _get(arrs, sfxs)
    assert flag.tolist() == pytest.approx([0.0] * 5)


def test_relative_gap_strategy():
    values = [10, 10, 10, 10, 1, 1]
    arrs, sfxs = _run(
        values,
        {'segment': {
            'strategy': 'relative_gap', 'gap_threshold': 1,
            'reference_window': 4, 'relative_threshold': 0.5,
        }},
    )
    flag = _get(arrs, sfxs)
    # idx4 триггерит (1 < 0.5*7.75), grace period gap_threshold=1 поглощает её;
    # idx5 (тоже триггер, run=2>=1) исключается
    assert flag.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_mask_strategy_raises_clear_error():
    with pytest.raises(ValueError, match='mask'):
        _run([1, 2, 3], {'segment': {'strategy': 'mask', 'gap_threshold': 1, 'mask_column': 'is_active'}})


def test_missing_segment_param_raises_keyerror():
    with pytest.raises(KeyError):
        _run([1, 2, 3], {})
