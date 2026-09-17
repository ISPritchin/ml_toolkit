import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('rolling_min_max', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_min_max():
    # [10,80,40,20,5,30] w=6 → min=5, max=80
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, 'max_w6')[-1] == pytest.approx(80.0)


def test_all_zeros_min_max_both_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'max_w6')[-1] == pytest.approx(0.0)


def test_partial_window_uses_available_rows():
    # At row 2 only 3 values [10,80,40] are in window
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'min_w6')[2] == pytest.approx(10.0)
    assert _get(arrs, sfxs, 'max_w6')[2] == pytest.approx(80.0)


def test_monotone_ascending_max_equals_current():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    # max is always the last (current) value
    assert _get(arrs, sfxs, 'max_w6')[-1] == pytest.approx(60.0)
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(10.0)


def test_zero_in_window_sets_min_to_zero():
    # Any zero in window → min=0
    arrs, sfxs = _run([100, 50, 0, 80, 70, 90], {'windows': [6]})
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # min of [10,0,60,0,0,35]=0
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(0.0, abs=1e-06)
    # max of [10,0,60,0,0,35]=60
    assert _get(arrs, sfxs, 'max_w6')[-1] == pytest.approx(60.0, abs=1e-06)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'min_w4') == pytest.approx([6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'max_w4') == pytest.approx([6.0, 6.0, 12.0, 12.0, 12.0, 15.0, 15.0, 15.0, 20.0], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [10, 80, 40, 20, 5, 30]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['min_w6', 'max_w6']
    for a, b in zip(arrs_plain, arrs_explicit, strict=True):
        assert a.tolist() == b.tolist()


def test_dilation_2_reads_every_other_point():
    # row_idx=4 (last), ws=3, dilation=2 -> idx 0,2,4 -> [10,40,5]
    values = [10, 999, 40, 999, 5]
    arrs, sfxs = _run(values, {'windows': [3], 'dilations': [1, 2]})
    assert sfxs == ['min_w3', 'max_w3', 'min_w3_d2', 'max_w3_d2']
    assert _get(arrs, sfxs, 'min_w3_d2')[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, 'max_w3_d2')[-1] == pytest.approx(40.0)
