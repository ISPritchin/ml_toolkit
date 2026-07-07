"""10 сценарных тестов FeatureScreener на реалистичных датасетах.

Каждый сценарий — отдельная ситуация, которая встречается на практике.
Используется polars-native API (ml_toolkit.feature_selection.FeatureScreener).
"""

import numpy as np
import polars as pl
import pytest

from ml_toolkit.feature_selection import FeatureScreener

# ── Хелпер для удобного чтения строки report() ──────────────────────────────

def _row(report: pl.DataFrame, feature: str) -> dict:
    """Возвращает строку report() для указанного признака как dict."""
    rows = report.filter(pl.col('feature') == feature).to_dicts()
    assert rows, f'Признак {feature!r} не найден в report()'
    return rows[0]


# ── Сценарий 1: все признаки хорошие, ничего не удаляется ───────────────────

def test_scenario_all_features_pass():
    """Чистый датасет без мусора — screener не удаляет ничего."""
    rng = np.random.default_rng(0)
    n = 500
    y = pl.Series((rng.random(n) > 0.6).astype(int))
    X = pl.DataFrame({
        'f1': rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'f2': rng.normal(0, 1, n) + y.to_numpy() * 2.5,
        'f3': rng.normal(0, 1, n) + y.to_numpy() * 2.0,
    })
    screener = FeatureScreener(min_univariate_auc=0.55).fit(X, y)
    assert set(screener.selected_features_) == {'f1', 'f2', 'f3'}
    assert screener.report()['kept'].all()


# ── Сценарий 2: весь датасет — мусор, все удаляются ─────────────────────────

def test_scenario_all_garbage():
    """Датасет состоит только из мусора — все признаки удаляются."""
    rng = np.random.default_rng(1)
    n = 300
    y = pl.Series((rng.random(n) > 0.7).astype(int))
    X = pl.DataFrame({
        'const':   pl.Series([1.0] * n),
        'high_na': pl.Series([None] * 280 + list(rng.normal(0, 1, 20))),
        'quasi_a': pl.Series(np.where(rng.random(n) < 0.98, 5.0, 0.0)),
        'quasi_b': pl.Series(np.where(rng.random(n) < 0.99, 0.0, 1.0)),
    })
    screener = FeatureScreener(
        max_null_rate=0.90,
        min_variance=1e-5,
        max_quasi_constant_rate=0.97,
        min_univariate_auc=0.55,
    ).fit(X, y)
    assert screener.selected_features_ == []
    assert not screener.report()['kept'].any()


# ── Сценарий 3: дубликаты ───────────────────────────────────────────────────

def test_scenario_duplicates_removed():
    """Точные копии признаков — вторая копия удаляется как duplicate."""
    rng = np.random.default_rng(2)
    n = 200
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    signal = rng.normal(0, 1, n) + y.to_numpy() * 2.0
    X = pl.DataFrame({
        'original': signal,
        'copy_a':   signal.copy(),
        'noise':    rng.normal(0, 1, n),
    })
    screener = FeatureScreener(
        min_univariate_auc=0.0,
        drop_duplicates=True,
    ).fit(X, y)
    assert 'original' in screener.selected_features_
    assert 'copy_a' not in screener.selected_features_
    assert _row(screener.report(), 'copy_a')['removed_by'] == 'duplicate'


def test_scenario_two_duplicate_pairs():
    """Два независимых дубль-пары — из каждой пары удаляется вторая."""
    rng = np.random.default_rng(3)
    n = 200
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    a = rng.normal(0, 1, n) + y.to_numpy() * 2.0
    b = rng.normal(0, 1, n) + y.to_numpy() * 1.5
    X = pl.DataFrame({'a1': a, 'a2': a.copy(), 'b1': b, 'b2': b.copy()})
    screener = FeatureScreener(min_univariate_auc=0.0, drop_duplicates=True).fit(X, y)
    assert 'a1' in screener.selected_features_
    assert 'a2' not in screener.selected_features_
    assert 'b1' in screener.selected_features_
    assert 'b2' not in screener.selected_features_


# ── Сценарий 4: высокий NaN, но сильный сигнал ──────────────────────────────

def test_scenario_high_nan_strong_signal_still_removed():
    """Даже сильный сигнал удаляется если NaN-ов больше порога — null-фильтр приоритетнее."""
    rng = np.random.default_rng(4)
    n = 500
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    signal_vals = rng.normal(0, 1, 50) + y.to_numpy()[:50] * 5.0
    X = pl.DataFrame({
        'partial_signal': pl.Series([None] * 450 + list(signal_vals)),
    })
    screener = FeatureScreener(max_null_rate=0.85).fit(X, y)
    assert 'partial_signal' not in screener.selected_features_
    assert _row(screener.report(), 'partial_signal')['removed_by'] == 'high_null_rate'


def test_scenario_moderate_nan_signal_survives():
    """Признак с 50% NaN и сильным сигналом НЕ удаляется — null_rate < порога."""
    rng = np.random.default_rng(5)
    n = 400
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    feat = pl.Series([None] * 200 + list(rng.normal(0, 1, 200) + y.to_numpy()[200:] * 4.0))
    X = pl.DataFrame({'half_nan_signal': feat})
    screener = FeatureScreener(max_null_rate=0.60, min_univariate_auc=0.55).fit(X, y)
    assert 'half_nan_signal' in screener.selected_features_


# ── Сценарий 5: каждый фильтр удаляет хотя бы один признак ─────────────────

def test_scenario_each_filter_fires():
    """Mixed-датасет: каждая стадия фильтрации удаляет ровно один признак."""
    rng = np.random.default_rng(6)
    n = 600
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    X = pl.DataFrame({
        'good':      rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'high_null': pl.Series([None] * 570 + list(rng.normal(0, 1, 30))),
        'constant':  pl.Series([42.0] * n),
        'quasi':     pl.Series(np.where(rng.random(n) < 0.97, 1.0, 0.0)),
        'noise':     rng.normal(0, 1, n),
    })
    screener = FeatureScreener(
        max_null_rate=0.90,
        min_variance=1e-5,
        max_quasi_constant_rate=0.96,
        min_univariate_auc=0.55,
    ).fit(X, y)

    report = screener.report()
    assert _row(report, 'high_null')['removed_by'] == 'high_null_rate'
    assert _row(report, 'constant')['removed_by'] == 'low_variance'
    assert _row(report, 'quasi')['removed_by'] == 'quasi_constant'
    assert _row(report, 'noise')['removed_by'] == 'low_auc'
    assert _row(report, 'good')['kept']
    assert screener.selected_features_ == ['good']


# ── Сценарий 6: transform и fit_transform ───────────────────────────────────

def test_scenario_transform_output():
    """transform() возвращает DataFrame только с отобранными признаками."""
    rng = np.random.default_rng(7)
    n = 300
    y = pl.Series((rng.random(n) > 0.6).astype(int))
    X = pl.DataFrame({
        'keep_me': rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'drop_me': pl.Series([1.0] * n),
    })
    screener = FeatureScreener().fit(X, y)
    X_out = screener.transform(X)
    assert X_out.columns == ['keep_me']
    assert len(X_out) == n


def test_scenario_fit_transform_equivalent():
    """fit_transform(X, y) == fit(X, y).transform(X)."""
    rng = np.random.default_rng(8)
    n = 300
    y = pl.Series((rng.random(n) > 0.6).astype(int))
    X = pl.DataFrame({
        'signal': rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'noise':  rng.normal(0, 1, n),
    })
    s1 = FeatureScreener(min_univariate_auc=0.55)
    s2 = FeatureScreener(min_univariate_auc=0.55)
    X1 = s1.fit_transform(X, y)
    X2 = s2.fit(X, y).transform(X)
    assert X1.equals(X2)


# ── Сценарий 7: много признаков, много мусора ────────────────────────────────

def test_scenario_many_features():
    """50 признаков: 5 сигнальных, 45 мусора — screener убирает большинство."""
    rng = np.random.default_rng(9)
    n = 800
    y = pl.Series((rng.random(n) > 0.5).astype(int))

    cols: dict[str, list | np.ndarray] = {}
    for i in range(5):
        cols[f'signal_{i}'] = rng.normal(0, 1, n) + y.to_numpy() * (2.5 + i * 0.2)
    for i in range(10):
        cols[f'noise_{i}'] = rng.normal(0, 1, n)
    for i in range(10):
        cols[f'const_{i}'] = [float(i)] * n
    for i in range(10):
        cols[f'null_{i}'] = pl.Series([None] * 720 + list(rng.normal(0, 1, 80)))
    for i in range(10):
        cols[f'quasi_{i}'] = np.where(rng.random(n) < 0.97, float(i), float(i + 1))

    X = pl.DataFrame(cols)
    screener = FeatureScreener(
        max_null_rate=0.85,
        max_quasi_constant_rate=0.96,
        min_univariate_auc=0.55,
    ).fit(X, y)

    selected = screener.selected_features_
    assert all(f'signal_{i}' in selected for i in range(5)), 'все сигнальные должны выжить'
    assert len(selected) <= 10, 'большинство мусора должно быть удалено'

    report = screener.report()
    reasons = report.filter(~pl.col('kept'))['removed_by'].to_list()
    assert 'high_null_rate' in reasons
    assert 'low_variance' in reasons
    assert 'quasi_constant' in reasons


# ── Сценарий 8: маленький датасет (n=30) ────────────────────────────────────

def test_scenario_tiny_dataset():
    """Screener работает корректно на очень маленьком датасете."""
    rng = np.random.default_rng(10)
    n = 30
    y_arr = (rng.random(n) > 0.5).astype(int)
    y_arr[:5] = 0
    y_arr[5:10] = 1
    y = pl.Series(y_arr)
    X = pl.DataFrame({
        'signal': rng.normal(0, 1, n) + y_arr * 4.0,
        'const':  [1.0] * n,
    })
    screener = FeatureScreener(min_univariate_auc=0.55).fit(X, y)
    assert 'signal' in screener.selected_features_
    assert 'const' not in screener.selected_features_


# ── Сценарий 9: MI-фильтр убирает шум, оставляет сигнал ────────────────────

def test_scenario_mutual_info_removes_noise():
    """mutual_info фильтр удаляет шум при отключённом AUC-фильтре."""
    pytest.importorskip('sklearn')
    rng = np.random.default_rng(11)
    n = 500
    y = pl.Series((rng.random(n) > 0.65).astype(int))
    X = pl.DataFrame({
        'signal': rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'noise':  rng.normal(0, 1, n),
    })
    screener = FeatureScreener(
        min_univariate_auc=0.0,
        min_mutual_info=0.01,
    ).fit(X, y)

    report = screener.report()
    assert _row(report, 'signal')['mutual_info'] > _row(report, 'noise')['mutual_info']
    assert 'signal' in screener.selected_features_
    assert 'noise' not in screener.selected_features_
    assert _row(report, 'noise')['removed_by'] == 'low_mutual_info'


# ── Сценарий 10: removal_summary покрывает все удалённые ────────────────────

def test_scenario_removal_summary_complete():
    """Итоговая сумма в removal_summary совпадает с числом удалённых в report."""
    rng = np.random.default_rng(12)
    n = 400
    y = pl.Series((rng.random(n) > 0.5).astype(int))
    X = pl.DataFrame({
        'good':    rng.normal(0, 1, n) + y.to_numpy() * 3.0,
        'const':   [1.0] * n,
        'null96':  pl.Series([None] * 384 + list(rng.normal(0, 1, 16))),
        'quasi98': np.where(rng.random(n) < 0.98, 7.0, 0.0),
        'noise':   rng.normal(0, 1, n),
    })
    screener = FeatureScreener(
        max_null_rate=0.90,
        max_quasi_constant_rate=0.97,
        min_univariate_auc=0.55,
    ).fit(X, y)

    report = screener.report()
    summary = screener.removal_summary()

    n_removed_report = int((~report['kept']).sum())
    n_removed_summary = int(
        summary.filter(pl.col('причина') == 'ИТОГО удалено')['признаков'][0]
    )
    assert n_removed_report == n_removed_summary

    per_reason = summary.filter(pl.col('причина') != 'ИТОГО удалено')
    assert (per_reason['признаков'] > 0).all()
    assert int(per_reason['признаков'].sum()) == n_removed_report
