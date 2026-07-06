"""Тесты для FeatureScreener — статистического пре-фильтра признаков."""

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.feature_screening import FeatureScreener


# ─── Фикстуры ────────────────────────────────────────────────────────────────


@pytest.fixture()
def basic_dataset():
    """1000 строк, 20% позитивов; смесь хороших и плохих признаков."""
    rng = np.random.default_rng(42)
    n, n_pos = 1000, 200
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1

    X = pd.DataFrame({
        'strong':      rng.normal(0, 1, n) + y * 3.0,      # AUC ≈ 0.98
        'weak':        rng.normal(0, 1, n) + y * 0.7,      # AUC ≈ 0.69
        'noise':       rng.normal(0, 1, n),                 # AUC ≈ 0.50
        'constant':    np.ones(n),                           # var=0
        'quasi':       np.where(rng.random(n) < 0.995, 7, 0),  # mode_rate>0.99
        'nulls':       pd.array([np.nan]*960 + list(rng.normal(0, 1, 40))),  # 96% NaN
    })
    return X, y


# ─── Базовые случаи ──────────────────────────────────────────────────────────


def test_constant_removed(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(min_univariate_auc=0.51)
    screener.fit(X, y)
    assert 'constant' not in screener.selected_features_


def test_quasi_constant_removed(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(max_quasi_constant_rate=0.99)
    screener.fit(X, y)
    assert 'quasi' not in screener.selected_features_


def test_high_null_rate_removed(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(max_null_rate=0.95)
    screener.fit(X, y)
    assert 'nulls' not in screener.selected_features_


def test_noise_removed_by_auc(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(min_univariate_auc=0.55)
    screener.fit(X, y)
    assert 'noise' not in screener.selected_features_


def test_strong_signal_kept(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(min_univariate_auc=0.55)
    screener.fit(X, y)
    assert 'strong' in screener.selected_features_


def test_weak_signal_kept_at_mild_threshold(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(min_univariate_auc=0.51)
    screener.fit(X, y)
    assert 'weak' in screener.selected_features_


def test_weak_signal_removed_at_aggressive_threshold(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(min_univariate_auc=0.80)
    screener.fit(X, y)
    assert 'weak' not in screener.selected_features_


# ─── Фильтры отключаются через граничные значения ────────────────────────────


def test_disable_null_filter(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener(max_null_rate=1.0, min_univariate_auc=0.0,
                                max_quasi_constant_rate=1.0, min_variance=0.0)
    screener.fit(X, y)
    # Ни один признак не удаляется ни одним фильтром
    assert set(screener.selected_features_) == set(X.columns)


# ─── Порядок применения фильтров (первая сработавшая причина) ────────────────


def test_removal_reason_priority():
    """Признак с null_rate>0.95 и variance=0 получает причину high_null_rate (первый фильтр)."""
    n = 100
    X = pd.DataFrame({
        'col': pd.array([np.nan] * 97 + [1.0] * 3),  # 97% NaN, var≈0
    })
    y = np.zeros(n, dtype=int)
    y[:20] = 1

    screener = FeatureScreener(max_null_rate=0.95)
    screener.fit(X, y)
    assert screener.report().loc['col', 'removed_by'] == 'high_null_rate'


# ─── report() и removal_summary() ───────────────────────────────────────────


def test_report_has_expected_columns(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener().fit(X, y)
    report = screener.report()
    for col in ('null_rate', 'variance', 'quasi_constant_rate', 'univariate_auc', 'kept', 'removed_by'):
        assert col in report.columns


def test_removal_summary_counts_match(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener().fit(X, y)
    report = screener.report()
    summary = screener.removal_summary()

    total_removed_in_summary = summary[summary['причина'] == 'ИТОГО удалено']['признаков'].iloc[0]
    total_removed_in_report = (~report['kept']).sum()
    assert int(total_removed_in_summary) == int(total_removed_in_report)


def test_selected_features_consistent_with_report(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener().fit(X, y)
    kept_in_report = set(screener.report().index[screener.report()['kept']])
    assert set(screener.selected_features_) == kept_in_report


# ─── method chaining ─────────────────────────────────────────────────────────


def test_method_chaining(basic_dataset):
    X, y = basic_dataset
    screener = FeatureScreener()
    result = screener.fit(X, y)
    assert result is screener


# ─── Ошибка при обращении до fit ─────────────────────────────────────────────


def test_report_before_fit_raises():
    screener = FeatureScreener()
    with pytest.raises(RuntimeError, match='fit'):
        screener.report()


def test_removal_summary_before_fit_raises():
    screener = FeatureScreener()
    with pytest.raises(RuntimeError, match='fit'):
        screener.removal_summary()


# ─── Граничные случаи ────────────────────────────────────────────────────────


def test_all_positive_class():
    """Только один класс — AUC=0.5, все фичи с сигналом убираются по AUC."""
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({'f': rng.normal(0, 1, n)})
    y = np.ones(n, dtype=int)
    screener = FeatureScreener(min_univariate_auc=0.51)
    screener.fit(X, y)
    assert 'f' not in screener.selected_features_


def test_single_feature_kept():
    """Один сильный признак проходит все фильтры."""
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) > 0.7).astype(int)
    X = pd.DataFrame({'f': rng.normal(0, 1, n) + y * 5.0})
    screener = FeatureScreener(min_univariate_auc=0.55)
    screener.fit(X, y)
    assert screener.selected_features_ == ['f']


def test_subsample_gives_same_verdict():
    """AUC на подвыборке и без неё должен дать одинаковое решение для явных случаев."""
    rng = np.random.default_rng(0)
    n = 5000
    y = np.zeros(n, dtype=int)
    y[:1000] = 1
    X = pd.DataFrame({
        'signal': rng.normal(0, 1, n) + y * 3.0,
        'noise':  rng.normal(0, 1, n),
    })

    s_full = FeatureScreener(min_univariate_auc=0.55, auc_subsample=None).fit(X, y)
    s_sub  = FeatureScreener(min_univariate_auc=0.55, auc_subsample=1000).fit(X, y)

    assert set(s_full.selected_features_) == set(s_sub.selected_features_)


def test_mutual_info_filter():
    """mutual_info_filter удаляет чистый шум при включении."""
    sklearn = pytest.importorskip('sklearn')
    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) > 0.7).astype(int)
    X = pd.DataFrame({
        'signal': rng.normal(0, 1, n) + y * 2.0,
        'noise':  rng.normal(0, 1, n),
    })
    screener = FeatureScreener(min_univariate_auc=0.0, min_mutual_info=0.005)
    screener.fit(X, y)
    # Шум должен иметь близкую к нулю взаимную информацию
    assert screener.report().loc['noise', 'mutual_info'] < 0.01
