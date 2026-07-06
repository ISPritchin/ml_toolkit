import numpy as np
import polars as pl

from ml_toolkit.correlation_filter import filter_correlated_features


def test_drops_perfectly_correlated_duplicate():
    n = 200
    base = np.linspace(0, 100, n)
    df = pl.DataFrame(
        {
            "a": base,
            "b": base * 2.0,  # идеально коррелирует с a
            "c": np.random.default_rng(0).normal(0, 1, n),  # независимый шум
        }
    )
    accepted = filter_correlated_features(df, ["a", "b", "c"], threshold=0.9)
    assert accepted == ["a", "c"]


def test_ignores_observations_where_both_are_zero():
    # a и b совпадают только в ненулевых наблюдениях (где сильно коррелируют),
    # а большинство строк - совместные нули, которые по правилу должны
    # игнорироваться при расчёте корреляции
    n = 50
    a = np.zeros(n)
    b = np.zeros(n)
    a[:10] = np.arange(1, 11)
    b[:10] = np.arange(1, 11) * 3.0  # сильно коррелирует на ненулевых
    df = pl.DataFrame({"a": a, "b": b})
    accepted = filter_correlated_features(df, ["a", "b"], threshold=0.9)
    assert accepted == ["a"]


def test_keeps_uncorrelated_features():
    rng = np.random.default_rng(1)
    n = 300
    df = pl.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    accepted = filter_correlated_features(df, ["x", "y"], threshold=0.9)
    assert accepted == ["x", "y"]
