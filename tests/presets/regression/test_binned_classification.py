"""Тесты RegressionByBinnedClassification (ml_toolkit/presets/regression/binned_classification.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from ml_toolkit.presets.regression import RegressionByBinnedClassification

_SMALL_PARAMS = {'iterations': 80, 'max_depth': 4, 'early_stopping_rounds': 30}


@pytest.fixture
def bimodal_data():
    """Таргет с двумя чётко разделёнными модами — MSE-регрессия «размазывает».

    прогноз между ними, ожидаем, что биннинг справляется заметно лучше.
    """
    rng = np.random.default_rng(3)
    n_train, n_valid = 400, 120
    cols = [f'f{i}' for i in range(5)]

    def make(n):
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        mode = (X['f0'] > 0).astype(int)
        y = pd.Series(np.where(mode == 1, 20.0, -20.0) + rng.normal(scale=1.0, size=n))
        return X, y

    return (*make(n_train), *make(n_valid))


# ── 1. Валидация конструктора ───────────────────────────────────────────────

def test_constructor_rejects_invalid_n_bins():
    with pytest.raises(ValueError, match='n_bins должен быть'):
        RegressionByBinnedClassification(n_bins=1)


def test_constructor_rejects_invalid_binning():
    with pytest.raises(ValueError, match='binning должен быть'):
        RegressionByBinnedClassification(binning='bogus')


def test_constructor_rejects_invalid_decode():
    with pytest.raises(ValueError, match='decode должен быть'):
        RegressionByBinnedClassification(decode='bogus')


def test_fit_rejects_degenerate_target(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    y_train = y_train.copy()
    y_train[:] = 5.0
    model = RegressionByBinnedClassification(n_bins=32, base_params=_SMALL_PARAMS)
    with pytest.raises(ValueError, match='вырожден'):
        model.fit(X_train, y_train, X_valid, y_valid)


# ── 2. Смоук fit/predict для комбинаций binning/decode ──────────────────────

@pytest.mark.parametrize('binning', ['quantile', 'uniform'])
@pytest.mark.parametrize('decode', ['mean', 'median'])
def test_fit_predict(regression_data, binning, decode):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RegressionByBinnedClassification(
        n_bins=16, binning=binning, decode=decode, base_params=_SMALL_PARAMS,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.n_bins_actual_ <= 16
    assert model.bin_edges_.shape == (model.n_bins_actual_ + 1,)
    assert model.bin_repr_.shape == (model.n_bins_actual_,)

    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    assert np.allclose(model.valid_pred_, pred)


def test_quantile_binning_gives_roughly_equal_bin_counts(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RegressionByBinnedClassification(n_bins=8, binning='quantile', base_params=_SMALL_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    bin_tr = model._assign_bins(y_train.values, model.bin_edges_)
    counts = np.bincount(bin_tr, minlength=model.n_bins_actual_)
    assert counts.min() > 0
    assert counts.max() / counts.min() < 3  # примерно равное число строк на бин


# ── 3. Мультимодальный таргет: предсказания не «застревают» в пустом провале ──

def test_predictions_avoid_the_empty_gap_between_modes(bimodal_data):
    """С quantile-биннингом на этих данных один из бинов может «перекрыть».

    пустой промежуток между модами (квантильные границы схлопывают редкие
    хвостовые точки обеих мод в один бин) — используем binning='uniform',
    для которого при n_bins=16 ширина бина (~2.9) намного меньше промежутка
    между модами (~34), поэтому «мостовых» бинов не возникает и decode
    остаётся внутри реальных кластеров, а не размазывается между ними
    (в отличие от предсказания одной MSE-регрессии — та тоже стабильно
    разделяет моды по f0 в этом простом синтетическом случае, поэтому здесь
    проверяется не превосходство над ней, а то, что сам биннинг-подход
    механически корректен: почти ни одно предсказание не должно попасть
    в промежуток, где вообще нет обучающих данных).
    """
    X_train, y_train, X_valid, y_valid = bimodal_data
    model = RegressionByBinnedClassification(
        n_bins=16, binning='uniform', decode='mean',
        base_params={'iterations': 300, 'early_stopping_rounds': 60},
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    pred = model.predict(X_valid)

    assert mean_absolute_error(y_valid, pred) < 3.0
    assert np.mean(np.abs(pred) < 10.0) < 0.05  # почти никогда не попадает в пустой промежуток [-10, 10]


# ── 4. Optuna тюнит архитектуру, score — MAE декодированного прогноза ──────

@pytest.mark.slow
def test_optuna_tunes_architecture(regression_data):
    X_train, y_train, X_valid, y_valid = regression_data
    model = RegressionByBinnedClassification(n_bins=8, n_optuna_trials=3, random_seed=42)
    model.fit(X_train, y_train, X_valid, y_valid)

    for key, bounds in {
        'iterations': (300, 1000), 'max_depth': (3, 7),
        'learning_rate': (0.001, 0.3), 'l2_leaf_reg': (1e-5, 10.0),
        'subsample': (0.5, 1.0), 'min_data_in_leaf': (1, 30),
    }.items():
        assert bounds[0] <= model.best_params_[key] <= bounds[1], (key, model.best_params_[key])

    pred = model.predict(X_valid)
    assert np.all(np.isfinite(pred))
