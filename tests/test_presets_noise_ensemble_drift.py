"""Тесты для пресетов 025-030 (шумные метки/PU), 034/035/040 (ансамбли/стекинг),

044/045 (дрифт). Смоук fit/predict + несколько содержательных проверок
поведения (не только "не упало"), где это дёшево сделать на синтетике.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification.high_pr_auc import (
    AdversarialValidationWeighting,
    BaggingPUClassifier,
    ConfidentLearningCleaner,
    CoTeachingClassifier,
    DriftRobustClassifier,
    ElkanNotoHoldoutPU,
    GreedyForwardEnsembleSelection,
    HeterogeneousStacking,
    MultiSeedBlend,
    NNPUClassifier,
    SpyPUClassifier,
)

_BP = {'iterations': 50, 'verbose': 0, 'random_seed': 42}


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series((rng.random(n_train) < 0.15).astype(int))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series((rng.random(n_valid) < 0.15).astype(int))
    return X_train, y_train, X_valid, y_valid


def _assert_valid_proba(model, X_valid):
    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))


# ── 025 ConfidentLearningCleaner ────────────────────────────────────────────

class TestConfidentLearningCleaner:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = ConfidentLearningCleaner(n_folds=3, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert model.confident_joint_.shape == (2, 2)

    def test_recovers_deliberately_flipped_labels(self):
        # Строим разделимый датасет, затем сознательно переворачиваем метку
        # части позитивов на негатив — эти индексы должны попасть в
        # removed_indices_ значительно чаще случайного угадывания.
        rng = np.random.default_rng(1)
        n = 500
        X = rng.normal(size=(n, 4))
        true_score = X[:, 0] + X[:, 1]
        y = (true_score > np.quantile(true_score, 0.85)).astype(int)

        pos_idx = np.where(y == 1)[0]
        flipped = rng.choice(pos_idx, size=max(1, len(pos_idx) // 3), replace=False)
        y_noisy = y.copy()
        y_noisy[flipped] = 0

        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
        y_series = pd.Series(y_noisy)
        X_valid = pd.DataFrame(rng.normal(size=(100, 4)), columns=[f'f{i}' for i in range(4)])
        y_valid = pd.Series((X_valid['f0'] + X_valid['f1'] > np.quantile(true_score, 0.85)).astype(int))

        model = ConfidentLearningCleaner(n_folds=5, base_params=_BP)
        model.fit(X_df, y_series, X_valid, y_valid)

        recall = len(set(model.removed_indices_.tolist()) & set(flipped.tolist())) / len(flipped)
        assert recall > 0.3, f'Ожидали найти существенную долю перевёрнутых меток, recall={recall:.2f}'


# ── 026 CoTeachingClassifier ─────────────────────────────────────────────────

class TestCoTeachingClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = CoTeachingClassifier(n_rounds=2, forget_rate=0.3, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert len(model.round_scores_a_) == 3  # init + 2 раунда
        assert len(model.keep_fraction_history_) == 3

    def test_small_loss_selection_keeps_both_classes(self, binary_data):
        # Регресс: при агрессивном forget_rate + сильном дисбалансе глобальный
        # (не постратный) top-k по loss может целиком вымыть позитивы.
        X_train, y_train, X_valid, y_valid = binary_data
        model = CoTeachingClassifier(n_rounds=3, forget_rate=0.8, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)  # не должно упасть с CatBoostError
        _assert_valid_proba(model, X_valid)


# ── 027 BaggingPUClassifier ──────────────────────────────────────────────────

class TestBaggingPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = BaggingPUClassifier(n_estimators=10, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert 0.0 < model.oob_coverage_ <= 1.0
        assert len(model.estimators_) == 10

    def test_rejects_too_few_estimators(self):
        with pytest.raises(ValueError):
            BaggingPUClassifier(n_estimators=1)


# ── 028 SpyPUClassifier ───────────────────────────────────────────────────────

class TestSpyPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SpyPUClassifier(spy_frac=0.15, spy_threshold_pct=10.0, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert model.n_spies_ > 0
        assert model.n_reliable_negative_ >= 0

    def test_rejects_invalid_spy_frac(self):
        with pytest.raises(ValueError):
            SpyPUClassifier(spy_frac=0.9)


# ── 029 ElkanNotoHoldoutPU ────────────────────────────────────────────────────

class TestElkanNotoHoldoutPU:
    def test_fit_predict_and_ci(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = ElkanNotoHoldoutPU(c_holdout_frac=0.3, n_bootstrap=30, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert model.c_ci_[0] <= model.c_ci_[1]
        assert model.c_bootstrap_std_ >= 0.0

    def test_rejects_too_few_bootstrap(self):
        with pytest.raises(ValueError):
            ElkanNotoHoldoutPU(n_bootstrap=1)


# ── 030 NNPUClassifier ────────────────────────────────────────────────────────

class TestNNPUClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = NNPUClassifier(class_prior=0.15, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)

    def test_rejects_invalid_class_prior(self):
        with pytest.raises(ValueError):
            NNPUClassifier(class_prior=1.5)


# ── 034 HeterogeneousStacking ─────────────────────────────────────────────────

class TestHeterogeneousStacking:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'logistic'], n_folds=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert set(model.zoo_used_) == {'catboost', 'lightgbm', 'logistic'}

    def test_missing_xgboost_gracefully_skipped(self, binary_data):
        # xgboost не установлен в тестовом окружении — дефолтный зоопарк должен
        # молча отфильтровать его, а не упасть.
        X_train, y_train, X_valid, y_valid = binary_data
        model = HeterogeneousStacking(n_folds=3)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'xgboost' not in model.zoo_used_

    def test_meta_variants(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        for meta in ('logistic', 'weighted', 'catboost'):
            model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm'], meta=meta, n_folds=3)
            model.fit(X_train, y_train, X_valid, y_valid)
            _assert_valid_proba(model, X_valid)

    def test_rejects_too_small_zoo_after_filtering(self):
        with pytest.raises(ValueError):
            HeterogeneousStacking(base_zoo=['xgboost'])


# ── 035 MultiSeedBlend ────────────────────────────────────────────────────────

class TestMultiSeedBlend:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = MultiSeedBlend(n_seeds=4, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert len(model.seed_scores_) == 4
        assert len(model.models_) == 4


# ── 040 GreedyForwardEnsembleSelection ───────────────────────────────────────

class TestGreedyForwardEnsembleSelection:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        library = []
        for seed in (1, 2, 3, 4):
            m = MultiSeedBlend(n_seeds=2, base_params={**_BP, 'random_seed': seed})
            m.fit(X_train, y_train, X_valid, y_valid)
            library.append(m)

        model = GreedyForwardEnsembleSelection(model_library=library, max_members=3, n_bags=10)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert len(model.weights_) == len(library)
        assert abs(model.weights_.sum() - 1.0) < 1e-9
        assert model.train_pred_ is None

    def test_rejects_too_small_library(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        m = MultiSeedBlend(n_seeds=2, base_params=_BP)
        m.fit(X_train, y_train, X_valid, y_valid)
        with pytest.raises(ValueError):
            GreedyForwardEnsembleSelection(model_library=[m])


# ── 044 DriftRobustClassifier ─────────────────────────────────────────────────

class TestDriftRobustClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = DriftRobustClassifier(target_auc=0.55, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert model.psi_report_ is not None
        assert len(model.adversarial_auc_history_) >= 1

    def test_removes_deliberately_drifted_feature(self):
        rng = np.random.default_rng(2)
        n = 400
        X_train = pd.DataFrame({
            'stable': rng.normal(size=n),
            'drifted': rng.normal(loc=0.0, size=n),
        })
        y_train = pd.Series((rng.random(n) < 0.15).astype(int))
        X_valid = pd.DataFrame({
            'stable': rng.normal(size=100),
            'drifted': rng.normal(loc=8.0, size=100),  # сильный сдвиг среднего
        })
        y_valid = pd.Series((rng.random(100) < 0.15).astype(int))

        model = DriftRobustClassifier(target_auc=0.55, base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert 'drifted' in model.removed_features_
        assert 'stable' in model.selected_features_


# ── 045 AdversarialValidationWeighting ───────────────────────────────────────

class TestAdversarialValidationWeighting:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AdversarialValidationWeighting(base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        _assert_valid_proba(model, X_valid)
        assert abs(float(np.mean(model.weights_)) - 1.0) < 1e-6
        assert model.weight_stats_['min'] >= model.clip_weights[0] / model.weight_stats_['mean_before_norm'] - 1e-6

    def test_detects_real_drift_with_nontrivial_weights(self):
        # Умеренный (не экстремальный) сдвиг: при слишком сильном/равномерном
        # сдвиге все train-строки одинаково "непохожи" на valid и после клипа
        # схлопываются в один и тот же нижний порог -> веса становятся
        # тривиально одинаковыми (1.0 после нормализации) — это ожидаемое
        # поведение клипа, а не то, что здесь проверяется.
        rng = np.random.default_rng(3)
        n = 400
        X_train = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})
        y_train = pd.Series((rng.random(n) < 0.15).astype(int))
        X_valid = pd.DataFrame({'a': rng.normal(loc=1.0, size=100), 'b': rng.normal(loc=1.0, size=100)})
        y_valid = pd.Series((rng.random(100) < 0.15).astype(int))

        model = AdversarialValidationWeighting(base_params=_BP)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.adversarial_auc_ > 0.6
        assert model.weights_.std() > 0.05

    def test_rejects_invalid_clip_weights(self):
        with pytest.raises(ValueError):
            AdversarialValidationWeighting(clip_weights=(2.0, 1.0))
