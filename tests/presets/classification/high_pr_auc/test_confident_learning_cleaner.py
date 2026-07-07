"""Тесты для ConfidentLearningCleaner (ml_toolkit/presets/classification/high_pr_auc/confident_learning_cleaner.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_toolkit.presets.classification.high_pr_auc import ConfidentLearningCleaner
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestConfidentLearningCleaner:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = ConfidentLearningCleaner(n_folds=3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
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

        model = ConfidentLearningCleaner(n_folds=5, base_params=BASE_PARAMS)
        model.fit(X_df, y_series, X_valid, y_valid)

        recall = len(set(model.removed_indices_.tolist()) & set(flipped.tolist())) / len(flipped)
        assert recall > 0.3, f'Ожидали найти существенную долю перевёрнутых меток, recall={recall:.2f}'
