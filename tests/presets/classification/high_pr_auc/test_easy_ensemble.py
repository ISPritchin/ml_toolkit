"""Тесты для EasyEnsembleClassifier (ml_toolkit/presets/classification/high_pr_auc/easy_ensemble.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import EasyEnsembleClassifier
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestEasyEnsembleClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = EasyEnsembleClassifier(n_estimators=5, neg_ratio=5, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert len(model.estimators_) == 5
        assert len(model.estimator_scores_) == 5

    def test_estimators_are_diverse(self, binary_data):
        """Каждый estimator обучен на своём подсэмпле негативов (свой seed rng) +
        своём random_seed модели — предсказания разных estimator'ов на одном X
        не должны совпадать между собой.
        """
        X_train, y_train, X_valid, y_valid = binary_data
        model = EasyEnsembleClassifier(n_estimators=5, neg_ratio=5, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        X_va_feats = X_valid[model.selected_features_]
        raw_scores = [model._predict_one(est, X_va_feats) for est in model.estimators_]

        n = len(raw_scores)
        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(raw_scores[i], raw_scores[j]), (
                    f'estimators {i} and {j} produce identical predictions — no diversity'
                )

        # Дополнительная проверка на уровне обучающих подвыборок: у каждого
        # estimator свой срез негативов, поэтому наборы отобранных индексов различны.
        y_tr = y_train.values
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = int((y_tr == 1).sum())
        n_neg_sample = min(len(neg_idx), model.neg_ratio * n_pos)
        samples = []
        for i in range(model.n_estimators):
            rng = np.random.default_rng(model.random_seed + i)
            samples.append(set(rng.choice(neg_idx, size=n_neg_sample, replace=False).tolist()))
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                assert samples[i] != samples[j], f'negative subsamples {i} and {j} are identical'
