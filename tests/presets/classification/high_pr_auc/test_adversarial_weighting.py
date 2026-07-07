"""Тесты для AdversarialValidationWeighting

(ml_toolkit/presets/classification/high_pr_auc/adversarial_weighting.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_toolkit.presets.classification.high_pr_auc import AdversarialValidationWeighting
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestAdversarialValidationWeighting:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AdversarialValidationWeighting(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
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

        model = AdversarialValidationWeighting(base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert model.adversarial_auc_ > 0.6
        assert model.weights_.std() > 0.05

    def test_rejects_invalid_clip_weights(self):
        with pytest.raises(ValueError):
            AdversarialValidationWeighting(clip_weights=(2.0, 1.0))
