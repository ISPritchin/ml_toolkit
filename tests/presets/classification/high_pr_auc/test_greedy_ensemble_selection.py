"""Тесты для GreedyForwardEnsembleSelection

(ml_toolkit/presets/classification/high_pr_auc/greedy_ensemble_selection.py).
"""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.high_pr_auc import (
    GreedyForwardEnsembleSelection,
    MultiSeedBlend,
)
from tests.presets.classification.high_pr_auc.conftest import (
    BASE_PARAMS,
    assert_valid_proba,
)


class TestGreedyForwardEnsembleSelection:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        library = []
        for seed in (1, 2, 3, 4):
            m = MultiSeedBlend(n_seeds=2, base_params={**BASE_PARAMS, 'random_seed': seed})
            m.fit(X_train, y_train, X_valid, y_valid)
            library.append(m)

        model = GreedyForwardEnsembleSelection(model_library=library, max_members=3, n_bags=10)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert len(model.weights_) == len(library)
        assert abs(model.weights_.sum() - 1.0) < 1e-9
        assert model.train_pred_ is None

    def test_rejects_too_small_library(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        m = MultiSeedBlend(n_seeds=2, base_params=BASE_PARAMS)
        m.fit(X_train, y_train, X_valid, y_valid)
        with pytest.raises(ValueError):
            GreedyForwardEnsembleSelection(model_library=[m])
