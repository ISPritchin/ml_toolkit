"""Тесты для SnapshotEnsembleClassifier (ml_toolkit/presets/classification/high_pr_auc/snapshot_ensemble.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import SnapshotEnsembleClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestSnapshotEnsembleClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SnapshotEnsembleClassifier(snapshot_fracs=[0.5, 1.0], base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert len(model.tree_counts_) <= 2
        assert len(model.snapshot_scores_) == len(model.tree_counts_)
        assert 0.0 <= model.ensemble_score_ <= 1.0

    def test_snapshot_fracs_collapse_when_equal_after_rounding(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SnapshotEnsembleClassifier(
            snapshot_fracs=[0.99, 1.0], base_params={**BASE_PARAMS, 'iterations': 10},
        )
        model.fit(X_train, y_train, X_valid, y_valid)
        assert len(model.tree_counts_) == len(set(model.tree_counts_))

    def test_empty_snapshot_fracs_raises(self):
        import pytest
        with pytest.raises(ValueError, match='snapshot_fracs'):
            SnapshotEnsembleClassifier(snapshot_fracs=[])

    def test_out_of_range_snapshot_frac_raises(self):
        import pytest
        with pytest.raises(ValueError, match='snapshot_fracs'):
            SnapshotEnsembleClassifier(snapshot_fracs=[1.5])

    def test_optuna_tunes_architecture(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = SnapshotEnsembleClassifier(snapshot_fracs=[0.5, 1.0], n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
