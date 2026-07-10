"""Тесты для AnomalyBlendClassifier (ml_toolkit/presets/classification/high_pr_auc/anomaly_blend.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import AnomalyBlendClassifier
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_CBT_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'min_data_in_leaf': 5, 'early_stopping_rounds': 20,
    'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'random_seed': 42, 'verbose': 0,
}


class TestAnomalyBlendClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AnomalyBlendClassifier(
            n_if_estimators=30, supervised_params=FAST_CBT_PARAMS, n_alpha_steps=11,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert 0.0 <= model.alpha_ <= 1.0
        assert 0.0 <= model.if_pr_auc_ <= 1.0
        assert 0.0 <= model.sup_pr_auc_ <= 1.0
        assert model.alpha_scan_df_ is not None
        assert len(model.alpha_scan_df_) == 11

    def test_predict_proba_matches_alpha_blend_formula(self, binary_data):
        """predict_proba() должен точно воспроизводить alpha_ * sup + (1-alpha_) * IF."""
        import numpy as np

        X_train, y_train, X_valid, y_valid = binary_data
        model = AnomalyBlendClassifier(n_if_estimators=30, supervised_params=FAST_CBT_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)

        num_feats = [f for f in model.selected_features_ if f not in model.cat_features_]
        sup = model._sup_score(X_valid)
        if_score = model._if_score(X_valid[num_feats].values)
        expected = model.alpha_ * sup + (1 - model.alpha_) * if_score

        np.testing.assert_allclose(model.predict_proba(X_valid), expected)

    def test_optuna_tunes_supervised_model(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = AnomalyBlendClassifier(n_if_estimators=30, n_optuna_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
