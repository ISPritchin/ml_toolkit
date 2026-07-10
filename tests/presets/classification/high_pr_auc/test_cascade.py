"""Тесты для TwoStageCascade (ml_toolkit/presets/classification/high_pr_auc/cascade.py)."""

from __future__ import annotations

from catboost import Pool

from ml_toolkit.presets.classification.high_pr_auc import TwoStageCascade
from tests.presets.classification.high_pr_auc.conftest import assert_valid_proba

FAST_STAGE_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'scale_pos_weight': 5.0,
    'l2_leaf_reg': 3.0, 'subsample': 0.8, 'loss_function': 'Logloss', 'eval_metric': 'Recall',
    'early_stopping_rounds': 20, 'random_seed': 42, 'verbose': 0,
}
FAST_STAGE2_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2, 'l2_leaf_reg': 3.0,
    'subsample': 0.8, 'loss_function': 'Logloss', 'eval_metric': 'PRAUC',
    'early_stopping_rounds': 20, 'random_seed': 42, 'verbose': 0,
}


class TestTwoStageCascade:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = TwoStageCascade(
            recall_target=0.80, stage1_params=FAST_STAGE_PARAMS, stage2_params=FAST_STAGE2_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        assert_valid_proba(model, X_valid)
        assert model.model1_ is not None
        assert model.model2_ is not None
        assert 0.0 <= model.stage1_recall_ <= 1.0
        assert 0.0 <= model.stage2_coverage_ <= 1.0

    def test_candidates_score_above_threshold1(self, binary_data):
        """Непрерывный ранкинг: прошедшие Stage 1 маппятся в [threshold1, 1]."""
        X_train, y_train, X_valid, y_valid = binary_data
        model = TwoStageCascade(
            recall_target=0.80, stage1_params=FAST_STAGE_PARAMS, stage2_params=FAST_STAGE2_PARAMS,
        )
        model.fit(X_train, y_train, X_valid, y_valid)

        proba = model.predict_proba(X_valid)
        s1 = model.model1_.predict_proba(
            Pool(X_valid[model.selected_features_], cat_features=model.cat_features_)
        )[:, 1]
        candidates = s1 >= model.threshold1_
        if candidates.any():
            assert (proba[candidates] >= model.threshold1_ - 1e-9).all()

    def test_optuna_stages(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = TwoStageCascade(recall_target=0.80, stage1_n_trials=2, stage2_n_trials=2)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
