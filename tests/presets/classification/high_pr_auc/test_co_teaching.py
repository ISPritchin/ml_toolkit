"""Тесты для CoTeachingClassifier (ml_toolkit/presets/classification/high_pr_auc/co_teaching.py)."""

from __future__ import annotations

from ml_toolkit.presets.classification.high_pr_auc import CoTeachingClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS, assert_valid_proba


class TestCoTeachingClassifier:
    def test_fit_predict(self, binary_data):
        X_train, y_train, X_valid, y_valid = binary_data
        model = CoTeachingClassifier(n_rounds=2, forget_rate=0.3, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)
        assert_valid_proba(model, X_valid)
        assert len(model.round_scores_a_) == 3  # init + 2 раунда
        assert len(model.keep_fraction_history_) == 3

    def test_small_loss_selection_keeps_both_classes(self, binary_data):
        # Регресс: при агрессивном forget_rate + сильном дисбалансе глобальный
        # (не постратный) top-k по loss может целиком вымыть позитивы.
        X_train, y_train, X_valid, y_valid = binary_data
        model = CoTeachingClassifier(n_rounds=3, forget_rate=0.8, base_params=BASE_PARAMS)
        model.fit(X_train, y_train, X_valid, y_valid)  # не должно упасть с CatBoostError
        assert_valid_proba(model, X_valid)
