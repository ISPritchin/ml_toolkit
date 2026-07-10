"""Тесты для EasyEnsembleClassifier (ml_toolkit/presets/classification/high_pr_auc/easy_ensemble.py)."""

from __future__ import annotations

import numpy as np
import optuna
import pytest

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
        # estimator свой срез негативов (SeedSequence.spawn — см. easy_ensemble.py
        # fit()), поэтому наборы отобранных индексов попарно различны.
        y_tr = y_train.values
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = int((y_tr == 1).sum())
        n_neg_sample = min(len(neg_idx), model.neg_ratio * n_pos)
        seed_seqs = np.random.SeedSequence(model.random_seed).spawn(model.n_estimators + 1)[1:]
        samples = []
        for seq in seed_seqs:
            rng = np.random.default_rng(seq)
            samples.append(set(rng.choice(neg_idx, size=n_neg_sample, replace=False).tolist()))
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                assert samples[i] != samples[j], f'negative subsamples {i} and {j} are identical'

    def test_tuning_subsample_independent_from_first_estimator(self, binary_data):
        """Регрессионный тест: подвыборка негативов для тюнинга (rng0) и для
        estimator #0 не должны совпадать. До фикса default_rng(seed) и
        default_rng(seed + 0) были одним и тем же генератором — с одинаковой
        последовательностью вызовов (choice, затем shuffle) подвыборка estimator'а
        #0 побитово совпадала с той, на которой Optuna искала архитектуру, и первый
        member ансамбля переставал быть независимым «взглядом» на негативы.
        """
        X_train, y_train, X_valid, y_valid = binary_data
        model = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, n_optuna_trials=3, random_seed=42)
        model.fit(X_train, y_train, X_valid, y_valid)

        y_tr = y_train.values
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = int((y_tr == 1).sum())
        n_neg_sample = min(len(neg_idx), model.neg_ratio * n_pos)

        tune_seq, est0_seq = np.random.SeedSequence(model.random_seed).spawn(model.n_estimators + 1)[:2]
        tune_sample = set(
            np.random.default_rng(tune_seq).choice(neg_idx, size=n_neg_sample, replace=False).tolist()
        )
        est0_sample = set(
            np.random.default_rng(est0_seq).choice(neg_idx, size=n_neg_sample, replace=False).tolist()
        )
        assert tune_sample != est0_sample

    def test_optuna_params_flow_to_final_estimators(self, binary_data, monkeypatch):
        """Архитектура, выбранная Optuna (best trial), должна долететь до финальных
        estimator'ов как есть — кроме random_seed, который намеренно переопределяется
        per-estimator (см. best_params_['base_params'] в easy_ensemble.py fit()).
        """
        X_train, y_train, X_valid, y_valid = binary_data

        captured = {}
        orig_create_study = optuna.create_study

        def spy_create_study(*args, **kwargs):
            study = orig_create_study(*args, **kwargs)
            captured['study'] = study
            return study

        monkeypatch.setattr(optuna, 'create_study', spy_create_study)

        model = EasyEnsembleClassifier(n_estimators=3, neg_ratio=5, n_optuna_trials=5, random_seed=42)
        model.fit(X_train, y_train, X_valid, y_valid)

        assert 'study' in captured
        best_cb_params = captured['study'].best_trial.user_attrs['cb_params']
        arch_keys = ['iterations', 'max_depth', 'learning_rate', 'l2_leaf_reg', 'subsample', 'min_data_in_leaf']
        for key in arch_keys:
            assert model.best_params_['base_params'][key] == pytest.approx(best_cb_params[key])
