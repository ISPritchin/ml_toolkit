"""Смоук-тест fit/predict для DiceLossClassifier (ml_toolkit/presets/classification/high_pr_auc/dice_loss.py)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.high_pr_auc import DiceLossClassifier
from tests.presets.classification.high_pr_auc.conftest import BASE_PARAMS


def test_fit_predict(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    model = DiceLossClassifier(base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))

    pred = model.predict(X_valid, threshold=0.5)
    assert set(np.unique(pred)) <= {0, 1}
    assert model.train_pred_.shape == (len(X_train),)
