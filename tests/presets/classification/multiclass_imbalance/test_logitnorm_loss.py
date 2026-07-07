"""Смоук-тест fit/predict для LogitNormLossClassifier

(ml_toolkit/presets/classification/multiclass_imbalance/logitnorm_loss.py).
"""

from __future__ import annotations

import numpy as np

from ml_toolkit.presets.classification.multiclass_imbalance import LogitNormLossClassifier
from tests.presets.classification.multiclass_imbalance.conftest import BASE_PARAMS


def test_fit_predict(multiclass_data):
    X_train, y_train, X_valid, y_valid = multiclass_data
    model = LogitNormLossClassifier(base_params=BASE_PARAMS)
    model.fit(X_train, y_train, X_valid, y_valid)

    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    pred = model.predict(X_valid)
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
    assert model.n_classes_ == 4
