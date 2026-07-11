"""Тесты общего поведения _CustomLossClassifierMulticlassBase.

(ml_toolkit/presets/classification/multiclass_imbalance/_custom_loss_base.py),
проверенного через конкретный сабкласс EqualizationLossClassifier.
"""

from __future__ import annotations

import pytest

from ml_toolkit.presets.classification.multiclass_imbalance import (
    EqualizationLossClassifier,
)
from tests.presets.classification.multiclass_imbalance.conftest import BASE_PARAMS


def test_rejects_binary_target(binary_data):
    X_train, y_train, X_valid, y_valid = binary_data
    model = EqualizationLossClassifier(base_params=BASE_PARAMS)
    with pytest.raises(ValueError, match='рассчитан на мультикласс'):
        model.fit(X_train, y_train, X_valid, y_valid)
