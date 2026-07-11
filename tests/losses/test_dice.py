"""Тесты градиентной корректности ml_toolkit/losses/_dice.py (DiceLoss)."""

from __future__ import annotations

import numpy as np

from ml_toolkit.losses import DiceLoss, TverskyLoss


class TestDiceLoss:
    def test_equivalent_to_tversky_half_half(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=50)
        y = (rng.random(50) < 0.4).astype(np.float64)
        dice = DiceLoss(smooth=1.0)
        tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)
        der1_d, der2_d = zip(*dice.calc_ders_range(f, y, None), strict=False)
        der1_t, der2_t = zip(*tversky.calc_ders_range(f, y, None), strict=False)
        assert np.allclose(der1_d, der1_t)
        assert np.allclose(der2_d, der2_t)
