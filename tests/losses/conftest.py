"""Общие helpers для тестов градиентной корректности ml_toolkit/losses/."""

from __future__ import annotations

import numpy as np


def sigmoid(f):
    return 1.0 / (1.0 + np.exp(-f))


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def numeric_der1_binary(loss_obj, f, y, idx, h=1e-6):
    f_plus = f.copy()
    f_plus[idx] += h
    f_minus = f.copy()
    f_minus[idx] -= h
    der1_plus = np.array([d[0] for d in loss_obj.calc_ders_range(f_plus, y, None)])
    der1_minus = np.array([d[0] for d in loss_obj.calc_ders_range(f_minus, y, None)])
    return (der1_plus[idx] - der1_minus[idx]) / (2 * h)
