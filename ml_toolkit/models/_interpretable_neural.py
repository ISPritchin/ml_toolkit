"""Нейросетевые интерпретируемые модели: GAMINET (объединяет NAM при n_interactions=0).

GAMINET (Yang et al. 2021): отдельная нейросеть на каждый признак + попарные interaction networks.
Предсказание = Σ f_i(x_i) + Σ f_ij(x_i, x_j) + bias.

Optuna тюнит n_interactions ∈ [0, 10]. При n_interactions=0 модель эквивалентна NAM
(Neural Additive Models, Agarwal et al. 2021) — только main effect networks, без взаимодействий.

Требует PyTorch — опциональная зависимость, не входит в pyproject.toml (`pip install torch`).
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    resolve_metric_fn,
    resolve_timeout,
    set_optuna_verbosity,
)

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger(__name__)


def _num_features(selected_features: list[str], cat_features: list[str]) -> list[str]:
    cat_set = set(cat_features)
    return [f for f in selected_features if f not in cat_set]


def _preprocess(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame | None,
    X_inference: pd.DataFrame | None,
    num_feats: list[str],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, SimpleImputer, QuantileTransformer]:
    imputer = SimpleImputer(strategy='median')
    qt = QuantileTransformer(n_quantiles=min(1000, len(X_train)), output_distribution='normal', random_state=42)
    X_tr = qt.fit_transform(imputer.fit_transform(X_train[num_feats].to_numpy(dtype=float)))
    X_va = qt.transform(imputer.transform(X_valid[num_feats].to_numpy(dtype=float))) if X_valid is not None else None
    X_in = qt.transform(imputer.transform(X_inference[num_feats].to_numpy(dtype=float))) if X_inference is not None else None
    return X_tr, X_va, X_in, imputer, qt


# ── Кастомные PyTorch реализации ─────────────────────────────────────────────

def _build_additive_model(n_features: int, hidden_dim: int, n_layers: int, n_interactions: int = 0) -> nn.Module:
    """Строит аддитивную модель: feature networks + опциональные interaction networks.

    При n_interactions=0 эквивалентна NAM (только main effects).
    При n_interactions>0 — GAMINET с попарными взаимодействиями.
    """
    import itertools

    import torch
    from torch import nn

    def _make_net(in_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    n_pairs = min(n_interactions, n_features * (n_features - 1) // 2)
    all_pairs = list(itertools.combinations(range(n_features), 2))
    pairs = all_pairs[:n_pairs]

    class _AdditiveModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_nets = nn.ModuleList([_make_net(1) for _ in range(n_features)])
            self.pair_nets = nn.ModuleList([_make_net(2) for _ in pairs])
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outs = [net(x[:, i:i + 1]) for i, net in enumerate(self.feature_nets)]
            outs += [net(x[:, [i, j]]) for net, (i, j) in zip(self.pair_nets, pairs, strict=False)]
            return torch.stack(outs, dim=1).sum(dim=1).squeeze(-1) + self.bias

    return _AdditiveModel()


def _train_torch_reg(
    model_fn: Callable[[int, int, int], Any],
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_in: np.ndarray,
    hidden_dim: int, n_layers: int, lr: float, n_epochs: int,
    patience: int = 20,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch import nn, optim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = X_tr.shape[1]
    net = model_fn(n_features, hidden_dim, n_layers).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    X_in_t = torch.tensor(X_in, dtype=torch.float32, device=device)

    best_val = float('inf')
    best_state = None
    no_improve = 0

    for _ in range(n_epochs):
        net.train()
        opt.zero_grad()
        loss = nn.functional.l1_loss(net(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            val_mae = nn.functional.l1_loss(net(X_va_t), torch.tensor(y_va, dtype=torch.float32, device=device)).item()
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        tr_pred = net(X_tr_t).cpu().numpy()
        va_pred = net(X_va_t).cpu().numpy()
        in_pred = net(X_in_t).cpu().numpy()

    return net, tr_pred, va_pred, in_pred


# ── Классы (новый API) ────────────────────────────────────────────────────────

class InterpretableNeuralRegressor(BaseModel):
    """GAMINET/NAM для регрессии с подбором архитектуры через Optuna.

    Хранит PyTorch net как _model, _imputer, _qt, _num_feats_.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> InterpretableNeuralRegressor:
        import functools

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[GAMINET Reg] features=%d', len(self._num_feats_))

        X_tr, X_va, _, self._imputer, self._qt = _preprocess(X_train, X_valid, None, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=float)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)

        if self.params is not None:
            p = self.params
            build_fn = functools.partial(_build_additive_model, n_interactions=p.get('n_interactions', 0))
            _va = X_va if X_va is not None else X_tr
            _yva = y_valid.to_numpy(dtype=float) if y_valid is not None else y_tr
            self._model, self.train_pred_, _, _ = _train_torch_reg(
                build_fn, X_tr, y_tr, _va, _yva, _va,
                hidden_dim=p.get('hidden_dim', 64), n_layers=p.get('n_layers', 2),
                lr=p.get('lr', 1e-3), n_epochs=p.get('n_epochs', 100),
            )
            self.best_params_ = self.params
        else:
            if X_va is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            y_va = y_valid.to_numpy(dtype=float)

            def objective(trial: optuna.Trial) -> float:
                hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
                n_layers = trial.suggest_int('n_layers', 1, 4)
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                n_epochs = trial.suggest_int('n_epochs', 50, 300, step=50)
                n_interactions = trial.suggest_int('n_interactions', 0, 10)
                build_fn = functools.partial(_build_additive_model, n_interactions=n_interactions)
                _, _, va_pred, _ = _train_torch_reg(build_fn, X_tr, y_tr, X_va, y_va, X_va,
                                                    hidden_dim=hidden_dim, n_layers=n_layers,
                                                    lr=lr, n_epochs=n_epochs)
                return metric_fn(y_va, va_pred)

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[GAMINET Reg] Best score=%.4f params=%s', study.best_value, self.best_params_)

            bp = self.best_params_
            build_fn = functools.partial(_build_additive_model, n_interactions=bp.get('n_interactions', 0))
            self._model, self.train_pred_, valid_pred, _ = _train_torch_reg(
                build_fn, X_tr, y_tr, X_va, y_va, X_va,
                hidden_dim=bp['hidden_dim'], n_layers=bp['n_layers'],
                lr=bp['lr'], n_epochs=bp['n_epochs'],
            )
            self.valid_pred_ = valid_pred
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        device = next(self._model.parameters()).device
        X_t = self._qt.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        with torch.no_grad():
            return self._model(torch.tensor(X_t, dtype=torch.float32, device=device)).cpu().numpy()


class InterpretableNeuralClassifier(BaseModel):
    """LogisticRegression на QT-признаках (fallback для классификации GAMINET).

    Хранит _model (LogisticRegression), _imputer, _qt, _num_feats_.
    Вероятности калибруются изотонической регрессией.
    params=None → Optuna; params=dict → прямое обучение без тюнинга.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> InterpretableNeuralClassifier:
        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._num_feats_ = _num_features(self.selected_features_, self.cat_features_)
        logger.info('[GAMINET Cls] features=%d (LogisticRegression на QT-признаках)', len(self._num_feats_))

        X_tr, X_va, _, self._imputer, self._qt = _preprocess(X_train, X_valid, None, self._num_feats_)
        y_tr = y_train.to_numpy(dtype=int)

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)

        if self.params is not None:
            direct_params = {'max_iter': 2000, 'class_weight': 'balanced', **self.params}
            self._model = LogisticRegression(**direct_params)
            self._model.fit(X_tr, y_tr)
            self.best_params_ = direct_params
        else:
            if X_va is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')
            y_va = y_valid.to_numpy(dtype=int)

            def objective(trial: optuna.Trial) -> float:
                C = trial.suggest_float('C', 1e-3, 1e2, log=True)
                m = LogisticRegression(C=C, solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
                m.fit(X_tr, y_tr)
                return metric_fn(y_va, m.predict_proba(X_va)[:, 1])

            study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False)
            self.best_params_ = study.best_params
            logger.info('[GAMINET Cls] Best score=%.4f params=%s', study.best_value, self.best_params_)

            self._model = LogisticRegression(
                C=self.best_params_.get('C', 1.0), solver='saga',
                max_iter=2000, class_weight='balanced', random_state=42,
            )
            self._model.fit(X_tr, y_tr)

        self.train_pred_ = self._model.predict_proba(X_tr)[:, 1]
        if X_va is not None:
            self.valid_pred_ = self._model.predict_proba(X_va)[:, 1]
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        X_t = self._qt.transform(self._imputer.transform(X[self._num_feats_].to_numpy(dtype=float)))
        raw = self._model.predict_proba(X_t)[:, 1]
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

