"""TabM adapter (ICLR 2025, yandex-research/tabm, pip install tabm).

Архитектура: параметрически-эффективный ансамбль MLP (BatchEnsemble).
- Numeric features: QuantileTransformer → float32.
- Categorical features: OrdinalEncoder → int64 (cardinalities передаются в модель).
- Regression: label standardisation (mean/std на train).
- Optuna: k, lr, weight_decay, d_block, n_blocks, dropout.
- Early stopping по валидационному MAE (reg) / PR-AUC (cls).

model_settings keys (опционально):
    n_epochs_per_trial  — эпох за Optuna trial (default 100)
    n_epochs_final      — эпох для финального обучения (default 1000)
    patience            — терпение early stopping (default 16)
    device              — 'auto'|'cpu'|'cuda' (default 'auto')
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
import torch
from torch import nn

from ml_toolkit.models._base import BaseModel
from ml_toolkit.models._utils import (
    CLS_METRICS,
    REG_METRICS,
    fit_calibrator,
    resolve_metric_fn,
    resolve_pruner,
    resolve_timeout,
    set_optuna_verbosity,
)

if TYPE_CHECKING:
    from tabm import TabM

logger = logging.getLogger(__name__)



# ─── preprocessing ────────────────────────────────────────────────────────────

class _Preprocessor:
    """Препроцессор признаков для TabM: обучается на train, применяется ко всем сплитам.

    Числовые признаки: NaN-импутация медианой + QuantileTransformer (normal distribution).
    Категориальные признаки: OrdinalEncoder с unknown_value=-1.

    Attributes:
        num_features: Список числовых признаков.
        cat_features: Список категориальных признаков.
        cat_cardinalities: Количество уникальных значений каждого категориального признака.

    """

    def __init__(self, num_features: list[str], cat_features: list[str], n_train: int) -> None:
        self.num_features = num_features
        self.cat_features = cat_features
        n_quantiles = max(min(n_train // 30, 1000), 10)
        self._qt = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution='normal', subsample=10 ** 9
        )
        self._oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X: pd.DataFrame) -> _Preprocessor:
        """Обучает QuantileTransformer и OrdinalEncoder на обучающей выборке."""
        if self.num_features:
            vals = X[self.num_features].values.astype(np.float32)
            # column-wise median for NaN imputation (computed on non-NaN values)
            col_medians = np.nanmedian(vals, axis=0)
            nan_mask = np.isnan(vals)
            vals[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
            self._num_medians = col_medians
            noise = np.random.default_rng(0).normal(0, 1e-5, vals.shape).astype(np.float32)
            self._qt.fit(vals + noise)
        if self.cat_features:
            self._oe.fit(X[self.cat_features].astype(str))
        self.cat_cardinalities: list[int] = [
            len(cats) for cats in self._oe.categories_
        ] if self.cat_features else []
        return self

    def transform(self, X: pd.DataFrame, device: torch.device) -> dict[str, torch.Tensor]:
        """Преобразует DataFrame в тензоры для TabM.

        Args:
            X: Входной DataFrame.
            device: PyTorch-устройство для размещения тензоров.

        Returns:
            Словарь с ключами `'x_num'` (float32) и/или `'x_cat'` (int64).

        """
        result: dict[str, torch.Tensor] = {}
        if self.num_features:
            vals = X[self.num_features].values.astype(np.float32)
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                vals[nan_mask] = np.take(self._num_medians, np.where(nan_mask)[1])
            x_num = self._qt.transform(vals)
            result['x_num'] = torch.as_tensor(x_num, dtype=torch.float32, device=device)
        if self.cat_features:
            x_cat = self._oe.transform(X[self.cat_features].astype(str)).astype(np.int64)
            result['x_cat'] = torch.as_tensor(x_cat, dtype=torch.int64, device=device)
        return result


# ─── training loop ─────────────────────────────────────────────────────────────

def _make_model(
    tabm: ModuleType,
    n_num: int,
    cat_cardinalities: list[int],
    d_out: int,
    k: int,
    d_block: int,
    n_blocks: int,
    dropout: float,
    device: torch.device,
) -> TabM:
    """Создаёт и переносит на `device` экземпляр TabM-модели с заданной архитектурой."""
    model = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        arch_type='tabm',
        k=k,
        d_block=d_block,
        n_blocks=n_blocks,
        dropout=dropout,
    )
    return model.to(device)


def _forward(model: TabM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Выполняет forward-pass TabM; возвращает тензор формы (B, k, d_out)."""
    return model(
        batch.get('x_num'),
        batch.get('x_cat'),
    )  # → (B, k, d_out)


def _train_epoch(
    model: TabM,
    data: dict[str, torch.Tensor],
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
) -> None:
    """Выполняет одну эпоху обучения TabM с mini-batch SGD и gradient clipping (norm=1.0).

    Args:
        model: TabM-модель в режиме train.
        data: Тензоры признаков (x_num и/или x_cat) на нужном устройстве.
        y: Тензор целевой переменной.
        optimizer: Оптимизатор (AdamW).
        loss_fn: Функция потерь (MSE для регрессии, BCE для классификации).
        batch_size: Размер мини-батча.

    """
    model.train()
    k = model.backbone.k
    n = len(y)
    perm = torch.randperm(n, device=y.device)
    for idx in perm.split(batch_size):
        batch = {key: val[idx] for key, val in data.items()}
        y_batch = y[idx]
        optimizer.zero_grad()
        y_pred = _forward(model, batch).squeeze(-1)  # (B, k)
        y_pred_flat = y_pred.flatten(0, 1)           # (B*k,)
        y_true_flat = y_batch.repeat_interleave(k)   # (B*k,)
        loss_fn(y_pred_flat, y_true_flat).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.inference_mode()
def _predict_raw(model: TabM, data: dict[str, torch.Tensor], batch_size: int = 8192) -> np.ndarray:
    """Returns (N, k) raw predictions."""
    model.eval()
    n = len(next(iter(data.values())))
    parts = []
    for idx in torch.arange(n, device=next(iter(data.values())).device).split(batch_size):
        batch = {k: v[idx] for k, v in data.items()}
        parts.append(_forward(model, batch).squeeze(-1).cpu().float())
    return torch.cat(parts).numpy()  # (N, k)


def _avg_pred(raw: np.ndarray, task_type: str) -> np.ndarray:
    """Average k predictions: mean for regression, softmax-then-mean for classification."""
    if task_type == 'regression':
        return raw.mean(axis=1)
    # binary classification: sigmoid → mean probability
    return 1.0 / (1.0 + np.exp(-raw.mean(axis=1)))


def _run_training(
    tabm: ModuleType,
    n_num: int,
    cat_cardinalities: list[int],
    d_out: int,
    k: int,
    d_block: int,
    n_blocks: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    task_type: str,
    data_tr: dict[str, torch.Tensor],
    data_va: dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    y_va_np: np.ndarray,
    y_stats: tuple[float, float] | None,
    n_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    X_valid_full: pd.DataFrame,
    postprocess_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
    metric_fn: Callable | None = None,
    direction: str | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[TabM, float]:
    """Train one TabM model, return (best_model, best_val_score)."""
    model = _make_model(tabm, n_num, cat_cardinalities, d_out, k, d_block, n_blocks, dropout, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task_type == 'regression':
        loss_fn = nn.functional.mse_loss
    else:
        # балансировка классов: вес положительного класса = n_neg / n_pos (аналог
        # class_weight='balanced' у sklearn), считается один раз по всей обучающей выборке.
        n_pos = float(y_tr.sum().item())
        n_neg = float(y_tr.numel()) - n_pos
        pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

    _init_dir = direction if direction is not None else ('minimize' if task_type == 'regression' else 'maximize')
    best_score = float('inf') if _init_dir == 'minimize' else -float('inf')
    best_state: dict = deepcopy(model.state_dict())  # always have a valid fallback
    no_improve = 0

    for epoch in range(n_epochs):
        _train_epoch(model, data_tr, y_tr, optimizer, loss_fn, batch_size)

        raw_va = _predict_raw(model, data_va)
        pred_va = _avg_pred(raw_va, task_type)

        if task_type == 'regression':
            if y_stats is None:
                raise ValueError('y_stats required for regression')
            pred_denorm = pred_va * y_stats[1] + y_stats[0]
            pred_pp = postprocess_fn(X_valid_full, pred_denorm) if postprocess_fn else pred_denorm
            _reg_fn = metric_fn if metric_fn is not None else REG_METRICS['mae'][0]
            _dir = direction if direction is not None else 'minimize'
            score = float(_reg_fn(y_va_np, pred_pp))
            improved = np.isfinite(score) and (score < best_score if _dir == 'minimize' else score > best_score)
        else:
            pred_va_safe = np.nan_to_num(pred_va, nan=0.5)
            _cls_fn = metric_fn if metric_fn is not None else CLS_METRICS['pr_auc'][0]
            _dir = direction if direction is not None else 'maximize'
            score = float(_cls_fn(y_va_np, pred_va_safe))
            improved = np.isfinite(score) and (score < best_score if _dir == 'minimize' else score > best_score)

        if trial is not None:
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f'Trial pruned at epoch {epoch}.')

        if improved:
            best_score = score
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_score


# ─── helpers ──────────────────────────────────────────────────────────────────

def _resolve_device(setting: str) -> torch.device:
    """Определяет PyTorch-устройство по строке `'auto'`, `'cpu'` или `'cuda'`."""
    if setting == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(setting)


# ─── классы (новый API) ───────────────────────────────────────────────────────

def _tabm_optuna_params(trial: optuna.Trial) -> dict:
    return {
        'k': trial.suggest_categorical('k', [8, 16, 32, 64]),
        'd_block': trial.suggest_categorical('d_block', [64, 128, 256, 512]),
        'n_blocks': trial.suggest_int('n_blocks', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 5e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
    }


class TabMRegressor(BaseModel):
    """TabM для регрессии. Архитектурные гиперпараметры подбираются через Optuna (params=None).

    Хранит: _model (TabM PyTorch), _prep (_Preprocessor), _y_stats (mean, std), _device.
    model_settings['postprocess_fn'] — опциональная постобработка (X, pred) → pred.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> TabMRegressor:
        try:
            import tabm
        except ImportError as err:
            raise ImportError('tabm not installed. Run: pip install tabm') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._device = _resolve_device(ms.get('device', 'auto'))
        n_epochs_trial = int(ms.get('n_epochs_per_trial', 150))
        n_epochs_final = int(ms.get('n_epochs_final', 1000))
        patience = int(ms.get('patience', 16))
        batch_size = int(ms.get('batch_size', 256))
        postprocess_fn = ms.get('postprocess_fn', None)

        metric_fn, direction = resolve_metric_fn(ms, 'reg_metric', REG_METRICS['mae'][0], 'minimize', REG_METRICS)
        logger.info('[TabM Reg] device=%s', self._device)

        cat_in_sel = [f for f in self.cat_features_ if f in self.selected_features_]
        num_features = [f for f in self.selected_features_ if f not in cat_in_sel]
        self._prep = _Preprocessor(num_features, cat_in_sel, n_train=len(X_train)).fit(X_train)

        data_tr = self._prep.transform(X_train, self._device)
        data_va = self._prep.transform(X_valid, self._device) if X_valid is not None else data_tr

        y_mean = float(y_train.mean())
        y_std = float(y_train.std()) or 1.0
        self._y_stats = (y_mean, y_std)
        y_tr = torch.as_tensor(((y_train.values - y_mean) / y_std).astype(np.float32), device=self._device)
        y_va_np = y_valid.values.astype(np.float64) if y_valid is not None else y_train.values.astype(np.float64)

        n_num = len(num_features)
        card = self._prep.cat_cardinalities

        _kw = dict(
            task_type='regression', data_tr=data_tr, data_va=data_va,
            y_tr=y_tr, y_va_np=y_va_np, y_stats=self._y_stats,
            patience=patience, batch_size=batch_size, device=self._device,
            X_valid_full=X_valid if X_valid is not None else X_train,
            postprocess_fn=postprocess_fn, metric_fn=metric_fn, direction=direction,
        )

        if self.params is not None:
            p = self.params
            self._model, _ = _run_training(
                tabm, n_num, card, 1, k=p['k'], d_block=p['d_block'], n_blocks=p['n_blocks'],
                dropout=p.get('dropout', 0.0), lr=p.get('lr', 1e-3), weight_decay=p.get('weight_decay', 1e-4),
                n_epochs=n_epochs_final, **_kw,
            )
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')

            def objective(trial: optuna.Trial) -> float:
                p = _tabm_optuna_params(trial)
                _, score = _run_training(tabm, n_num, card, 1, **p, n_epochs=n_epochs_trial, trial=trial, **_kw)
                return score

            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
            )
            study.optimize(
                objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False,
            )
            bp = study.best_params
            self.best_params_ = {**bp, 'n_epochs_final': n_epochs_final, 'patience': patience,
                                  'device': str(self._device), 'batch_size': batch_size}
            logger.info('[TabM Reg] Best score=%.4f params=%s', study.best_value, bp)
            self._model, _ = _run_training(
                tabm, n_num, card, 1, k=bp['k'], d_block=bp['d_block'], n_blocks=bp['n_blocks'],
                dropout=bp['dropout'], lr=bp['lr'], weight_decay=bp['weight_decay'],
                n_epochs=n_epochs_final, **_kw,
            )

        _pp = postprocess_fn or (lambda _X, p: p)

        def _denorm(data: dict) -> np.ndarray:
            raw = _avg_pred(_predict_raw(self._model, data), 'regression') * y_std + y_mean
            return np.nan_to_num(raw, nan=y_mean, posinf=y_mean, neginf=y_mean)

        self.train_pred_ = _pp(X_train, _denorm(data_tr))
        if X_valid is not None:
            self.valid_pred_ = _pp(X_valid, _denorm(data_va))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        data = self._prep.transform(X, self._device)
        y_mean, y_std = self._y_stats
        raw = _avg_pred(_predict_raw(self._model, data), 'regression') * y_std + y_mean
        return np.nan_to_num(raw, nan=y_mean, posinf=y_mean, neginf=y_mean)


class TabMClassifier(BaseModel):
    """TabM для бинарной классификации. Гиперпараметры подбираются через Optuna (params=None).

    Вероятности калибруются изотонической регрессией на валидационной выборке.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        selected_features: list[str] | None = None,
        cat_features: list[str] | None = None,
    ) -> TabMClassifier:
        try:
            import tabm
        except ImportError as err:
            raise ImportError('tabm not installed. Run: pip install tabm') from err

        X_train, y_train, X_valid, y_valid = self._coerce_inputs(X_train, y_train, X_valid, y_valid)
        self.selected_features_ = self._resolve_features(X_train, selected_features)
        self.cat_features_ = list(cat_features or [])
        ms = self.model_settings
        _optuna_prev_verbosity = set_optuna_verbosity(ms)

        self._device = _resolve_device(ms.get('device', 'auto'))
        n_epochs_trial = int(ms.get('n_epochs_per_trial', 100))
        n_epochs_final = int(ms.get('n_epochs_final', 1000))
        patience = int(ms.get('patience', 16))
        batch_size = int(ms.get('batch_size', 256))

        metric_fn, direction = resolve_metric_fn(ms, 'cls_metric', CLS_METRICS['pr_auc'][0], 'maximize', CLS_METRICS)
        logger.info('[TabM Cls] device=%s', self._device)

        cat_in_sel = [f for f in self.cat_features_ if f in self.selected_features_]
        num_features = [f for f in self.selected_features_ if f not in cat_in_sel]
        self._prep = _Preprocessor(num_features, cat_in_sel, n_train=len(X_train)).fit(X_train)

        data_tr = self._prep.transform(X_train, self._device)
        data_va = self._prep.transform(X_valid, self._device) if X_valid is not None else data_tr

        y_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        y_tr = torch.as_tensor(y_arr.astype(np.float32), device=self._device)
        y_va_np = (y_valid.values if y_valid is not None else y_arr).astype(np.float64)

        n_num = len(num_features)
        card = self._prep.cat_cardinalities

        _kw = dict(
            task_type='classification', data_tr=data_tr, data_va=data_va,
            y_tr=y_tr, y_va_np=y_va_np, y_stats=None,
            patience=patience, batch_size=batch_size, device=self._device,
            X_valid_full=X_valid if X_valid is not None else X_train,
            metric_fn=metric_fn, direction=direction,
        )

        if self.params is not None:
            p = self.params
            self._model, _ = _run_training(
                tabm, n_num, card, 1, k=p['k'], d_block=p['d_block'], n_blocks=p['n_blocks'],
                dropout=p.get('dropout', 0.0), lr=p.get('lr', 1e-3), weight_decay=p.get('weight_decay', 1e-4),
                n_epochs=n_epochs_final, **_kw,
            )
            self.best_params_ = self.params
        else:
            if X_valid is None:
                raise ValueError('X_valid обязателен при params=None (режим Optuna)')

            def objective(trial: optuna.Trial) -> float:
                p = _tabm_optuna_params(trial)
                _, score = _run_training(tabm, n_num, card, 1, **p, n_epochs=n_epochs_trial, trial=trial, **_kw)
                return score

            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42), pruner=resolve_pruner(ms),
            )
            study.optimize(
                objective, n_trials=max(1, self.n_optuna_trials), timeout=resolve_timeout(ms), show_progress_bar=False,
            )
            bp = study.best_params
            self.best_params_ = {**bp, 'n_epochs_final': n_epochs_final, 'patience': patience,
                                  'device': str(self._device), 'batch_size': batch_size}
            logger.info('[TabM Cls] Best score=%.4f params=%s', study.best_value, bp)
            self._model, _ = _run_training(
                tabm, n_num, card, 1, k=bp['k'], d_block=bp['d_block'], n_blocks=bp['n_blocks'],
                dropout=bp['dropout'], lr=bp['lr'], weight_decay=bp['weight_decay'],
                n_epochs=n_epochs_final, **_kw,
            )

        self.train_pred_ = np.nan_to_num(_avg_pred(_predict_raw(self._model, data_tr), 'classification'), nan=0.5)
        if X_valid is not None:
            self.valid_pred_ = np.nan_to_num(_avg_pred(_predict_raw(self._model, data_va), 'classification'), nan=0.5)
            self.calibrator_ = fit_calibrator(self.valid_pred_, y_valid.to_numpy(dtype=int))
        optuna.logging.set_verbosity(_optuna_prev_verbosity)
        return self

    def _predict_proba_impl(self, X: pd.DataFrame) -> np.ndarray:
        data = self._prep.transform(X, self._device)
        raw = np.nan_to_num(_avg_pred(_predict_raw(self._model, data), 'classification'), nan=0.5)
        return self.calibrator_.predict(raw) if self.calibrator_ is not None else raw

