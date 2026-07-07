"""Визуализация важности признаков и интерпретация моделей.

Поддерживаемые адаптеры:
    catboost       — встроенная важность (PredictionValuesChange) + SHAP TreeExplainer
    lightgbm/_dart/_goss — gain importance + SHAP TreeExplainer
    xgboost        — gain importance + SHAP TreeExplainer
    random_forest  — feature_importances_ + SHAP TreeExplainer
    extra_trees    — feature_importances_ + SHAP TreeExplainer
    hist_gbm       — feature_importances_ + SHAP TreeExplainer
    quantile_forest — feature_importances_ + SHAP TreeExplainer
    oblique_forest  — feature_importances_ + SHAP TreeExplainer
    decision_tree  — feature_importances_ + SHAP TreeExplainer
    mondrian       — permutation importance (SHAP не поддерживается)
    lama           — permutation importance; predict_fn обязателен
    tabm           — permutation importance; predict_fn обязателен
    linear (все)   — |coef_|
    прочие интерпретируемые — permutation importance; predict_fn обязателен

Точка входа: ``plot_feature_importance()``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error
from tqdm import tqdm

from ml_toolkit.models import ALL_TREE_NAMES as _TREE_MODELS, LIGHTGBM_VARIANTS as _LIGHTGBM_VARIANTS, SKLEARN_TREE_NAMES as _SKLEARN_TREE_MODELS

logger = logging.getLogger(__name__)
_TOP_N = 30


def _linear_importance(model_tuple: tuple, feature_names: list[str]) -> np.ndarray:
    """Важность признаков линейной модели: |coef_|, выровненная с feature_names.

    Модель хранится как (sklearn_estimator, prep_pipeline, num_feature_names).
    Признаки, исключённые из num_features (категориальные), получают 0.
    """
    lin_model, _prep, num_features = model_tuple
    coef = np.abs(lin_model.coef_).flatten()
    nf_index = {f: i for i, f in enumerate(num_features)}
    return np.array([coef[nf_index[f]] if f in nf_index else 0.0 for f in feature_names])


# ── individual SHAP waterfall ─────────────────────────────────────────────────

def plot_shap_individuals(
    model: Any,
    model_name: str,
    feature_names: list[str],
    X_sample: pd.DataFrame,
    save_path: Path | str,
    n_show: int = 12,
    n_features: int = 15,
    task: str = 'regression',
) -> None:
    """Строит waterfall-диаграммы SHAP для отдельных наблюдений (для tree-моделей).

    Сохраняет один PNG с сеткой ``ceil(n_show/3)`` строк × 3 столбца.
    Для LAMA и TabM функция молча завершается (SHAP не поддерживается).

    Args:
        model: Обученная tree-модель.
        model_name: Имя адаптера.
        feature_names: Список признаков модели.
        X_sample: DataFrame клиентов для объяснения (≥ n_show строк).
        save_path: Путь сохранения PNG.
        n_show: Число клиентов на графике.
        n_features: Число признаков в каждом waterfall.
        task: 'regression' или 'classification'.
    """
    # Mondrian не поддерживает TreeExplainer
    _shap_supported = _TREE_MODELS - {'mondrian'}
    if model_name not in _shap_supported:
        logger.debug('SHAP individuals: skipped for %s', model_name)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import shap

        X_sub = X_sample[feature_names].head(n_show)

        if model_name == 'catboost':
            # New-style __call__ принимает DataFrame напрямую (Pool не нужен)
            explainer = shap.TreeExplainer(model)
            sv = explainer(X_sub)
        elif model_name in _LIGHTGBM_VARIANTS:
            explainer = shap.TreeExplainer(model)
            sv_raw = explainer.shap_values(X_sub)
            if isinstance(sv_raw, list):
                sv_raw = sv_raw[1] if len(sv_raw) > 1 else sv_raw[0]
            ev = explainer.expected_value
            ev_arr = np.asarray(ev).ravel()
            ev_scalar = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
            sv = shap.Explanation(
                values=sv_raw,
                base_values=np.full(len(X_sub), ev_scalar),
                data=X_sub.values,
                feature_names=list(feature_names),
            )
        elif model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
            sv_raw = explainer.shap_values(X_sub)
            sv = shap.Explanation(
                values=sv_raw,
                base_values=np.full(len(X_sub), float(explainer.expected_value)),
                data=X_sub.values,
                feature_names=list(feature_names),
            )
        elif model_name in _SKLEARN_TREE_MODELS:
            actual = model[0] if isinstance(model, tuple) else model
            if hasattr(actual, 'named_steps'):
                X_sub = pd.DataFrame(
                    actual[:-1].transform(X_sub), columns=X_sub.columns,
                )
                actual = actual.named_steps.get('estimator', actual[-1])
            explainer = shap.TreeExplainer(actual)
            sv_raw = explainer.shap_values(X_sub)
            # Новые версии SHAP возвращают 3D (n_samples, n_features, n_classes) для классификаторов;
            # старые — list[n_classes × (n_samples, n_features)]. Берём class-1 либо единственный.
            if isinstance(sv_raw, list):
                sv_raw = sv_raw[1] if len(sv_raw) > 1 else sv_raw[0]
            elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
                sv_raw = sv_raw[:, :, 1]
            ev = explainer.expected_value
            # expected_value может быть: scalar, list, 1-elem array (reg), или 2-elem array (cls)
            ev_arr = np.asarray(ev).ravel()
            ev_scalar = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
            sv = shap.Explanation(
                values=sv_raw,
                base_values=np.full(len(X_sub), ev_scalar),
                data=X_sub.values,
                feature_names=list(feature_names),
            )
        else:
            return

        import io

        # shap.waterfall_plot всегда создаёт собственную фигуру; рендерим каждую в буфер
        # (plt.gcf() после вызова содержит SHAP-фигуру), затем вставляем в нашу сетку
        images: list[np.ndarray] = []
        for i in range(n_show):
            shap.waterfall_plot(sv[i], max_display=n_features, show=False)
            shap_fig = plt.gcf()  # SHAP создал эту фигуру
            buf = io.BytesIO()
            shap_fig.savefig(buf, format='png', dpi=250, bbox_inches='tight')
            buf.seek(0)
            plt.close(shap_fig)
            images.append(plt.imread(buf))

        n_cols = 3
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 4.5))
        axes_flat = np.array(axes).flatten()

        for i, img in enumerate(images):
            axes_flat[i].imshow(img, aspect='auto')
            axes_flat[i].axis('off')

        for ax in axes_flat[n_show:]:
            ax.set_visible(False)

        fig.suptitle(
            f'SHAP Waterfall — {model_name.upper()} [{task.capitalize()}] · {n_show} observations',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        logger.info('shap_individuals_%s.png → %s', task, save_path)

    except Exception:
        logger.exception('SHAP individuals failed for %s (%s)', model_name, task)


# ── importance computation ────────────────────────────────────────────────────

def _tree_importance(model: Any, model_name: str, feature_names: list[str]) -> np.ndarray | None:
    """Встроенная важность для tree-моделей; возвращает массив, выровненный с feature_names.

    Возвращает None если встроенная важность недоступна (напр. HistGBM на старом sklearn).
    """
    if model_name == 'catboost':
        return np.array(model.get_feature_importance())

    if model_name in _LIGHTGBM_VARIANTS:
        booster = model.booster_ if hasattr(model, 'booster_') else model
        return np.array(booster.feature_importance(importance_type='gain'), dtype=float)

    if model_name == 'xgboost':
        scores = model.get_booster().get_score(importance_type='gain')
        return np.array([scores.get(f, 0.0) for f in feature_names], dtype=float)

    if model_name in _SKLEARN_TREE_MODELS:
        # quantile_forest classification возвращает кортеж (qrf, clf, imp) — берём qrf
        actual = model[0] if isinstance(model, tuple) else model
        # RF/ET/OF/QRF возвращаются как Pipeline([imputer, estimator])
        if hasattr(actual, 'named_steps'):
            actual = actual.named_steps.get('estimator', actual[-1])
        if hasattr(actual, 'feature_importances_'):
            return actual.feature_importances_
        # HistGBM в sklearn < 1.4 не имеет feature_importances_ — сигнализируем None для fallback
        return None

    raise ValueError(f'Unknown tree model: {model_name!r}')


def _permutation_importance(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    feature_names: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
    n_repeats: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Permutation importance для моделей без встроенного механизма.

    Для регрессии: ΔMae = MAE_permuted − MAE_base  (выше → признак важнее).
    Для классификации: ΔPR-AUC = PR-AUC_base − PR-AUC_permuted  (выше → важнее).

    verbose: показывать tqdm-прогресс по признакам (n_features × n_repeats
        вызовов predict_fn — может быть медленно на больших моделях/выборках).
    """
    X_work = X_val[feature_names].copy()
    base_pred = predict_fn(X_work)
    if task == 'regression':
        base_score = mean_absolute_error(y_val, base_pred)
        def delta(pred: np.ndarray) -> float:
            return float(mean_absolute_error(y_val, pred) - base_score)
    else:
        base_score = average_precision_score(y_val, base_pred)
        def delta(pred: np.ndarray) -> float:
            return float(base_score - average_precision_score(y_val, pred))

    rng = np.random.default_rng(42)
    deltas = np.zeros((len(feature_names), n_repeats), dtype=float)
    for fi, feat in enumerate(tqdm(feature_names, desc='permutation importance', disable=not verbose)):
        original = X_work[feat].values.copy()
        for rep in range(n_repeats):
            X_work[feat] = rng.permutation(original)
            try:
                deltas[fi, rep] = delta(predict_fn(X_work))
            except Exception:
                deltas[fi, rep] = 0.0
            finally:
                X_work[feat] = original
    return deltas.mean(axis=1)


# ── SHAP ──────────────────────────────────────────────────────────────────────

def _try_shap_plot(
    ax: plt.Axes,
    model: Any,
    model_name: str,
    feature_names: list[str],
    X_val: pd.DataFrame,
    task: str,
    top_n: int,
) -> bool:
    """Рисует SHAP beeswarm в `ax`. Возвращает True при успехе."""
    try:
        import shap

        X_sample = X_val[feature_names]

        if model_name == 'catboost':
            from catboost import Pool
            cat_idx = list(model.get_cat_feature_indices())
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Pool(X_sample, cat_features=cat_idx))
        elif model_name in _LIGHTGBM_VARIANTS:
            explainer = shap.TreeExplainer(model)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='shap')
                sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
        elif model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
        elif model_name in _SKLEARN_TREE_MODELS:
            actual = model[0] if isinstance(model, tuple) else model
            if hasattr(actual, 'named_steps'):
                # Pipeline: impute X, then use underlying estimator
                X_sample = pd.DataFrame(
                    actual[:-1].transform(X_sample), columns=X_sample.columns,
                )
                actual = actual.named_steps.get('estimator', actual[-1])
            explainer = shap.TreeExplainer(actual)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            return False

        if not isinstance(sv, np.ndarray) or sv.ndim != 2:
            return False

        # SHAP рисует на текущих осях (plt.gca()); делаем ax текущими
        plt.sca(ax)
        with warnings.catch_warnings():
            # SHAP внутренне вызывает np.random.seed() — предупреждение NumPy о глобальном RNG
            warnings.filterwarnings('ignore', category=FutureWarning)
            shap.summary_plot(sv, X_sample, show=False, max_display=min(top_n, 25), plot_size=None)
        ax.set_title(f'SHAP beeswarm — {task}', fontsize=9)
        return True

    except Exception:
        logger.debug('SHAP plot failed for %s (%s)', model_name, task, exc_info=True)
        return False


# ── bar chart ─────────────────────────────────────────────────────────────────

def _draw_bar(
    ax: plt.Axes,
    importance: pd.Series,
    model_name: str,
    task: str,
    top_n: int,
    total_features: int,
    imp_type: str,
) -> None:
    top = importance.head(top_n)
    n = len(top)
    # Цветовой градиент: самые важные — тёмно-синие, менее важные — светлее
    palette = plt.cm.Blues(np.linspace(0.38, 0.92, n))[::-1]

    bars = ax.barh(range(n), top.values, color=palette, edgecolor='none', height=0.72)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(imp_type, fontsize=9)
    ax.set_title(
        f'{model_name.upper()} · {task}\n{imp_type}  (top {n} / {total_features})',
        fontsize=9, pad=8,
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    x_max = float(top.values.max()) if n > 0 else 1.0
    for bar, val in zip(bars, top.values):
        label = f'{val:.3f}' if abs(val) < 10 else f'{val:.1f}'
        ax.text(
            min(val + x_max * 0.01, x_max * 1.12),
            bar.get_y() + bar.get_height() / 2,
            label, va='center', fontsize=7,
        )


# ── public API ────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model: Any,
    model_name: str,
    feature_names: list[str],
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    task: str,
    save_path: Path | str,
    predict_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
    top_n: int = _TOP_N,
) -> pd.Series | None:
    """Строит и сохраняет график важности признаков + (для tree-моделей) SHAP beeswarm.

    Левая панель: горизонтальный bar chart топ-``top_n`` признаков.
    Правая панель: SHAP beeswarm (catboost / lightgbm / xgboost); для LAMA и TabM — отсутствует.

    Args:
        model: Обученная модель.
        model_name: Имя адаптера ('catboost', 'lightgbm', 'xgboost', 'lama', 'tabm').
        feature_names: Список признаков, использованных при обучении.
        X_valid: Валидационная выборка (Pandas DataFrame).
        y_valid: Целевая переменная валидационной выборки.
        task: 'regression' или 'classification'.
        save_path: Путь сохранения PNG.
        predict_fn: Функция ``f(X_df) -> np.ndarray``. Обязательна для LAMA/TabM.
        top_n: Число признаков в bar chart.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── compute importance ────────────────────────────────────────────────────
    try:
        from ml_toolkit.models import LINEAR_NAMES  # noqa: PLC0415
        if model_name in _TREE_MODELS:
            raw = _tree_importance(model, model_name, feature_names)
            if raw is not None:
                imp_type = 'Feature Importance (gain)'
            elif predict_fn is not None:
                # fallback: feature_importances_ недоступен (напр. HistGBM < sklearn 1.4)
                raw = _permutation_importance(predict_fn, feature_names, X_valid, y_valid, task)
                imp_type = 'Permutation Importance (ΔMAE)' if task == 'regression' else 'Permutation Importance (ΔPR-AUC)'
            else:
                logger.warning('no feature_importances_ and predict_fn is None for %s; skipping', model_name)
                return
        elif model_name in LINEAR_NAMES:
            raw = _linear_importance(model, feature_names)
            imp_type = '|Coefficient| (StandardScaler units)'
        else:
            if predict_fn is None:
                logger.warning('predict_fn is None for %s; skipping feature importance', model_name)
                return
            raw = _permutation_importance(predict_fn, feature_names, X_valid, y_valid, task)
            imp_type = 'Permutation Importance (ΔMAE)' if task == 'regression' else 'Permutation Importance (ΔPR-AUC)'
    except Exception:
        logger.exception('Importance computation failed for %s (%s)', model_name, task)
        return

    importance = pd.Series(raw, index=feature_names, name='importance').sort_values(ascending=False)

    # ── layout ────────────────────────────────────────────────────────────────
    has_shap = model_name in (_TREE_MODELS - {'mondrian'})
    n_show = min(top_n, len(importance))
    fig_h = max(6.0, n_show * 0.36 + 2.5)

    if has_shap:
        fig, axes = plt.subplots(1, 2, figsize=(22, fig_h))
        ax_bar, ax_shap = axes
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=(11, fig_h))
        ax_shap = None

    _draw_bar(ax_bar, importance, model_name, task, top_n, len(feature_names), imp_type)

    if ax_shap is not None:
        shap_ok = _try_shap_plot(ax_shap, model, model_name, feature_names, X_valid, task, top_n)
        if not shap_ok:
            ax_shap.set_visible(False)

    fig.suptitle(
        f'Feature Importance & Interpretation — {model_name.upper()} [{task.capitalize()}]',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('feature_importance_%s.png → %s', task, save_path)
    return importance
