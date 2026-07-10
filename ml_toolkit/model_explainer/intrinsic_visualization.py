"""Нативная интерпретация интерпретируемых моделей — визуализации.

Поддерживаемые адаптеры и тип визуализации:
    decision_tree      — plot_tree: структура дерева (sklearn)
    linear_tree/m5_tree — feature importances + текстовое дерево
    ebm                — shape functions (EBM.explain_global)
    pygam              — partial dependence по каждому терму
    mars               — вклады базисных функций (summary)
    rulefit            — топ-правил по важности (bar chart)
    figs               — структура tree-компонентов
    skope_rules        — правила с precision/recall
    brl                — decision list (текст)
    ripper             — правила (текст)
    nam/gaminet        — shape functions feature networks (PyTorch)

Точка входа: ``plot_interpretable_extra()``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_toolkit.models import (
    GAM_NAMES as _GAM_SET,
)
from ml_toolkit.models import (
    IMODELS_NAMES as _RULE_TEXT_SET,
)
from ml_toolkit.models import (
    INTERPRETABLE_NEURAL_NAMES as _NAM_SET,
)
from ml_toolkit.models import (
    INTERPRETABLE_TREE_NAMES as _INTERPRETABLE_TREE_SET,
)
from ml_toolkit.models import (
    LINEAR_TREE_NAMES as _LINEAR_TREE_SET,
)

logger = logging.getLogger(__name__)

_DECISION_TREE_SET: frozenset[str] = frozenset({'decision_tree'})

ALL_INTERPRETABLE: frozenset[str] = (
    _DECISION_TREE_SET | _LINEAR_TREE_SET | _GAM_SET
    | frozenset({'rulefit'}) | _RULE_TEXT_SET | _NAM_SET | _INTERPRETABLE_TREE_SET
)


# ── Decision Tree ─────────────────────────────────────────────────────────────

def _plot_decision_tree(
    model: Any,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """Структура дерева: sklearn plot_tree с ограничением глубины для читаемости."""
    try:
        from sklearn.tree import plot_tree

        dt = model.named_steps.get('estimator', model[-1])
        depth = dt.get_depth()
        n_leaves = dt.get_n_leaves()
        max_show = min(4, depth)

        fig, ax = plt.subplots(figsize=(max(16, max_show * 5), max(6, max_show * 2.5)))
        plot_tree(
            dt,
            feature_names=feature_names,
            filled=True,
            max_depth=max_show,
            fontsize=7,
            ax=ax,
            impurity=False,
            rounded=True,
            precision=3,
        )
        ax.set_title(
            f'Decision Tree — {task}  '
            f'(depth={depth}, leaves={n_leaves}, showing ≤{max_show} levels)',
            fontsize=10, pad=10,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('decision_tree_structure_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('Decision tree plot failed (%s)', task, exc_info=True)
        return False


# ── Linear Tree / M5 ─────────────────────────────────────────────────────────

def _plot_linear_tree(
    model_tuple: tuple,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """Feature importances из LinearTree + сводка по листьям (текст)."""
    try:
        ltr_model, _imputer, _scaler, num_feats = model_tuple

        # Feature importances
        fi: np.ndarray | None = None
        if hasattr(ltr_model, 'feature_importances_'):
            raw_fi = np.array(ltr_model.feature_importances_)
            nf_idx = {f: i for i, f in enumerate(num_feats)}
            fi = np.array([raw_fi[nf_idx[f]] if f in nf_idx else 0.0 for f in feature_names])

        # Текстовое резюме через summary() если доступно
        summary_text: str = ''
        if hasattr(ltr_model, 'summary'):
            try:
                buf = io.StringIO()
                import contextlib
                with contextlib.redirect_stdout(buf):
                    ltr_model.summary(feature_names=feature_names[:len(num_feats)])
                summary_text = buf.getvalue()[:3000]
            except Exception:
                summary_text = f'{type(ltr_model).__name__}: summary() unavailable'

        if fi is None and not summary_text:
            return False

        has_bar = fi is not None
        fig_h = max(5, len(feature_names) * 0.3 + 2)
        if has_bar and summary_text:
            fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(18, fig_h))
        elif has_bar:
            fig, ax_bar = plt.subplots(1, 1, figsize=(10, fig_h))
            ax_txt = None
        else:
            fig, ax_txt = plt.subplots(1, 1, figsize=(12, fig_h))
            ax_bar = None

        if ax_bar is not None:
            imp_s = pd.Series(fi, index=feature_names).sort_values(ascending=False).head(30)
            n = len(imp_s)
            palette = plt.cm.Blues(np.linspace(0.38, 0.92, n))[::-1]
            ax_bar.barh(range(n), imp_s.values, color=palette, edgecolor='none', height=0.72)
            ax_bar.set_yticks(range(n))
            ax_bar.set_yticklabels(imp_s.index, fontsize=8)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel('Feature Importance', fontsize=9)
            ax_bar.set_title(f'{type(ltr_model).__name__} · {task}', fontsize=9)
            ax_bar.spines[['top', 'right']].set_visible(False)
            ax_bar.grid(axis='x', alpha=0.25, linestyle='--')

        if ax_txt is not None and summary_text:
            ax_txt.text(
                0.02, 0.98, summary_text,
                transform=ax_txt.transAxes, fontsize=7,
                verticalalignment='top', family='monospace',
            )
            ax_txt.axis('off')
            ax_txt.set_title('Tree Summary', fontsize=9)

        fig.suptitle(
            f'Linear Tree Interpretation — {type(ltr_model).__name__} [{task.capitalize()}]',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('linear_tree_interpretation_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('Linear tree plot failed (%s)', task, exc_info=True)
        return False


# ── EBM shape functions ───────────────────────────────────────────────────────

def _plot_ebm_shapes(
    model: Any,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """Shape functions EBM (interpret): одна панель на term, топ-12 по важности."""
    try:
        explanation = model.explain_global()
        term_names: list[str] = list(model.term_names_)
        term_imps = np.array(model.term_importances())

        # Сортируем по важности, берём топ-12
        order = np.argsort(term_imps)[::-1][:12]
        n_terms = len(order)
        n_cols = 3
        n_rows = int(np.ceil(n_terms / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
        axes_flat = np.array(axes).flatten() if n_terms > 1 else [axes]

        for plot_i, term_i in enumerate(order):
            ax = axes_flat[plot_i]
            term_data = explanation.data(int(term_i))
            names = term_data.get('names', [])
            scores = term_data.get('scores', [])
            names_arr = np.array(names) if names else np.array([])
            scores_arr = np.array(scores) if scores else np.array([])

            if len(names_arr) > 1 and len(scores_arr) > 0:
                x_vals = names_arr[:-1] if len(names_arr) > len(scores_arr) else names_arr
                x_num = np.arange(len(scores_arr))
                try:
                    x_num = x_vals.astype(float)
                    ax.plot(x_num, scores_arr[:len(x_num)], lw=2, color='steelblue')
                except (ValueError, TypeError):
                    ax.bar(range(len(scores_arr)), scores_arr, color='steelblue')
                ax.axhline(0, color='gray', lw=0.5)

            ax.set_title(
                f'{term_names[term_i]}  (imp={term_imps[term_i]:.4f})',
                fontsize=7.5,
            )
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        for ax in axes_flat[n_terms:]:
            ax.set_visible(False)

        fig.suptitle(f'EBM Shape Functions — {task} (top {n_terms} terms)', fontsize=11)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('ebm_shapes_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('EBM shape plot failed (%s)', task, exc_info=True)
        return False


# ── pyGAM partial dependence ──────────────────────────────────────────────────

def _plot_gam_shapes(
    model_tuple: tuple,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """pyGAM: partial dependence с 95% доверительным интервалом для топ-12 термов."""
    try:
        gam_model, _prep, num_feats = model_tuple
        n_terms = min(12, len(num_feats))
        n_cols = 3
        n_rows = int(np.ceil(n_terms / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
        axes_flat = np.array(axes).flatten() if n_terms > 1 else [axes]

        for i in range(n_terms):
            ax = axes_flat[i]
            try:
                XX = gam_model.generate_X_grid(term=i, n=100)
                pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=0.95)
                x_vals = XX[:, i]
                ax.plot(x_vals, pdep, lw=2, color='steelblue')
                ax.fill_between(x_vals, confi[:, 0], confi[:, 1], alpha=0.2, color='steelblue')
                ax.axhline(0, color='gray', lw=0.5)
            except Exception:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

            ax.set_title(num_feats[i], fontsize=8)
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        for ax in axes_flat[n_terms:]:
            ax.set_visible(False)

        fig.suptitle(f'pyGAM Partial Dependence — {task} (95% CI)', fontsize=11)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('pygam_shapes_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('pyGAM shape plot failed (%s)', task, exc_info=True)
        return False


# ── MARS basis contributions ──────────────────────────────────────────────────

def _plot_mars_summary(
    model_tuple: tuple,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """MARS: вклады признаков (feature_importances_) + текстовый summary."""
    try:
        if len(model_tuple) == 4:
            # classification: (earth, imputer, clf, num_feats)
            earth_model, _imputer, _clf, num_feats = model_tuple
        else:
            earth_model, _imputer, num_feats = model_tuple

        # Feature importance через внутренние атрибуты MARS
        fi_raw: np.ndarray | None = None
        if hasattr(earth_model, 'feature_importances_'):
            raw = earth_model.feature_importances_
            if isinstance(raw, dict):
                nf_idx = {f: i for i, f in enumerate(num_feats)}
                fi_arr = np.array([raw.get(f, 0.0) for f in num_feats])
            else:
                fi_arr = np.array(raw)
            nf_idx = {f: i for i, f in enumerate(num_feats)}
            fi_raw = np.array([
                fi_arr[nf_idx[f]] if f in nf_idx and nf_idx[f] < len(fi_arr) else 0.0
                for f in feature_names
            ])

        # Текстовый summary
        summary_text = ''
        if hasattr(earth_model, 'summary'):
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                earth_model.summary()
            summary_text = buf.getvalue()[:3000]

        if fi_raw is None and not summary_text:
            return False

        has_bar = fi_raw is not None and np.any(fi_raw > 0)
        fig_h = max(5, len(feature_names) * 0.28 + 2)
        if has_bar and summary_text:
            fig, (ax_bar, ax_txt) = plt.subplots(1, 2, figsize=(18, fig_h))
        elif has_bar:
            fig, ax_bar = plt.subplots(1, 1, figsize=(10, fig_h))
            ax_txt = None
        else:
            fig, ax_txt = plt.subplots(1, 1, figsize=(12, fig_h))
            ax_bar = None

        if ax_bar is not None:
            imp_s = pd.Series(fi_raw, index=feature_names).sort_values(ascending=False).head(30)
            n = len(imp_s)
            palette = plt.cm.Oranges(np.linspace(0.38, 0.92, n))[::-1]
            ax_bar.barh(range(n), imp_s.values, color=palette, edgecolor='none', height=0.72)
            ax_bar.set_yticks(range(n))
            ax_bar.set_yticklabels(imp_s.index, fontsize=8)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel('Feature Importance (MARS)', fontsize=9)
            ax_bar.set_title(f'MARS · {task}', fontsize=9)
            ax_bar.spines[['top', 'right']].set_visible(False)
            ax_bar.grid(axis='x', alpha=0.25, linestyle='--')

        if ax_txt is not None and summary_text:
            ax_txt.text(
                0.01, 0.99, summary_text,
                transform=ax_txt.transAxes, fontsize=6.5,
                verticalalignment='top', family='monospace',
            )
            ax_txt.axis('off')
            ax_txt.set_title('MARS Basis Summary', fontsize=9)

        fig.suptitle(f'MARS Interpretation — {task.capitalize()}', fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('mars_interpretation_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('MARS plot failed (%s)', task, exc_info=True)
        return False


# ── RuleFit rules ─────────────────────────────────────────────────────────────

def _plot_rulefit_rules(
    model_tuple: tuple,
    save_path: Path,
    task: str,
) -> bool:
    """RuleFit: горизонтальный bar chart топ-20 правил по важности; цвет = знак coef."""
    try:
        rulefit_model, _prep, _num_feats = model_tuple
        rules = rulefit_model.get_rules()
        rules = rules[rules['coef'] != 0].sort_values('importance', ascending=False).head(20)

        if rules.empty:
            return False

        fig_h = max(4.0, len(rules) * 0.38 + 2.0)
        fig, ax = plt.subplots(figsize=(15, fig_h))
        colors = ['#d73027' if c > 0 else '#4575b4' for c in rules['coef']]
        bars = ax.barh(range(len(rules)), rules['importance'].values, color=colors, edgecolor='none', height=0.7)

        labels = [
            f'{row["rule"][:80]}  [coef={row["coef"]:.3f}, supp={row.get("support", "?")}]'
            for _, row in rules.iterrows()
        ]
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=9)
        ax.set_title(
            f'RuleFit Top Rules — {task}  '
            f'(red = positive coef, blue = negative coef)',
            fontsize=10,
        )
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

        x_max = float(rules['importance'].max()) if len(rules) > 0 else 1.0
        for bar, val in zip(bars, rules['importance'].values):
            ax.text(
                min(val + x_max * 0.01, x_max * 1.08),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=6.5,
            )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('rulefit_rules_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('RuleFit rules plot failed (%s)', task, exc_info=True)
        return False


# ── Rule-text models (FIGS, SKOPE, BRL, RIPPER) ───────────────────────────────

def _collect_rules_text(model: Any, model_name: str) -> str:
    """Собирает текстовое представление правил модели."""
    lines: list[str] = []

    if model_name == 'figs':
        # FIGSRegressor/Classifier: захват stdout из print_tree()
        if hasattr(model, 'print_tree'):
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                try:
                    model.print_tree()
                except Exception:
                    pass
            out = buf.getvalue()
            if out.strip():
                return out[:4000]
        if hasattr(model, 'trees_'):
            lines.append(f'FIGS: {len(model.trees_)} tree components')
            for i, tree in enumerate(model.trees_[:5]):
                lines.append(f'  Tree {i}: {tree}')

    elif model_name == 'skope_rules':
        if hasattr(model, 'rules_') and model.rules_:
            lines.append(f'SKOPE-Rules: {len(model.rules_)} rules')
            for r in model.rules_[:15]:
                rule_str = getattr(r, 'args', None)
                if rule_str:
                    lines.append(f'  IF {rule_str[0]}  → P={r.args[1]:.3f}  R={r.args[2]:.3f}')
                else:
                    lines.append(f'  {r}')

    elif model_name == 'brl':
        # BRL: захват stdout из print_list()
        if hasattr(model, 'print_list'):
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                try:
                    model.print_list()
                except Exception:
                    pass
            out = buf.getvalue()
            if out.strip():
                return out[:4000]
        if hasattr(model, 'rules_'):
            lines.append(f'BRL rules: {model.rules_}')

    elif model_name == 'ripper':
        if hasattr(model, 'rules_'):
            lines.append(f'RIPPER: {len(model.rules_)} rules')
            for r in model.rules_[:20]:
                lines.append(f'  {r}')
        elif hasattr(model, 'ruleset_') and model.ruleset_:
            lines.append(str(model.ruleset_)[:4000])

    return '\n'.join(lines) if lines else f'({model_name}: no rule text available)'


def _plot_rules_text(
    model_tuple: tuple,
    model_name: str,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """Текстовое представление правил для FIGS/SKOPE/BRL/RIPPER."""
    try:
        model, _prep, _num_feats = model_tuple
        text = _collect_rules_text(model, model_name)

        n_lines = text.count('\n') + 1
        fig_h = max(4.0, min(20.0, n_lines * 0.22 + 1.5))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.text(
            0.01, 0.99, text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top', family='monospace',
        )
        ax.axis('off')
        ax.set_title(
            f'{model_name.upper()} Rules — {task}  '
            f'({type(model).__name__})',
            fontsize=10, pad=8,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('%s_rules_%s.png → %s', model_name, task, save_path)
        return True
    except Exception:
        logger.debug('%s rules plot failed (%s)', model_name, task, exc_info=True)
        return False


# ── NAM / GAMINET shape functions ─────────────────────────────────────────────

def _plot_nam_shapes(
    model_tuple: tuple,
    model_name: str,
    feature_names: list[str],
    X_valid: pd.DataFrame,
    save_path: Path,
    task: str,
) -> bool:
    """NAM/GAMINET: shape functions для топ-12 признаков (каждый feature_net отдельно)."""
    try:
        import torch

        net, imputer, qt, num_feats = model_tuple

        # Для классификации model_tuple содержит (LogisticRegression, imputer, qt, num_feats)
        if not hasattr(net, 'feature_nets'):
            # Это LogisticRegression (classification fallback) — используем coef_
            return _plot_logistic_coef(net, imputer, qt, num_feats, feature_names, save_path, task)

        # Квантильные преобразованные данные для масштаба
        X_np = qt.transform(imputer.transform(X_valid[num_feats].to_numpy(dtype=float)))

        n_terms = min(12, len(num_feats))
        n_cols = 3
        n_rows = int(np.ceil(n_terms / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
        axes_flat = np.array(axes).flatten() if n_terms > 1 else [axes]

        net.eval()
        for i in range(n_terms):
            ax = axes_flat[i]
            x_min, x_max = float(X_np[:, i].min()), float(X_np[:, i].max())
            x_grid = np.linspace(x_min, x_max, 100)

            fnet = net.feature_nets[i]
            with torch.no_grad():
                x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32)
                y_grid = fnet(x_t).squeeze(-1).numpy()

            ax.plot(x_grid, y_grid, lw=2, color='darkorange')
            ax.axhline(0, color='gray', lw=0.5)
            ax.set_title(num_feats[i], fontsize=8)
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        # GAMINET: добавить 2D interaction heatmap для первой пары
        if model_name == 'gaminet' and hasattr(net, 'pair_nets') and net.pair_nets:
            pair_i, pair_j = net.pairs[0]
            ax = axes_flat[min(n_terms, len(axes_flat) - 1)]
            ax.set_visible(True)
            xi = np.linspace(float(X_np[:, pair_i].min()), float(X_np[:, pair_i].max()), 30)
            xj = np.linspace(float(X_np[:, pair_j].min()), float(X_np[:, pair_j].max()), 30)
            grid_i, grid_j = np.meshgrid(xi, xj)
            X_pair = np.stack([grid_i.ravel(), grid_j.ravel()], axis=1)
            with torch.no_grad():
                pair_net = net.pair_nets[0]
                z = pair_net(torch.tensor(X_pair, dtype=torch.float32)).squeeze(-1).numpy()
            ax.contourf(xi, xj, z.reshape(30, 30), levels=20, cmap='RdBu_r')
            ax.set_title(f'Interaction: {num_feats[pair_i]} × {num_feats[pair_j]}', fontsize=8)

        for ax in axes_flat[n_terms + (1 if model_name == 'gaminet' else 0):]:
            ax.set_visible(False)

        fig.suptitle(
            f'{model_name.upper()} Shape Functions — {task} (top {n_terms} features)',
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('%s_shapes_%s.png → %s', model_name, task, save_path)
        return True
    except Exception:
        logger.warning('%s shape plot failed (%s)', model_name, task, exc_info=True)
        return False


def _plot_logistic_coef(
    clf: Any,
    imputer: Any,
    qt: Any,
    num_feats: list[str],
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """Коэффициенты LogisticRegression как surrogate importance для nam/gaminet classification."""
    try:
        coef = np.abs(clf.coef_).flatten()
        nf_idx = {f: i for i, f in enumerate(num_feats)}
        raw = np.array([coef[nf_idx[f]] if f in nf_idx and nf_idx[f] < len(coef) else 0.0
                        for f in feature_names])
        imp_s = pd.Series(raw, index=feature_names).sort_values(ascending=False).head(30)

        fig_h = max(5, len(imp_s) * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        n = len(imp_s)
        palette = plt.cm.Purples(np.linspace(0.38, 0.92, n))[::-1]
        ax.barh(range(n), imp_s.values, color=palette, edgecolor='none', height=0.72)
        ax.set_yticks(range(n))
        ax.set_yticklabels(imp_s.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('|LogisticRegression coef|', fontsize=9)
        ax.set_title(f'NAM/GAMINET Cls — {task} (LogisticRegression coef)', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        return False


# ── Soft Decision Tree ───────────────────────────────────────────────────────

def _plot_soft_decision_tree(
    model_tuple: tuple,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """SoftDecisionTree: важность признаков из весов inner_w + распределение значений листьев."""
    try:

        model, _imputer, _scaler, num_feats = model_tuple
        if model._net is None:
            return False

        net = model._net
        depth = model.depth
        n_leaves = 2 ** depth

        # Важность признаков: сумма |весов| по всем внутренним узлам
        inner_w = net.inner_w.weight.detach().cpu()  # (n_inner, n_features)
        fi_num = inner_w.abs().sum(dim=0).numpy()
        nf_idx = {f: i for i, f in enumerate(num_feats)}
        fi_all = np.array([fi_num[nf_idx[f]] if f in nf_idx and nf_idx[f] < len(fi_num) else 0.0
                           for f in feature_names])
        imp_s = pd.Series(fi_all, index=feature_names).sort_values(ascending=False).head(30)

        # Значения листьев
        leaf_vals = net.leaf_vals.detach().cpu().squeeze(-1).numpy()

        has_bar = len(imp_s) > 0
        fig_h = max(6.0, len(imp_s) * 0.35 + 2.5)
        fig, (ax_bar, ax_leaf) = plt.subplots(1, 2, figsize=(18, fig_h))

        if has_bar:
            n = len(imp_s)
            palette = plt.cm.Greens(np.linspace(0.38, 0.92, n))[::-1]
            ax_bar.barh(range(n), imp_s.values, color=palette, edgecolor='none', height=0.72)
            ax_bar.set_yticks(range(n))
            ax_bar.set_yticklabels(imp_s.index, fontsize=8)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel('Sum |inner_w| (feature routing importance)', fontsize=9)
            ax_bar.set_title(f'SoftDecisionTree · {task}\n(depth={depth}, leaves={n_leaves})', fontsize=9)
            ax_bar.spines[['top', 'right']].set_visible(False)
            ax_bar.grid(axis='x', alpha=0.25, linestyle='--')

        colors = ['#4575b4' if v < 0 else '#d73027' for v in leaf_vals]
        ax_leaf.bar(range(n_leaves), leaf_vals, color=colors, edgecolor='none')
        ax_leaf.axhline(0, color='gray', lw=0.8)
        ax_leaf.set_xlabel('Leaf index', fontsize=9)
        ax_leaf.set_ylabel('Leaf value', fontsize=9)
        ax_leaf.set_title('Leaf values  (blue=negative, red=positive)', fontsize=9)
        ax_leaf.spines[['top', 'right']].set_visible(False)
        ax_leaf.grid(axis='y', alpha=0.25, linestyle='--')

        fig.suptitle(
            f'SoftDecisionTree Interpretation — {task.capitalize()}  '
            f'(depth={depth}, leaves={n_leaves})',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('soft_decision_tree_interpretation_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('SoftDecisionTree plot failed (%s)', task, exc_info=True)
        return False


# ── Locally Linear Forest ─────────────────────────────────────────────────────

def _plot_locally_linear_forest(
    model_tuple: tuple,
    feature_names: list[str],
    save_path: Path,
    task: str,
) -> bool:
    """LocallyLinearForest: важность признаков из встроенного RandomForest."""
    try:
        model, _imputer, _scaler, num_feats = model_tuple
        if not (hasattr(model, 'rf') and hasattr(model.rf, 'feature_importances_')):
            return False

        fi = model.rf.feature_importances_
        nf_idx = {f: i for i, f in enumerate(num_feats)}
        fi_all = np.array([fi[nf_idx[f]] if f in nf_idx and nf_idx[f] < len(fi) else 0.0
                           for f in feature_names])
        imp_s = pd.Series(fi_all, index=feature_names).sort_values(ascending=False).head(30)

        n = len(imp_s)
        fig_h = max(6.0, n * 0.36 + 2.5)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        palette = plt.cm.Blues(np.linspace(0.38, 0.92, n))[::-1]
        ax.barh(range(n), imp_s.values, color=palette, edgecolor='none', height=0.72)
        ax.set_yticks(range(n))
        ax.set_yticklabels(imp_s.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('RF Feature Importance (MDI)', fontsize=9)
        ax.set_title(
            f'LocallyLinearForest · {task}\n'
            f'RF feature importances (top {n} / {len(feature_names)})',
            fontsize=9, pad=8,
        )
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

        x_max = float(imp_s.values.max()) if n > 0 else 1.0
        for bar, val in zip(ax.patches, imp_s.values):
            ax.text(
                min(val + x_max * 0.01, x_max * 1.12),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7,
            )

        fig.suptitle(
            f'LocallyLinearForest Interpretation — {task.capitalize()}',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info('locally_linear_forest_interpretation_%s.png → %s', task, save_path)
        return True
    except Exception:
        logger.debug('LocallyLinearForest plot failed (%s)', task, exc_info=True)
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def plot_interpretable_extra(
    model: Any,
    model_name: str,
    feature_names: list[str],
    X_valid: pd.DataFrame,
    save_path: Path | str,
    task: str = 'regression',
) -> bool:
    """Нативная интерпретационная визуализация для интерпретируемых моделей.

    Диспетчер — выбирает нужную функцию по `model_name`. Для неизвестных
    моделей молча возвращает False (пропускает).

    Args:
        model: Обученная модель (структура зависит от адаптера).
        model_name: Имя адаптера (`'catboost'`, `'lightgbm'`, `'ebm'`, ...).
        feature_names: Список признаков, использованных при обучении.
        X_valid: Валидационная выборка (Pandas DataFrame).
        save_path: Путь сохранения PNG.
        task: 'regression' или 'classification'.

    Returns:
        True если визуализация успешно сохранена.

    """
    if model_name not in ALL_INTERPRETABLE:
        return False

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if model_name == 'decision_tree':
        return _plot_decision_tree(model, feature_names, save_path, task)
    if model_name in _LINEAR_TREE_SET:
        return _plot_linear_tree(model, feature_names, save_path, task)
    if model_name == 'ebm':
        return _plot_ebm_shapes(model, feature_names, save_path, task)
    if model_name == 'pygam':
        return _plot_gam_shapes(model, feature_names, save_path, task)
    if model_name == 'mars':
        return _plot_mars_summary(model, feature_names, save_path, task)
    if model_name == 'rulefit':
        return _plot_rulefit_rules(model, save_path, task)
    if model_name in _RULE_TEXT_SET:
        return _plot_rules_text(model, model_name, feature_names, save_path, task)
    if model_name in _NAM_SET:
        return _plot_nam_shapes(model, model_name, feature_names, X_valid, save_path, task)
    if model_name == 'soft_decision_tree':
        return _plot_soft_decision_tree(model, feature_names, save_path, task)
    if model_name == 'locally_linear_forest':
        return _plot_locally_linear_forest(model, feature_names, save_path, task)
    return False
