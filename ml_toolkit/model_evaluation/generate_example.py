"""Генератор HTML-примера для ml_toolkit.model_evaluation.

Запуск:
    uv run python ml_toolkit/model_evaluation/generate_example.py

Создаёт example.html рядом со скриптом — самодостаточный HTML без доступа к сети.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Добавляем корень проекта в sys.path, чтобы работал `from ml_toolkit.model_evaluation import ...`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_toolkit.model_evaluation import (
    ClassificationEvaluator,
    RegressionEvaluator,
    compare_models,
    f1_at_threshold,
    lift_at_k,
    plot_model_comparison,
    plot_model_delta,
    plot_model_heatmap,
    precision_at_k,
    recall_at_k,
)

# ── Воспроизводимые синтетические данные ───────────────────────────────────────

rng = np.random.default_rng(42)

def _make_cls_data(n: int, base_rate: float = 0.15, skill: float = 0.75):
    y = (rng.random(n) < base_rate).astype(int)
    noise = rng.random(n)
    p = np.where(y == 1, skill + (1 - skill) * noise, (1 - skill) * noise)
    return y, np.clip(p, 0.01, 0.99)

y_train, p_train = _make_cls_data(3000, skill=0.80)
y_valid, p_valid = _make_cls_data(1000, skill=0.74)
y_test,  p_test  = _make_cls_data(1000, skill=0.68)

def _make_reg_data(n: int, noise: float = 0.25):
    x = rng.uniform(0, 10, n)
    y_true = 3 * np.sin(x) + x * 0.5 + rng.normal(0, noise, n)
    y_pred = y_true + rng.normal(0, noise * 2, n)
    return y_true, y_pred

yt_train, yp_train = _make_reg_data(3000)
yt_valid, yp_valid = _make_reg_data(1000)
yt_test,  yp_test  = _make_reg_data(1000, noise=0.45)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:12px 0">'


def _code(src: str) -> str:
    src = textwrap.dedent(src).strip()
    return f'<pre class="code"><code>{_escape(src)}</code></pre>'


def _escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _h2(text: str) -> str:
    return f'<h2>{text}</h2>'


def _h3(text: str) -> str:
    return f'<h3>{text}</h3>'


def _p(text: str) -> str:
    return f'<p>{text}</p>'


def _table_html(df) -> str:
    return '<div class="table-wrap">' + df.to_html(float_format='%.4f', classes='t') + '</div>'


# ── Sections ───────────────────────────────────────────────────────────────────

sections: list[str] = []

# ── 0. Вводный раздел ─────────────────────────────────────────────────────────

sections.append(
    _h2('О модуле') +
    _p(
        '<code>ml_toolkit.model_evaluation</code> — пакет для оценки качества ML-моделей. '
        'Поддерживает бинарную классификацию, многоклассовую классификацию и регрессию. '
        'Единый API для метрик, анализа порогов, бизнес-кривых, bootstrap и HTML-отчётов.'
    ) +
    _code('''
        from ml_toolkit.model_evaluation import (
            ClassificationEvaluator,
            RegressionEvaluator,
            precision_at_k, recall_at_k, lift_at_k, f1_at_threshold,
            compare_models, plot_model_comparison, plot_model_heatmap, plot_model_delta,
        )
        # или реэкспорт из ml_toolkit.models:
        from ml_toolkit.models import ClassificationEvaluator, RegressionEvaluator
    ''')
)

# ── 1. ClassificationEvaluator — базовый сценарий ─────────────────────────────

ev_cls = ClassificationEvaluator(task='binary')
ev_cls.add('train', y_train, p_train)
ev_cls.add('valid', y_valid, p_valid)
ev_cls.add('test',  y_test,  p_test)

ev_cls.add_default_metrics()
ev_cls.add_metric(precision_at_k(0.05), name='precision@5%')
ev_cls.add_metric(precision_at_k(0.10), name='precision@10%')
ev_cls.add_metric(precision_at_k(0.20), name='precision@20%')
ev_cls.add_metric(recall_at_k(0.05),    name='recall@5%')
ev_cls.add_metric(lift_at_k(0.10),      name='lift@10%')
ev_cls.add_metric(f1_at_threshold(0.3), name='f1@t=0.3')
ev_cls.add_metric(f1_at_threshold(0.5), name='f1@t=0.5')

sections.append(
    _h2('1. ClassificationEvaluator — регистрация данных и метрик') +
    _code('''
        ev = ClassificationEvaluator(task='binary')   # или task='multiclass'
        ev.add('train', y_true_train, y_proba_train)
        ev.add('valid', y_true_valid, y_proba_valid)
        ev.add('test',  y_true_test,  y_proba_test)

        # Набор метрик по умолчанию: roc_auc, pr_auc, log_loss, brier, ks, gini, mcc, ece
        ev.add_default_metrics()

        # Параметризованные фабрики
        ev.add_metric(precision_at_k(0.05), name='precision@5%')
        ev.add_metric(precision_at_k(0.10), name='precision@10%')
        ev.add_metric(precision_at_k(0.20), name='precision@20%')
        ev.add_metric(recall_at_k(0.05),    name='recall@5%')
        ev.add_metric(lift_at_k(0.10),      name='lift@10%')
        ev.add_metric(f1_at_threshold(0.3), name='f1@t=0.3')
        ev.add_metric(f1_at_threshold(0.5), name='f1@t=0.5')

        # Все метрики по всем сплитам
        df = ev.metrics()
    ''') +
    _h3('Таблица метрик') +
    _table_html(ev_cls.metrics())
)

# ── 2. compare_splits ─────────────────────────────────────────────────────────

sections.append(
    _h2('2. Сравнение сплитов (детектор переобучения)') +
    _code('''
        # metric | train | valid | delta | ratio
        df = ev.compare_splits(ref='train', target='valid')

        # valid vs test
        df = ev.compare_splits(ref='valid', target='test')
    ''') +
    _h3('train vs valid') +
    _table_html(ev_cls.compare_splits('train', 'valid')) +
    _h3('valid vs test') +
    _table_html(ev_cls.compare_splits('valid', 'test'))
)

# ── 3. ROC + PR кривые ────────────────────────────────────────────────────────

fig, axes_pair = plt.subplots(1, 2, figsize=(13, 5))
ev_cls.plot_roc(splits=['train', 'valid', 'test'], ax=axes_pair[0])
ev_cls.plot_pr(splits=['train', 'valid', 'test'], ax=axes_pair[1])
plt.tight_layout()
b64_roc_pr = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('3. ROC и Precision-Recall кривые') +
    _code('''
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ev.plot_roc(splits=['train', 'valid', 'test'], ax=ax1)
        ev.plot_pr(splits=['train', 'valid', 'test'],  ax=ax2)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_roc_pr)
)

# ── 4. Распределение скоров ────────────────────────────────────────────────────

fig, axes_sd = plt.subplots(1, 3, figsize=(14, 4))
ev_cls.plot_score_distribution(splits=['train', 'valid', 'test'], axes=axes_sd)
plt.tight_layout()
b64_sd = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('4. Распределение скоров по классам') +
    _code('''
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ev.plot_score_distribution(splits=['train', 'valid', 'test'], axes=axes)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_sd)
)

# ── 5. Бизнес-кривые (lift, gains, decile) ────────────────────────────────────

fig, axes_biz = plt.subplots(1, 3, figsize=(16, 5))
ev_cls.plot_lift(splits=['valid', 'test'], ax=axes_biz[0])
ev_cls.plot_gains(splits=['valid', 'test'], ax=axes_biz[1])
ev_cls.plot_decile_bar(split='test', ax=axes_biz[2])
plt.tight_layout()
b64_biz = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('5. Бизнес-кривые: lift, cumulative gains, decile bar') +
    _code('''
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        ev.plot_lift(splits=['valid', 'test'], ax=ax1)
        ev.plot_gains(splits=['valid', 'test'], ax=ax2)
        ev.plot_decile_bar(split='test', ax=ax3)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_biz)
)

# ── 6. Precision & Recall at k ────────────────────────────────────────────────

fig, ax_prk = plt.subplots(figsize=(10, 6))
ev_cls.plot_precision_recall_at_k(
    splits=['valid', 'test'],
    k_frac=[0.05, 0.10, 0.20],
    min_precision=0.35,
    show_f1=True,
    show_counts_axis=True,
    ax=ax_prk,
)
plt.tight_layout()
b64_prk = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('6. Precision & Recall at k') +
    _p(
        'Цвет кодирует метрику (синий = precision, красный = recall, зелёный = F1). '
        'Стиль линии кодирует сплит: solid = test, dashed = valid.'
    ) +
    _code('''
        ev.plot_precision_recall_at_k(
            splits=['valid', 'test'],
            k_frac=[0.05, 0.10, 0.20],  # вертикальные маркеры по доле выборки
            min_precision=0.35,          # горизонтальная линия: где precision падает ниже порога
            show_f1=True,
            show_counts_axis=True,       # верхняя ось с абсолютными числами
        )
    ''') +
    _img_tag(b64_prk)
)

# ── 7. Калибровка и матрица ошибок ────────────────────────────────────────────

fig, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
ev_cls.plot_calibration(splits=['valid', 'test'], ax=axes_cm[0])
ev_cls.plot_confusion_matrix(split='valid', ax=axes_cm[1])
ev_cls.plot_confusion_matrix(split='test', normalize='true', ax=axes_cm[2])
plt.tight_layout()
b64_cal_cm = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('7. Калибровка и матрица ошибок') +
    _code('''
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        ev.plot_calibration(splits=['valid', 'test'], ax=ax1)
        ev.plot_confusion_matrix(split='valid', ax=ax2)
        ev.plot_confusion_matrix(split='test', normalize='true', ax=ax3)  # recall по строкам
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_cal_cm)
)

# ── 8. Анализ порога ──────────────────────────────────────────────────────────

best = ev_cls.best_threshold(metric='f1', split='valid')
scan_df = ev_cls.threshold_scan(split='valid')

fig, axes_thr = plt.subplots(1, 2, figsize=(13, 5))
ev_cls.plot_threshold_scan(split='valid', ax=axes_thr[0])
ev_cls.plot_ks(split='valid', ax=axes_thr[1])
plt.tight_layout()
b64_thr = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('8. Анализ порога') +
    _code('''
        # Скан: threshold | precision | recall | f1 | accuracy | specificity
        df = ev.threshold_scan(split='valid', n_points=200)

        # Оптимальный порог по заданной метрике
        result = ev.best_threshold(metric='f1', split='valid')
        # → {'threshold': ..., 'f1': ..., 'precision': ..., 'recall': ...}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ev.plot_threshold_scan(split='valid', ax=ax1)
        ev.plot_ks(split='valid', ax=ax2)
        plt.tight_layout()
        plt.show()
    ''') +
    _p(f'Лучший порог по F1 на valid: threshold={best["threshold"]:.3f}, '
       f'f1={best["f1"]:.4f}, precision={best["precision"]:.4f}, recall={best["recall"]:.4f}') +
    _img_tag(b64_thr)
)

# ── 9. PSI ────────────────────────────────────────────────────────────────────

total_psi, psi_df = ev_cls.psi(ref='valid', target='test', n_bins=10)

fig, axes_psi = plt.subplots(1, 2, figsize=(13, 4))
ev_cls.plot_psi(ref='valid', target='test', n_bins=10, axes=axes_psi)
plt.tight_layout()
b64_psi = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('9. Population Stability Index (PSI)') +
    _p('PSI < 0.1 — стабильно, 0.1–0.25 — небольшой сдвиг, > 0.25 — существенный сдвиг.') +
    _code('''
        total_psi, bin_df = ev.psi(ref='valid', target='test', n_bins=10)
        # bin_df: bin | valid_pct | test_pct | psi

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ev.plot_psi(ref='valid', target='test', axes=axes)
        plt.tight_layout()
        plt.show()
    ''') +
    _p(f'Итого PSI (valid → test): <strong>{total_psi:.4f}</strong>') +
    _img_tag(b64_psi)
)

# ── 10. Bootstrap доверительные интервалы ─────────────────────────────────────

boot_df = ev_cls.bootstrap_metrics(split='valid', n_iter=500, ci=0.95, seed=0)

fig, ax_bci = plt.subplots(figsize=(9, 6))
ev_cls.plot_bootstrap_ci(split='valid', n_iter=500, ci=0.95, seed=0, ax=ax_bci)
plt.tight_layout()
b64_bci = _fig_to_b64(fig)
plt.close(fig)

sections.append(
    _h2('10. Bootstrap доверительные интервалы') +
    _code('''
        # DataFrame: строки = метрики, столбцы = [mean, std, ci_low, ci_high]
        df = ev.bootstrap_metrics(split='valid', n_iter=1000, ci=0.95, seed=42)

        # Визуализация: горизонтальные полосы ± CI
        ev.plot_bootstrap_ci(split='valid', n_iter=1000, ci=0.95, seed=42)

        # Гистограммы bootstrap-распределений по каждой метрике
        ev.plot_bootstrap_distributions(split='valid', n_iter=1000, ci=0.95, seed=42)
    ''') +
    _h3('Таблица bootstrap CI (valid, 500 итераций)') +
    _table_html(boot_df) +
    _img_tag(b64_bci)
)

# ── 11. Пользовательские метрики ──────────────────────────────────────────────

ev_custom = ClassificationEvaluator(task='binary')
ev_custom.add('valid', y_valid, p_valid)
ev_custom.add_default_metrics()

def profit_metric(y_true, y_proba, tp_reward=500, fp_cost=100, threshold=0.35):
    pred = (np.asarray(y_proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (np.asarray(y_true) == 1)).sum())
    fp = int(((pred == 1) & (np.asarray(y_true) == 0)).sum())
    return float(tp * tp_reward - fp * fp_cost)

ev_custom.add_metric(profit_metric, name='profit@t=0.35')
ev_custom.add_metric(
    lambda yt, yp: float(np.sum((np.asarray(yp) >= 0.4).astype(int))),
    name='n_selected@t=0.4',
)

sections.append(
    _h2('11. Пользовательские метрики') +
    _p(
        'Любой callable с сигнатурой <code>(y_true: np.ndarray, y_second: np.ndarray) → float</code>.'
    ) +
    _code('''
        def profit_metric(y_true, y_proba):
            pred = (np.asarray(y_proba) >= 0.35).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            return float(tp * 500 - fp * 100)

        ev.add_metric(profit_metric, name='profit@t=0.35')
        ev.add_metric(
            lambda yt, yp: float(np.sum((yp >= 0.4).astype(int))),
            name='n_selected@t=0.4',
        )
    ''') +
    _table_html(ev_custom.metrics())
)

# ── 12. RegressionEvaluator ───────────────────────────────────────────────────

ev_reg = RegressionEvaluator()
ev_reg.add('train', yt_train, yp_train)
ev_reg.add('valid', yt_valid, yp_valid)
ev_reg.add('test',  yt_test,  yp_test)

ev_reg.add_default_metrics()
ev_reg.add_metric('max_error')
ev_reg.add_metric(
    lambda yt, yp: float(np.percentile(np.abs(yt - yp), 90)),
    name='p90_abs_error',
)

fig_avp, axes_avp = plt.subplots(1, 3, figsize=(15, 5))
ev_reg.plot_actual_vs_predicted(splits=['train', 'valid', 'test'], axes=axes_avp)
plt.tight_layout()
b64_avp = _fig_to_b64(fig_avp)
plt.close(fig_avp)

fig_res, axes_res = plt.subplots(1, 3, figsize=(15, 4))
ev_reg.plot_residuals_distribution(splits=['train', 'valid', 'test'], axes=axes_res)
plt.tight_layout()
b64_res = _fig_to_b64(fig_res)
plt.close(fig_res)

fig_rvp, axes_rvp = plt.subplots(1, 3, figsize=(15, 4))
ev_reg.plot_residuals_vs_predicted(splits=['train', 'valid', 'test'], axes=axes_rvp)
plt.tight_layout()
b64_rvp = _fig_to_b64(fig_rvp)
plt.close(fig_rvp)

fig_ep, ax_ep = plt.subplots(figsize=(8, 5))
ev_reg.plot_error_percentile(splits=['train', 'valid', 'test'], ax=ax_ep)
plt.tight_layout()
b64_ep = _fig_to_b64(fig_ep)
plt.close(fig_ep)

fig_peb, ax_peb = plt.subplots(figsize=(8, 4))
ev_reg.plot_prediction_error_bins(split='test', n_bins=10, ax=ax_peb)
plt.tight_layout()
b64_peb = _fig_to_b64(fig_peb)
plt.close(fig_peb)

sections.append(
    _h2('12. RegressionEvaluator') +
    _code('''
        ev = RegressionEvaluator()
        ev.add('train', y_true_train, y_pred_train)
        ev.add('valid', y_true_valid, y_pred_valid)
        ev.add('test',  y_true_test,  y_pred_test)

        ev.add_default_metrics()   # mae, rmse, r2, mape, medae
        ev.add_metric('max_error')
        ev.add_metric(
            lambda yt, yp: float(np.percentile(np.abs(yt - yp), 90)),
            name='p90_abs_error',
        )
    ''') +
    _h3('Таблица метрик') +
    _table_html(ev_reg.metrics()) +
    _h3('Actual vs Predicted') +
    _code('''
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ev.plot_actual_vs_predicted(splits=['train', 'valid', 'test'], axes=axes)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_avp) +
    _h3('Распределение остатков') +
    _code('''
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ev.plot_residuals_distribution(splits=['train', 'valid', 'test'], axes=axes)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_res) +
    _h3('Остатки vs предсказания (гетероскедастичность)') +
    _code('''
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ev.plot_residuals_vs_predicted(splits=['train', 'valid', 'test'], axes=axes)
        plt.tight_layout()
        plt.show()
    ''') +
    _img_tag(b64_rvp) +
    _h3('Абсолютная ошибка по перцентилям') +
    _code('''
        ev.plot_error_percentile(splits=['train', 'valid', 'test'])
    ''') +
    _img_tag(b64_ep) +
    _h3('MAE по бинам реального значения') +
    _code('''
        ev.plot_prediction_error_bins(split='test', n_bins=10)
    ''') +
    _img_tag(b64_peb)
)

# ── 13. Сравнение нескольких моделей ──────────────────────────────────────────

def _make_ev(skill):
    ev = ClassificationEvaluator(task='binary')
    y_v, p_v = _make_cls_data(1000, skill=skill)
    y_t, p_t = _make_cls_data(1000, skill=skill - 0.04)
    ev.add('valid', y_v, p_v)
    ev.add('test',  y_t, p_t)
    # Только несколько метрик, чтобы фасетный график был компактным
    for m in ('roc_auc', 'pr_auc', 'ks', 'gini'):
        ev.add_metric(m)
    ev.add_metric(precision_at_k(0.10), name='precision@10%')
    ev.add_metric(lift_at_k(0.10),      name='lift@10%')
    return ev

evaluators = {
    'LightGBM':     _make_ev(0.78),
    'CatBoost':     _make_ev(0.76),
    'XGBoost':      _make_ev(0.74),
    'RandomForest': _make_ev(0.70),
}

cmp_df = compare_models(evaluators, split='valid')

def _plot_to_b64_file(fn, **kwargs) -> str:
    """Сохраняет график во временный файл и возвращает base64 PNG.

    plot_model_comparison / plot_model_delta используют _save_facet, который
    совмещает tight_layout() + subplots_adjust() + bbox_inches='tight' —
    с BytesIO это вызывает огромный размер изображения в некоторых версиях
    matplotlib. Сохранение в реальный файл обходит проблему.
    """
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp = f.name
    try:
        fn(path=tmp, **kwargs)
        with open(tmp, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    finally:
        os.unlink(tmp)

b64_cmp   = _plot_to_b64_file(plot_model_comparison, evaluators=evaluators, split='valid')
b64_heat  = _plot_to_b64_file(plot_model_heatmap,    evaluators=evaluators, split='valid')
b64_delta = _plot_to_b64_file(plot_model_delta, evaluators=evaluators, ref='LightGBM', split='valid')

sections.append(
    _h2('13. Сравнение нескольких моделей') +
    _code('''
        from ml_toolkit.model_evaluation import (
            compare_models, plot_model_comparison,
            plot_model_heatmap, plot_model_delta,
        )

        evaluators = {
            'LightGBM':     ev_lgb,
            'CatBoost':     ev_cat,
            'XGBoost':      ev_xgb,
            'RandomForest': ev_rf,
        }

        # DataFrame: строки = метрики, столбцы = модели
        df = compare_models(evaluators, split='valid')

        # Фасетный bar chart (одна панель на метрику)
        plot_model_comparison(evaluators, split='valid')

        # Тепловая карта (цвет = относительный ранг внутри метрики)
        plot_model_heatmap(evaluators, split='valid')

        # Дельта vs базовой модели
        plot_model_delta(evaluators, ref='LightGBM', split='valid')
    ''') +
    _h3('compare_models() — таблица (valid)') +
    _table_html(cmp_df) +
    _h3('plot_model_comparison()') +
    _img_tag(b64_cmp) +
    _h3('plot_model_heatmap()') +
    _img_tag(b64_heat) +
    _h3('plot_model_delta() vs LightGBM') +
    _img_tag(b64_delta)
)

# ── 14. HTML-отчёт ────────────────────────────────────────────────────────────

sections.append(
    _h2('14. Автоматический HTML-отчёт') +
    _p(
        'Метод <code>report(path)</code> генерирует самодостаточный HTML-файл: '
        'таблица метрик + все графики, изображения встроены как base64 — файл открывается без сети.'
    ) +
    _code('''
        ev.report('cls_report.html')   # ClassificationEvaluator
        ev.report('reg_report.html')   # RegressionEvaluator
    ''') +
    _p(
        'Для <code>ClassificationEvaluator(task="binary")</code> в отчёт попадают: '
        'таблица метрик, ROC, PR, Precision&amp;Recall@k, распределение скоров, CDF, '
        'калибровка, lift, gains, матрица ошибок, KS, decile bar, threshold scan, PSI.'
    )
)

# ── 15. Пример сборки кастомного дашборда (compose) ──────────────────────────

fig_dash, ax_arr = plt.subplots(2, 3, figsize=(16, 10))
ev_cls.plot_roc(splits=['valid', 'test'],       ax=ax_arr[0, 0])
ev_cls.plot_pr(splits=['valid', 'test'],        ax=ax_arr[0, 1])
ev_cls.plot_calibration(splits=['valid', 'test'], ax=ax_arr[0, 2])
ev_cls.plot_lift(splits=['valid', 'test'],      ax=ax_arr[1, 0])
ev_cls.plot_gains(splits=['valid', 'test'],     ax=ax_arr[1, 1])
ev_cls.plot_threshold_scan(split='valid',       ax=ax_arr[1, 2])
plt.suptitle('Кастомный дашборд', fontsize=13, y=1.01)
plt.tight_layout()
b64_dash = _fig_to_b64(fig_dash)
plt.close(fig_dash)

sections.append(
    _h2('15. Кастомный дашборд (ax= / axes=)') +
    _p(
        'Все plot-методы принимают <code>ax=</code> (одна панель) или <code>axes=</code> '
        '(несколько). Это позволяет собрать произвольный дашборд без промежуточных файлов.'
    ) +
    _code('''
        import matplotlib.pyplot as plt

        fig, ax_arr = plt.subplots(2, 3, figsize=(16, 10))
        ev.plot_roc(splits=['valid', 'test'],       ax=ax_arr[0, 0])
        ev.plot_pr(splits=['valid', 'test'],        ax=ax_arr[0, 1])
        ev.plot_calibration(splits=['valid', 'test'], ax=ax_arr[0, 2])
        ev.plot_lift(splits=['valid', 'test'],      ax=ax_arr[1, 0])
        ev.plot_gains(splits=['valid', 'test'],     ax=ax_arr[1, 1])
        ev.plot_threshold_scan(split='valid',       ax=ax_arr[1, 2])
        plt.suptitle('Кастомный дашборд')
        plt.tight_layout()
        plt.savefig('dashboard.png')
    ''') +
    _img_tag(b64_dash)
)

# ── Сборка HTML ────────────────────────────────────────────────────────────────

CSS = '''
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; background: #f6f8fa; color: #24292f;
}
.container { max-width: 1100px; margin: 0 auto; padding: 2em; }
h1 { font-size: 1.9em; border-bottom: 3px solid #0969da; padding-bottom: 0.3em; }
h2 { font-size: 1.35em; margin-top: 2.5em; border-bottom: 1px solid #d0d7de;
     padding-bottom: 0.25em; color: #0969da; }
h3 { font-size: 1.05em; margin-top: 1.5em; color: #57606a; }
p  { line-height: 1.65; }
pre.code {
    background: #161b22; color: #e6edf3;
    padding: 1em 1.2em; border-radius: 6px;
    overflow-x: auto; font-size: 0.82em; line-height: 1.55;
    margin: 0.8em 0 1.2em;
}
code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; }
p code {
    background: #eff1f3; color: #0550ae;
    padding: 0.1em 0.35em; border-radius: 4px; font-size: 0.88em;
}
.table-wrap { overflow-x: auto; margin: 0.8em 0 1.2em; }
table.t { border-collapse: collapse; font-size: 0.85em; }
table.t td, table.t th {
    border: 1px solid #d0d7de; padding: 5px 12px; text-align: right;
}
table.t th { background: #f0f6ff; text-align: center; font-weight: 600; }
table.t tr:nth-child(even) td { background: #f6f8fa; }
img { border: 1px solid #d0d7de; border-radius: 6px; }
nav {
    position: sticky; top: 0; background: #fff; border-bottom: 1px solid #d0d7de;
    padding: 0.5em 2em; font-size: 0.82em; z-index: 100;
}
nav a { margin-right: 1em; color: #0969da; text-decoration: none; }
nav a:hover { text-decoration: underline; }
'''

NAV_ITEMS = [
    ('sec1',  '1. Регистрация'),
    ('sec2',  '2. Сравнение сплитов'),
    ('sec3',  '3. ROC / PR'),
    ('sec4',  '4. Скоры'),
    ('sec5',  '5. Бизнес-кривые'),
    ('sec6',  '6. P&R@k'),
    ('sec7',  '7. Калибровка'),
    ('sec8',  '8. Порог'),
    ('sec9',  '9. PSI'),
    ('sec10', '10. Bootstrap'),
    ('sec11', '11. Custom метрики'),
    ('sec12', '12. Регрессия'),
    ('sec13', '13. Сравнение моделей'),
    ('sec14', '14. HTML-отчёт'),
    ('sec15', '15. Дашборд'),
]

nav_html = '<nav>' + ''.join(
    f'<a href="#{a}">{b}</a>' for a, b in NAV_ITEMS
) + '</nav>'

body_parts = []
for i, (sec_id, _) in enumerate(NAV_ITEMS):
    body_parts.append(f'<section id="{sec_id}">{sections[i]}</section>')

html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ml_toolkit.model_evaluation — руководство</title>
<style>{CSS}</style>
</head>
<body>
{nav_html}
<div class="container">
<h1>ml_toolkit.model_evaluation — руководство с примерами</h1>
{''.join(body_parts)}
</div>
</body>
</html>'''

OUT = Path(__file__).with_name('example.html')
OUT.write_text(html, encoding='utf-8')
print(f'HTML-пример сохранён: {OUT}  ({OUT.stat().st_size // 1024} КБ)')
