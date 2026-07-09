from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ml_toolkit.plots.utils import (
        PALETTES,
        add_bisector,
        add_hline,
        add_vline,
        apply_style,
        fill_region,
        hide_spines,
        modify_ticks,
        modify_ticks_percent,
        modify_xticks_for_date_axis,
        number_to_number_with_suffix,
    )

    rng  = np.random.default_rng(42)
    fig  = plt.figure(figsize=(18, 17))
    gs   = gridspec.GridSpec(4, 3, figure=fig, hspace=0.68, wspace=0.40)


    # ══ ROW 0 ════════════════════════════════════════════════════════════════════

    # ── add_bisector ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    y_true = rng.uniform(0, 10, 250)
    y_pred = y_true + rng.normal(0, 1.5, 250)
    ax.scatter(y_true, y_pred, alpha=0.35, s=14, color='#4E79A7', zorder=2)
    add_bisector(ax, color='crimson', lw=1.5, label='y = x (идеал)')
    ax.set_xlabel('Факт', fontsize=9);  ax.set_ylabel('Предсказание', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title('add_bisector', fontsize=10, fontweight='bold')
    hide_spines(ax)


    # ── add_vline  /  add_hline — rotation=0 vs rotation=90 ─────────────────────
    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 12, 400)
    ax.plot(t, np.sin(t) * np.exp(-t / 14) + rng.normal(0, 0.05, 400),
            color='#4E79A7', lw=1.5)

    # add_vline: rotation=90 (вдоль, дефолт) vs rotation=0 (перпендикулярно)
    add_vline(ax, x=3.0,  label='vline rot=90', loc='top',    color='#E15759', rotation=90)
    add_vline(ax, x=8.0,  label='vline rot=0',  loc='top',    color='#C0392B', rotation=0)

    # add_hline: rotation=0 (вдоль, дефолт) vs rotation=90 (перпендикулярно)
    add_hline(ax, y=0.55, label='hline rot=0',  loc='right',  color='#2ECC71', lw=1.2, rotation=0)
    add_hline(ax, y=-0.3, label='hline rot=90', loc='center', color='#27AE60', lw=1.2,
              linestyle=':', rotation=90)

    ax.set_title('add_vline / add_hline — rotation', fontsize=10, fontweight='bold')
    hide_spines(ax)


    # ── fill_region  (все 9 позиций label_loc) ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.linspace(0, 1, 80), np.sin(np.linspace(0, 2*np.pi, 80)) * 0.3 + 0.5,
            color='#4E79A7', lw=1.5, zorder=3)
    locs = [
        ('top-left',      0.00, 0.33, '#E15759'),
        ('top-center',    0.33, 0.66, '#F28E2B'),
        ('top-right',     0.66, 1.00, '#9467BD'),
        ('center-left',   0.00, 0.33, '#59A14F'),
        ('center',        0.33, 0.66, '#4E79A7'),
        ('center-right',  0.66, 1.00, '#EDC948'),
        ('bottom-left',   0.00, 0.33, '#8C564B'),
        ('bottom-center', 0.33, 0.66, '#76B7B2'),
        ('bottom-right',  0.66, 1.00, '#E15759'),
    ]
    for label, x1, x2, c in locs:
        fill_region(ax, x1, x2, label=label, color=c, alpha=0.18,
                    label_loc=label, fontsize=7.5)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_title('fill_region — все 9 позиций label_loc', fontsize=9, fontweight='bold')
    hide_spines(ax)


    # ══ ROW 1: форматирование тиков ══════════════════════════════════════════════

    # ── modify_ticks (K / M / B) ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн']
    values = [1_200_000, 4_500_000, 2_300_000, 8_100_000, 5_600_000, 12_400_000]
    bars = ax.bar(months, values, color=PALETTES['muted'][:6])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 200_000,
                number_to_number_with_suffix(v), ha='center', va='bottom', fontsize=8)
    modify_ticks(ax, axis='y')
    ax.set_title('modify_ticks (K/M/B)  +  number_to_number_with_suffix', fontsize=9, fontweight='bold')
    hide_spines(ax)


    # ── modify_ticks_percent ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    quarters   = ['Q1', 'Q2', 'Q3', 'Q4', "Q1'24", "Q2'24"]
    conversion = [0.12, 0.19, 0.17, 0.26, 0.31, 0.38]
    churn      = [0.08, 0.11, 0.09, 0.07, 0.06, 0.05]
    ax.plot(quarters, conversion, 'o-',  lw=1.8, color='#4E79A7', label='Конверсия')
    ax.plot(quarters, churn,      's--', lw=1.4, color='#E15759', label='Отток')
    modify_ticks_percent(ax, axis='y')
    ax.legend(fontsize=8);  ax.set_ylim(0, 0.45)
    ax.set_title('modify_ticks_percent', fontsize=10, fontweight='bold')
    hide_spines(ax)


    # ── modify_xticks_for_date_axis ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    dates  = pd.date_range('2023-01-01', periods=20, freq='ME')
    series = np.cumsum(rng.normal(0, 1, 20)) + 20
    ax.fill_between(dates, series, alpha=0.25, color='steelblue')
    ax.plot(dates, series, lw=1.8, color='steelblue')
    modify_xticks_for_date_axis(ax, rotation=30, lang='ru')
    ax.set_title('modify_xticks_for_date_axis', fontsize=10, fontweight='bold')
    hide_spines(ax)


    # ══ ROW 2: apply_style ═══════════════════════════════════════════════════════

    def _waves(ax):
        x = np.linspace(0, 2 * np.pi, 120)
        for k in range(5):
            ax.plot(x, np.sin(x + k * 0.6) * (1 - k * 0.12), lw=1.6)

    ax = fig.add_subplot(gs[2, 0])
    apply_style(ax, 'clean', palette='muted')
    _waves(ax)
    ax.set_title("style='clean'  palette='muted'", fontsize=9, fontweight='bold')

    ax = fig.add_subplot(gs[2, 1])
    apply_style(ax, 'minimal', palette='corporate')
    _waves(ax)
    ax.set_title("style='minimal'  palette='corporate'", fontsize=9, fontweight='bold')

    ax = fig.add_subplot(gs[2, 2])
    apply_style(ax, 'dark', palette='muted')
    fig.set_facecolor('white')
    _waves(ax)
    ax.set_title("style='dark'", fontsize=9, fontweight='bold', color='#DDDDDD')


    # ══ ROW 3 ════════════════════════════════════════════════════════════════════

    ax = fig.add_subplot(gs[3, 0])
    apply_style(ax, 'presentation', palette='dark')
    _waves(ax)
    ax.set_title("style='presentation'  palette='dark'", fontsize=9, fontweight='bold')

    # ── hide_spines — сравнение before/after ─────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    x = np.linspace(0, 3, 100)
    y = np.sin(x * 2) * np.exp(-x / 3)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ins

    ax.set_visible(False)
    ax_l = _ins(ax, width='45%', height='90%', loc='center left',
                bbox_to_anchor=(-0.05, 0, 1, 1), bbox_transform=ax.transAxes)
    ax_r = _ins(ax, width='45%', height='90%', loc='center right',
                bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax.transAxes)
    for a in (ax_l, ax_r):
        a.plot(x, y, color='steelblue', lw=1.8)
        a.set_xticks([]);  a.set_yticks([])
    ax_l.set_title('до', fontsize=9, color='gray')
    hide_spines(ax_r, ('top', 'right', 'bottom', 'left'))
    ax_r.set_title('hide_spines\n(top,right,bottom,left)', fontsize=8, color='gray')
    ax.annotate('', xy=(0.52, 0.5), xytext=(0.42, 0.5),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.set_visible(True);  ax.axis('off')
    ax.set_title('hide_spines — до / после', fontsize=10, fontweight='bold')


    # ── PALETTES swatches ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    for row, (name, colors) in enumerate(PALETTES.items()):
        for col, c in enumerate(colors):
            ax.barh(row, 1, left=col, color=c, height=0.72, edgecolor='white', lw=1.5)
        ax.text(-0.2, row, name, ha='right', va='center', fontsize=9.5)
    ax.set_xlim(-1.8, 7);  ax.set_ylim(-0.6, len(PALETTES) - 0.4)
    ax.axis('off')
    ax.set_title('PALETTES', fontsize=10, fontweight='bold')


    fig.suptitle('ml_toolkit/plots/utils — полный обзор утилит', fontsize=14, y=1.01, fontweight='bold')

    OUT = Path(__file__).parent / 'utils_demo.png'
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'OK: {OUT}')
