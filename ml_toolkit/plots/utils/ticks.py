"""Форматирование числовых значений и тиков осей matplotlib."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_RU_MONTHS: list[str] = [
    'янв', 'фев', 'мар', 'апр', 'май', 'июн',
    'июл', 'авг', 'сен', 'окт', 'ноя', 'дек',
]


def number_to_number_with_suffix(
    value: float,
    add_new_line_character: bool = False,
) -> str:
    """Форматирует число с суффиксом (K, M, B)."""
    sep = '\n' if add_new_line_character else ' '
    abs_val = abs(value)
    sign = '-' if value < 0 else ''
    if abs_val >= 1e9:
        return f'{sign}{abs_val / 1e9:.1f}{sep}B'
    if abs_val >= 1e6:
        return f'{sign}{abs_val / 1e6:.1f}{sep}M'
    if abs_val >= 1e3:
        return f'{sign}{abs_val / 1e3:.1f}{sep}K'
    return f'{sign}{abs_val:.0f}'


def modify_ticks(ax: plt.Axes, axis: str = 'y', func=None) -> None:
    """Форматирует тики оси с суффиксами (K/M/B)."""
    _fn = func if func is not None else number_to_number_with_suffix
    formatter = mticker.FuncFormatter(lambda x, _: _fn(x))
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)


def modify_ticks_percent(ax: plt.Axes, axis: str = 'y') -> None:
    """Форматирует тики оси в процентах."""
    formatter = mticker.PercentFormatter(xmax=1.0, decimals=0)
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(formatter)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(formatter)


def modify_xticks_for_date_axis(
    ax: plt.Axes,
    rotation: int = 30,
    fmt: str = 'auto',
    lang: str | None = None,
) -> None:
    """Форматирует ось X как даты с автоматическим выбором локатора.

    Args:
        fmt:  strftime-строка или 'auto' — ConciseDateFormatter (убирает повторяющийся год).
        lang: 'ru' — русские названия месяцев без зависимости от системного locale;
              год показывается только при первом появлении и смене.

    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    if lang == 'ru':
        _state: dict = {'prev_year': None}

        def _ru_fmt(x, pos):
            dt = mdates.num2date(x)
            m = _RU_MONTHS[dt.month - 1]
            if dt.year != _state['prev_year']:
                _state['prev_year'] = dt.year
                return f'{m} {dt.year}'
            return m

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_ru_fmt))
    elif fmt == 'auto':
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
