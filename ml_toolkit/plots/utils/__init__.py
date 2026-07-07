"""Утилиты форматирования осей matplotlib."""
from .lines import (
    add_bisector,
    add_hline,
    add_vline,
)
from .regions import (
    fill_region,
    hide_spines,
)
from .styles import (
    PALETTES,
    apply_style,
)
from .ticks import (
    modify_ticks,
    modify_ticks_percent,
    modify_xticks_for_date_axis,
    number_to_number_with_suffix,
)

__all__ = [
    'PALETTES',
    'add_bisector',
    'add_hline',
    'add_vline',
    'apply_style',
    'fill_region',
    'hide_spines',
    'modify_ticks',
    'modify_ticks_percent',
    'modify_xticks_for_date_axis',
    'number_to_number_with_suffix',
]
