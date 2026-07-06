"""Утилиты форматирования осей matplotlib."""
from .ticks import (
    number_to_number_with_suffix,
    modify_ticks,
    modify_ticks_percent,
    modify_xticks_for_date_axis,
)
from .lines import (
    add_bisector,
    add_vline,
    add_hline,
)
from .regions import (
    hide_spines,
    fill_region,
)
from .styles import (
    PALETTES,
    apply_style,
)

__all__ = [
    'number_to_number_with_suffix', 'modify_ticks', 'modify_ticks_percent',
    'modify_xticks_for_date_axis',
    'add_bisector', 'add_vline', 'add_hline',
    'hide_spines', 'fill_region',
    'PALETTES', 'apply_style',
]
