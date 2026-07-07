"""ml_toolkit.plots — визуализационные утилиты."""
from ml_toolkit.plots.annotations import annotate_bars
from ml_toolkit.plots.axes import log_axis, symmetrize_ylim
from ml_toolkit.plots.ml_viz import add_confusion_quadrant_labels, add_threshold_band
from ml_toolkit.plots.timeseries import (
    add_event_markers,
    add_forecast_region,
    add_period_bands,
)
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

__all__ = [
    # utils
    'number_to_number_with_suffix', 'modify_ticks', 'modify_ticks_percent',
    'modify_xticks_for_date_axis', 'add_bisector', 'add_vline', 'add_hline',
    'hide_spines', 'fill_region', 'apply_style', 'PALETTES',
    # новые
    'annotate_bars',
    'add_period_bands', 'add_forecast_region', 'add_event_markers',
    'add_threshold_band', 'add_confusion_quadrant_labels',
    'symmetrize_ylim', 'log_axis',
]
