"""Индикатор разрыва (гэпа) активности по сегментации ряда: 1 внутри разрыва, иначе 0.

Signal:
    Единственный трансформер, чей выход НЕ зануляется движком в разрывах — это его
    основная задача, показать, где проходят разрывы. Опционален: не подключается
    автоматически при использовании `segment` у других трансформеров, запрашивается
    явно в feature_spec/preset, как и любой другой трансформер. Используется вместе
    с наблюдением, что фичи с `segment` дают NaN ровно там, где segment_gap = 1.

Formula:
    Строится тем же однопроходным run-length-алгоритмом, что и подмена позиции
    для сегментированных трансформеров (`ml_toolkit.transformers._segmentation`):
    N=gap_threshold подряд идущих триггеров (условие зависит от strategy) образуют
    "grace period" ещё внутри сегмента, дальнейшие подряд идущие триггеры — разрыв,
    вплоть до первой не-триггерящей строки.
    segment_gap = 0.0, если строка в сегменте (in_segment=True)
    segment_gap = 1.0, если строка в разрыве (in_segment=False)

Outputs:
    {product}__segment_gap__seg-zerogap2 — при segment: {strategy: zero_gap, gap_threshold: 2}
    Суффикс кодирует конфиг сегмента (см. `segment_suffix_fragment`), как и у любого
    другого трансформера, использующего `segment`.

Preset:
    segment_gap:
      segment: short_gap   # ссылка на именованный конфиг в секции `segments:` пресета

    Ограничение: strategy='mask' не поддерживается для segment_gap напрямую — этому
    трансформеру видны только values и position своей product-колонки, а `mask`
    требует данные из ДРУГОЙ колонки, которые кернелам не передаются (маска
    резолвится движком `feature_generation.py` уровнем выше, до вызова compute()
    целевого трансформера). Для zero_gap/relative_gap ограничения нет — они строятся
    исключительно по values/position, доступным здесь.

Interpretation:
    segment_gap = 1 для всей группы строк, где сегментированные фичи этой же
    product-колонки получат NaN — удобно для фильтрации/диагностики без
    необходимости заново запускать сегментацию вручную.

Example:
    Ряд (15 мес): [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 4, 3, 5, 2]
    segment: {strategy: zero_gap, gap_threshold: 2}

    Первые 2 нуля (idx5,6) — grace period, ещё в сегменте; idx7..10 — разрыв.
    → segment_gap = [0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0]

"""

import numpy as np

from ml_toolkit.transformers._segmentation import compute_segment_position

FEATURE = 'segment_gap'


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"segment": {"strategy": "zero_gap", "gap_threshold": 2, ...}} (уже резолвлен)."""
    segment_cfg = params['segment']
    strategy = segment_cfg['strategy']
    if strategy == 'mask':
        raise ValueError(
            "segment_gap: strategy='mask' не поддерживается напрямую через "
            'compute() — маска читается движком feature_generation.py на уровне '
            'оркестрации, а не отдельным кернелом (см. докстринг модуля).'
        )
    _, in_segment = compute_segment_position(values, position, strategy, segment_cfg)
    return [np.where(in_segment, 0.0, 1.0)], ['']
