"""Разбиение ряда клиента на независимые сегменты активности (сквозной разрыв истории).

Мотивация: все 81 кернел в `ml_toolkit/transformers/kernels/` не знают, ПОЧЕМУ
`position_within_entity` сбрасывается в 0 — они лишь используют это через
`resolve_window_size(pos, w)` и `if pos == 0: сброс аккумуляторов`. Поэтому
сегментация не требует ни одного изменения в кернелах: достаточно подменить
`position_within_entity` на альтернативный `position_within_segment`,
вычисленный этим модулем, и передать его в `module.compute(...)` вместо
обычной позиции — с точки зрения кернела это неотличимо от границы сущности.

Стратегии определяют, какие строки «триггерят» потенциальный разрыв:
    - zero_gap:     триггер — value == 0
    - relative_gap: триггер — value < relative_threshold * (среднее по reference_window)
    - mask:         триггер — внешняя булева маска (True=активен) инвертирована;
                     сама маска читается вызывающим кодом (feature_generation.py),
                     а не этим модулем — он лишь принимает готовый external_mask.

Все три стратегии превращают ряд is_trigger в (position_within_segment, in_segment)
через ОДИН общий примитив `_run_length_segment`: gap_threshold подряд идущих
триггеров переводит хвост ряда в состояние «разрыв» (in_segment=False) вплоть до
первой не-триггерящей строки, с которой начинается новый сегмент (position=0).
Границы entity (position_within_entity[i]==0) сбрасывают состояние сегментации
безусловно — сегмент никогда не переходит через границу сущности.

Особый случай — zero_gap и ведущие нули: значения-триггеры (нули) до самого
первого ненулевого значения сущности не получают grace period gap_threshold и
исключаются целиком, каким бы коротким ни был этот ведущий пробег (даже 1 ноль).
Это не «разрыв в середине активности», а признак того, что ряд клиента ещё не
начался — тот же принцип, что и у new_client_flag/client_age (см. kernels/client_age.py).
`relative_gap`/`mask` этого не делают: relative_gap завязан на собственный
недавний уровень (нет уровня — нет триггера в вырожденном случае), mask задаёт
активность извне и «ещё не начался» там уже выражено самой маской.
"""

import numba as nb
import numpy as np

from ._windowing import compute_window_mean, resolve_window_size

SEGMENT_STRATEGIES = ('zero_gap', 'relative_gap', 'mask')

_COMMON_REQUIRED_KEYS = {'strategy', 'gap_threshold'}
_STRATEGY_EXTRA_REQUIRED_KEYS = {
    'zero_gap': set(),
    'relative_gap': {'reference_window', 'relative_threshold'},
    'mask': {'mask_column'},
}


@nb.njit(cache=True)
def _run_length_segment(
    is_trigger: np.ndarray,
    position_within_entity: np.ndarray,
    gap_threshold: int,
    exclude_leading_triggers: bool = False,
):
    """Однопроходное превращение is_trigger в (position_within_segment, in_segment).

    Строка i попадает в разрыв, если ей предшествует >= gap_threshold подряд
    идущих триггеров (включая саму себя — строка, на которой счётчик достигает
    порога, ещё считается частью сегмента; разрыв начинается со следующей).
    Выход из разрыва — первая не-триггерящая строка, с неё начинается новый
    сегмент с position_within_segment=0. Состояние сбрасывается на границе
    сущности (position_within_entity[i]==0), независимо от gap_threshold.

    exclude_leading_triggers=True: триггеры до первой не-триггерящей строки
    сущности не получают grace period gap_threshold — исключаются целиком
    (ряд ещё не начался), сколько бы их ни было подряд. Обычная grace-period
    логика включается только с первой не-триггерящей строки и действует для
    всех последующих пробегов триггеров как раньше.
    """
    n = is_trigger.shape[0]
    position_within_segment = np.zeros(n, dtype=np.int64)
    in_segment = np.zeros(n, dtype=np.bool_)

    run_length = 0
    in_gap = False
    seg_pos = 0
    started = not exclude_leading_triggers

    for i in range(n):
        if position_within_entity[i] == 0:
            run_length = 0
            in_gap = False
            seg_pos = 0
            started = not exclude_leading_triggers

        if not started:
            if is_trigger[i]:
                position_within_segment[i] = 0
                in_segment[i] = False
                continue
            started = True

        if in_gap and not is_trigger[i]:
            in_gap = False
            seg_pos = 0

        if in_gap:
            position_within_segment[i] = 0
            in_segment[i] = False
        else:
            position_within_segment[i] = seg_pos
            in_segment[i] = True

        if is_trigger[i]:
            run_length += 1
        else:
            run_length = 0

        if run_length >= gap_threshold:
            in_gap = True

        seg_pos = 0 if in_gap else seg_pos + 1

    return position_within_segment, in_segment


@nb.njit(cache=True)
def _trigger_zero_gap(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        out[i] = values[i] == 0.0
    return out


@nb.njit(cache=True)
def _trigger_relative_gap(
    values: np.ndarray,
    position_within_entity: np.ndarray,
    reference_window: int,
    relative_threshold: float,
) -> np.ndarray:
    n = values.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        ws = resolve_window_size(position_within_entity[i], reference_window)
        ref = compute_window_mean(values, i, ws)
        out[i] = values[i] < relative_threshold * ref
    return out


def compute_segment_position(
    values: np.ndarray,
    position: np.ndarray,
    strategy: str,
    params: dict,
    external_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Резолвит стратегию сегментации в (position_within_segment, in_segment).

    Args:
        values: Значения product-колонки (та же строка `values`, что видит
            кернел-получатель).
        position: position_within_entity для того же ряда.
        strategy: 'zero_gap' | 'relative_gap' | 'mask'.
        params: Резолвленный конфиг сегмента (см. `SEGMENT_STRATEGIES`),
            должен содержать 'gap_threshold' и специфичные для стратегии ключи.
        external_mask: Только для strategy='mask' — булев массив (True=строка
            активна/не в разрыве), читается вызывающим кодом заранее.

    Returns:
        (position_within_segment, in_segment) — та же форма, что и values.

    Raises:
        ValueError: Неизвестная стратегия, gap_threshold < 1, либо
            strategy='mask' без external_mask.

    """
    gap_threshold = int(params['gap_threshold'])
    if gap_threshold < 1:
        raise ValueError(f'segment: gap_threshold должен быть >= 1, получено {gap_threshold}')

    if strategy == 'zero_gap':
        is_trigger = _trigger_zero_gap(values)
        # Ведущие нули = ряд ещё не начался, а не «пауза в активности» —
        # исключаются целиком, без grace period (см. докстринг модуля).
        exclude_leading_triggers = True
    elif strategy == 'relative_gap':
        is_trigger = _trigger_relative_gap(
            values, position, int(params['reference_window']), float(params['relative_threshold'])
        )
        exclude_leading_triggers = False
    elif strategy == 'mask':
        if external_mask is None:
            raise ValueError("segment: strategy='mask' требует external_mask (см. mask_column)")
        is_trigger = ~external_mask.astype(np.bool_)
        exclude_leading_triggers = False
    else:
        raise ValueError(f'segment: неизвестная стратегия {strategy!r}. Доступные: {SEGMENT_STRATEGIES}')

    return _run_length_segment(is_trigger, position, gap_threshold, exclude_leading_triggers)


def validate_segment_config(cfg: dict, context: str) -> dict:
    """Проверяет структуру резолвленного конфига сегмента (не строку-ссылку).

    Args:
        cfg: Словарь с ключом 'strategy' и специфичными для неё параметрами.
        context: Человекочитаемый контекст для сообщения об ошибке.

    Returns:
        cfg без изменений (для удобного встраивания в цепочку резолва).

    Raises:
        ValueError: Неизвестная стратегия или отсутствуют обязательные ключи.

    """
    strategy = cfg.get('strategy')
    if strategy not in _STRATEGY_EXTRA_REQUIRED_KEYS:
        raise ValueError(
            f'{context}: неизвестная стратегия сегментации {strategy!r}. '
            f'Доступные: {SEGMENT_STRATEGIES}'
        )
    required = _COMMON_REQUIRED_KEYS | _STRATEGY_EXTRA_REQUIRED_KEYS[strategy]
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"{context}: стратегии '{strategy}' не хватает ключей {sorted(missing)}")
    return cfg


def segment_suffix_fragment(cfg: dict) -> str:
    """Каноническая строка-суффикс для конфига сегмента: 'seg-{strategy}{param}'.

    Используется и как компонент имени выходной колонки, и как сигнатура для
    дедупликации/конфликт-детекции в `_resolve_feature_spec` — две группы,
    запросившие один и тот же трансформер с РАЗНЫМИ сегментами, дают разные
    фрагменты и потому не конфликтуют, а сосуществуют как разные кандидаты.
    """
    strategy = cfg['strategy']
    gap = int(cfg['gap_threshold'])
    if strategy == 'zero_gap':
        return f'seg-zerogap{gap}'
    if strategy == 'relative_gap':
        thr = _fmt_num(cfg['relative_threshold'])
        ref = int(cfg['reference_window'])
        return f'seg-relgap{thr}r{ref}g{gap}'
    if strategy == 'mask':
        return f"seg-mask-{cfg['mask_column']}g{gap}"
    raise ValueError(f'segment: неизвестная стратегия {strategy!r}. Доступные: {SEGMENT_STRATEGIES}')


def _fmt_num(x: float) -> str:
    return f'{x:g}'.replace('.', 'p').replace('-', 'm')
