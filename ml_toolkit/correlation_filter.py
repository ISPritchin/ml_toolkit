"""Жадный отбор фич по корреляции.

Правило из задания: для пары столбцов сначала отсеиваются наблюдения,
где ОБА значения равны нулю, и только на оставшихся считается корреляция
Пирсона. Если |corr| > threshold хотя бы с одним из уже принятых
признаков - новый признак не добавляется в набор.
"""

import logging

import numpy as np
import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _compute_correlation_excluding_both_zero(
    left_values: np.ndarray, right_values: np.ndarray
) -> float:
    """Корреляция Пирсона по наблюдениям, где не оба значения равны нулю.

    Args:
        left_values: Значения первого признака.
        right_values: Значения второго признака (та же длина, что
            `left_values`).

    Returns:
        Коэффициент корреляции Пирсона по наблюдениям, где хотя бы одно из
        `left_values`/`right_values` отлично от нуля. Возвращает 0.0, если
        таких наблюдений меньше двух или один из признаков на них
        константен (корреляция не определена).
    """
    keep_mask = ~((left_values == 0.0) & (right_values == 0.0))
    if keep_mask.sum() < 2:
        return 0.0
    filtered_left = left_values[keep_mask]
    filtered_right = right_values[keep_mask]
    if np.std(filtered_left) == 0.0 or np.std(filtered_right) == 0.0:
        return 0.0
    return float(np.corrcoef(filtered_left, filtered_right)[0, 1])


def filter_correlated_features(
    df: pl.DataFrame,
    candidate_cols: list[str],
    threshold: float = 0.9,
    preselected_cols: list[str] | None = None,
    max_rows_for_correlation: int | None = 100_000,
    random_seed: int = 0,
) -> list[str]:
    """Жадно отбирает кандидатов фич, отбрасывая коррелирующие с уже принятыми.

    Идёт по `candidate_cols` по порядку и копит список принятых колонок.
    Каждый новый кандидат сравнивается со всеми уже принятыми (включая
    `preselected_cols`, если они переданы как стартовый набор) при помощи
    `_compute_correlation_excluding_both_zero` - если корреляция по модулю
    выше `threshold` хотя бы с одной из принятых колонок, кандидат
    отбрасывается.

    Доминирующая стоимость этой функции - O(n_candidates * n_accepted)
    вызовов `np.corrcoef`, каждый по строкам `df`. Чтобы не зависеть от
    размера `df` на больших датасетах, корреляции считаются на случайной
    подвыборке строк фиксированного размера (`max_rows_for_correlation`), а
    не на всех строках - оценка корреляции от этого статистически не
    страдает (нужный размер выборки для устойчивой оценки корреляции не
    растёт линейно с числом строк), а сама функция перестаёт замедляться
    пропорционально объёму данных.

    Args:
        df: Датасет, содержащий колонки из `candidate_cols` (и
            `preselected_cols`, если они переданы).
        candidate_cols: Имена колонок-кандидатов в порядке, в котором их
            нужно рассматривать (порядок влияет на итоговый набор, так как
            отбор жадный).
        threshold: Порог абсолютной корреляции, выше которого кандидат
            считается избыточным.
        preselected_cols: Необязательный стартовый набор уже принятых
            колонок - новые кандидаты сравниваются и с ними, но сами они не
            попадают в возвращаемый список.
        max_rows_for_correlation: Максимум строк, используемых для расчёта
            корреляции. Если в `df` строк больше - используется случайная
            подвыборка этого размера (без возвращения, фиксированная для
            всех пар в рамках одного вызова). `None` - использовать все
            строки без сэмплирования.
        random_seed: Сид генератора случайных чисел для подвыборки строк
            (см. `max_rows_for_correlation`) - фиксирован по умолчанию для
            воспроизводимости отбора фич между запусками.

    Returns:
        Подмножество `candidate_cols` (в исходном порядке появления), не
        коррелирующее по модулю выше `threshold` ни с одним из ранее
        принятых признаков.
    """
    accepted_cols: list[str] = list(preselected_cols) if preselected_cols else []
    column_arrays: dict[str, np.ndarray] = {
        col: df[col].to_numpy() for col in set(candidate_cols) | set(accepted_cols)
    }

    n_rows = df.height
    if max_rows_for_correlation is not None and n_rows > max_rows_for_correlation:
        rng = np.random.default_rng(random_seed)
        row_sample_idx = rng.choice(n_rows, size=max_rows_for_correlation, replace=False)
        column_arrays = {col: arr[row_sample_idx] for col, arr in column_arrays.items()}
        logger.info(
            "Корреляции считаются на подвыборке %d из %d строк (random_seed=%d)",
            max_rows_for_correlation,
            n_rows,
            random_seed,
        )

    logger.info(
        "Корреляционный фильтр: %d кандидатов, threshold=%.3f, %d предвыбранных колонок",
        len(candidate_cols),
        threshold,
        len(accepted_cols),
    )

    n_dropped = 0
    for candidate_col in tqdm(candidate_cols, desc="Корреляционный фильтр", unit="фича"):
        if candidate_col in accepted_cols:
            continue
        candidate_values = column_arrays[candidate_col]
        is_redundant = False
        for accepted_col in accepted_cols:
            correlation = _compute_correlation_excluding_both_zero(
                candidate_values, column_arrays[accepted_col]
            )
            if abs(correlation) > threshold:
                is_redundant = True
                logger.debug(
                    "Отброшен '%s': корреляция %.3f с уже принятым '%s'",
                    candidate_col,
                    correlation,
                    accepted_col,
                )
                break
        if not is_redundant:
            accepted_cols.append(candidate_col)
        else:
            n_dropped += 1

    result = [col for col in accepted_cols if col in candidate_cols]
    logger.info(
        "Корреляционный фильтр завершён: принято %d, отброшено %d из %d кандидатов",
        len(result),
        n_dropped,
        len(candidate_cols),
    )
    return result
