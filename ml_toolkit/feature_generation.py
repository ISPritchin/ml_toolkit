"""Генерический движок наварки и отбора фич для одного датасета за раз.

Модуль ничего не знает про то, сколько датасетов есть у вызывающей задачи и как
они связаны между собой — это уровень оркестрации конкретной бизнес-задачи (см.,
например, `tasks/auto_kkp_classification/feature_generation.py`, где два датасета
разной гранулярности комбинируются через два вызова функций отсюда). Публичных
сценариев два уровня — общий (групповой) и его частный случай (uniform):

`generate_feature_groups(df, ..., feature_spec, ...)` (общий):
  `feature_spec: list[tuple[columns, preset]]` — разным product-колонкам можно
  назначить разные трансформеры и разные параметры за один вызов. `columns` —
  имя колонки, объект `polars.selectors`/`pl.Expr`, или список/кортеж любой их
  смеси. `preset` **обязателен** — автоматического пресета по умолчанию нет:
  либо имя/`Path`/полный путь к yaml-файлу (применяется целиком), либо явный
  словарь `{transformer_name: params}` (можно перечислить только нужные
  трансформеры со своими параметрами; пустой `{}` — pass-through без наварки).
  Несколько групп могут называть одну и ту же колонку — совпадающая пара
  (колонка, трансформер) с одинаковыми параметрами вычисляется ровно один раз
  (без повторной наварки), а с разными параметрами — это конфликт (`ValueError`).
  Корреляционный фильтр (Phase B) по умолчанию **не запускается**
  (`corr_threshold=None`) — раз колонки/трансформеры выбраны явно, автоматическое
  отбрасывание фич по умолчанию не нужно; передайте `corr_threshold`, чтобы
  включить его. Возвращает список итоговых колонок, чтобы применить тот же набор
  к другому датасету той же схемы через `apply_feature_groups`.

`select_features(df, ..., product_cols, transformer_names, preset, ...)` (частный случай):
  Резолвит `transformer_names`+`preset` (оба необязательны, есть дефолты) в явный
  словарь `{transformer_name: params}` и передаёт его в
  `generate_feature_groups(feature_spec=[(product_cols, resolved)], ...)`: один и
  тот же набор трансформеров (и один пресет) — на все `product_cols`,
  корреляционный фильтр включён по умолчанию (`corr_threshold=0.9`). Возвращает
  `accepted_cols` — список принятых колонок, чтобы вызывающий код мог применить
  тот же набор к другому датасету той же схемы через `apply_selected_features`.

`apply_feature_groups`/`apply_selected_features` (парные к наварке выше,
второй датасет той же схемы):
  Наваривает для `df` только уже готовый список `accepted_cols` (без повторного
  запуска корреляционного фильтра) и пишет результат в `out_path`. Нужен, когда
  два датасета должны получить идентичную схему фич, но набор фич выбирается по
  одному из них (обычно — по более гранулярному/представительному).
  `apply_feature_groups` принимает тот же `feature_spec`, что и парный
  `generate_feature_groups`; `apply_selected_features` — его uniform-обёртка,
  парная к `select_features`.

Фазы внутри `generate_feature_groups`/`select_features`
──────────────────────────────
Phase A — Наварка кандидатов:
  Для каждой пары (product-колонка, трансформер) наваривается набор
  признаков-кандидатов и сохраняется сразу в
  {tmp_dir}/{name}_{product_col}__{transformer_name}.parquet.
  В памяти одновременно — массивы только одного трансформера одной
  product-колонки; полный широкий датасет кандидатов не материализуется.
  float32 применяется при записи в numpy, до записи в parquet.

Phase B — Стриминговый корреляционный фильтр:
  Жадный Пирсон загружает кандидатов по одной колонке из parquet (columnar
  pruning), держит принятые в памяти. Пиковая нагрузка:
  n_accepted × n_sample_rows float32 + 1 кандидат.

Phase C — Итоговый датасет:
  Из parquet-файлов читаются только принятые колонки, группа за группой
  (каждый файл — не более одного раза, после чего освобождается). Итоговый
  датасет строится через pyarrow без промежуточного pl.concat. Если заданы
  min_output_ts_key / max_output_ts_key, строки вне диапазона отсеиваются
  при загрузке через маску — они не попадают в промежуточные массивы.

`apply_feature_groups`/`apply_selected_features` (второй датасет той же схемы):
  Признаки навариваются заново, но сохраняются только те, что переданы в
  `accepted_cols`. Если заданы min_output_ts_key / max_output_ts_key, маска строк
  применяется при накоплении массивов — строки вне диапазона не занимают память.
  Корреляционный фильтр не запускается — вызывающая задача сама решает, какой
  датасет ведёт отбор, а какой переиспользует готовый список.

Реестр трансформеров (ml_toolkit.transformers.TRANSFORMERS)
═════════════════════════════════════════════════════
Каждый трансформер — отдельный модуль в ml_toolkit/transformers/ с интерфейсом:

    FEATURE: str                                      # уникальное имя
    compute(values, position, params) -> (arrays, suffixes)

Параметры трансформеров (окна, лаги и т.п.) задаются через пресет —
словарь {feature_name: params_dict}. В select_features/apply_selected_features
пресет передаётся общим параметром preset=<path>/<dict> (None по умолчанию грузит
ml_toolkit/transformers/presets/monthly.yaml); в generate_feature_groups/
apply_feature_groups пресет — второй элемент каждой пары в feature_spec,
(columns, preset), и обязателен — автоматического пресета по умолчанию там нет,
так что разным группам колонок можно (и нужно) назначать свои пресеты явно.

Именование признаков:
  - Если suffix непустой: {product_col}__{feature}__{suffix}
  - Если suffix пустой:   {product_col}__{feature}
"""

import contextlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from .transformers import TRANSFORMERS
from .transformers._windowing import compute_position_within_entity

logger = logging.getLogger(__name__)

_DEFAULT_PRESET_PATH = Path(__file__).parent / "transformers" / "presets" / "monthly.yaml"

# Размер row group для временных parquet-файлов. Одинаковое значение для всех
# файлов гарантирует выравнивание границ — _build_output_from_parquets читает
# row group i из base и из каждого файла фич одновременно без load всего файла.
# Настройка: уменьшите для экономии памяти на малых датасетах (< 10M rows),
# увеличьте для ускорения на больших датасетах (> 100M rows).
_ROW_GROUP_SIZE = 100_000

# Порог Пирсоновской корреляции |r| для корреляционного фильтра по умолчанию.
# Значение 0.9 означает: кандидат исключается, если |корреляция| > 0.9.
# Обоснование: 0.9 отсеивает сильно коррелированные признаки (~81% общей дисперсии),
# при этом сохраняя разнообразие сигналов. Значения 0.85-0.95 обычно оптимальны
# для балланса между размером модели и качеством.
# Настройка: уменьшите до 0.7-0.8 для более строгого отбора (меньше фич),
# увеличьте до 0.95+ для более мягкого (больше фич).
_DEFAULT_CORR_THRESHOLD = 0.9

# Максимальное количество строк для расчёта корреляций по умолчанию.
# Значение 100_000 означает: если датасет больше, используется случайная выборка.
# Обоснование: 100K достаточно для стабильной оценки корреляции (стандартная ошибка < 0.01),
# при этом вычисление остаётся быстрым (~ 1-2 сек на современных CPU).
# Настройка: уменьшите до 50_000 для очень больших датасетов (> 1B rows),
# увеличьте до 200_000+ если корреляция нестабильна (много NaN в данных).
_DEFAULT_MAX_ROWS_FOR_CORRELATION = 100_000

AVAILABLE_TRANSFORMER_NAMES: list[str] = list(TRANSFORMERS)

# Способ указать колонки в feature_spec: имя колонки, объект polars.selectors
# (или любое выражение pl.Expr, резолвящееся в колонки), либо список/кортеж любой
# смеси того и другого ("bare"-значение оборачивается в список из одного элемента).
ColumnsArg = str | Sequence[Any]

# Второй элемент feature_spec — обязателен, автоматического пресета по умолчанию
# нет: либо ссылка на пресет (имя из ml_toolkit/transformers/presets/, или Path/
# полный путь к yaml), либо явный словарь {transformer_name: params} — можно
# перечислить только нужные трансформеры со своими параметрами. Пустой словарь
# {} — осознанный pass-through (колонка без наваренных фич).
PresetArg = Path | str | dict[str, dict]

# Один элемент feature_spec: (колонки, preset_or_transformers).
FeatureSpecEntry = tuple[ColumnsArg, PresetArg]


def _load_preset(preset: Path | str | dict | None) -> dict[str, dict]:
    """Загружает пресет параметров трансформеров.

    Args:
        preset: Одна из:
            - None → ml_toolkit/transformers/presets/monthly.yaml (по умолчанию)
            - Строка-имя (например, "descriptive") → ml_toolkit/transformers/presets/{имя}.yaml
            - Путь Path или str → точный путь к yaml-файлу
            - dict → готовый словарь параметров

    Returns:
        {feature_name: params_dict} для каждого трансформера в пресете.
    """
    if preset is None:
        preset = _DEFAULT_PRESET_PATH

    if isinstance(preset, dict):
        return {k: (v or {}) for k, v in preset.items()}

    if isinstance(preset, str):
        preset_path = Path(preset)
        # Если это просто имя без пути (нет /, \, и не существует как файл),
        # ищем в директории presets
        if not any(sep in preset for sep in ('/', '\\')) and not preset_path.exists():
            preset_path = Path(__file__).parent / "transformers" / "presets" / f"{preset}.yaml"
        preset = preset_path

    with open(preset, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {k: (v or {}) for k, v in raw.items()}


def _select_transformers(
    transformer_names: list[str] | None,
    preset: dict[str, dict],
) -> list[tuple[str, Any, dict]]:
    """Строит список активных трансформеров с их параметрами.

    Args:
        transformer_names: Явный список имён для запуска. None — все из пресета.
        preset: {feature_name: params_dict} — загруженный пресет.

    Returns:
        Список кортежей (feature_name, module, params) в порядке пресета.

    Raises:
        ValueError: Если в transformer_names встретилось неизвестное имя.
    """
    if transformer_names is not None:
        unknown = set(transformer_names) - set(AVAILABLE_TRANSFORMER_NAMES)
        if unknown:
            raise ValueError(
                f"Неизвестные трансформеры: {sorted(unknown)}. "
                f"Доступные: {AVAILABLE_TRANSFORMER_NAMES}"
            )

    result: list[tuple[str, Any, dict]] = []
    for name, params in preset.items():
        if name not in TRANSFORMERS:
            logger.warning("Пресет содержит неизвестный трансформер '%s', пропуск", name)
            continue
        if transformer_names is not None and name not in transformer_names:
            continue
        result.append((name, TRANSFORMERS[name], params))
    return result


def _resolve_columns(df: pl.DataFrame | pl.LazyFrame, columns: ColumnsArg) -> list[str]:
    """Резолвит один элемент `feature_spec.columns` в список имён колонок.

    `columns` — это либо имя колонки (`str`), либо "голый" селектор
    (`polars.selectors`/`pl.Expr`), либо список/кортеж любой смеси того и другого.
    Строковый атом берётся как имя колонки как есть. Атом-селектор резолвится через
    `df.select(atom).collect_schema().names()` — работает одинаково для `DataFrame`
    и `LazyFrame`, не материализует данные (только схему).

    Args:
        df: Датасет, схема которого используется для резолва селекторов.
        columns: Имя колонки, селектор, или список/кортеж из них.

    Returns:
        Имена колонок в порядке первого появления, без дублей.
    """
    atoms = [columns] if isinstance(columns, str) or not isinstance(columns, (list, tuple)) else list(columns)

    resolved: list[str] = []
    seen: set[str] = set()
    for atom in atoms:
        names = [atom] if isinstance(atom, str) else df.select(atom).collect_schema().names()
        for name in names:
            if name not in seen:
                seen.add(name)
                resolved.append(name)
    return resolved


def _transformers_from_preset(preset_dict: dict[str, dict]) -> list[tuple[str, Any, dict]]:
    """Строит список (feature_name, module, params) из уже загруженного словаря.

    В отличие от `_select_transformers`, здесь нет фильтрации по отдельному
    списку имён — `preset_dict` сам по себе является полным и окончательным
    списком того, что нужно навариться (ключи — это то, что запросил вызывающий,
    явно и без скрытых дефолтов).

    Args:
        preset_dict: {feature_name: params_dict}.

    Returns:
        Список кортежей (feature_name, module, params) в порядке preset_dict.

    Raises:
        ValueError: Если встретилось имя вне `TRANSFORMERS`.
    """
    result: list[tuple[str, Any, dict]] = []
    for name, params in preset_dict.items():
        if name not in TRANSFORMERS:
            raise ValueError(
                f"Неизвестный трансформер '{name}'. Доступные: {AVAILABLE_TRANSFORMER_NAMES}"
            )
        result.append((name, TRANSFORMERS[name], params or {}))
    return result


def _resolve_feature_spec(
    df: pl.DataFrame | pl.LazyFrame,
    feature_spec: list[FeatureSpecEntry],
) -> dict[str, list[tuple[str, Any, dict]]]:
    """Резолвит `feature_spec` в {product_col: список активных трансформеров}.

    Каждый элемент `feature_spec` — это `(columns, preset)`. `preset` обязателен
    и явен — автоматического пресета по умолчанию нет: если не задать его,
    поднимается `ValueError` (см. `Raises`). `preset` — это:
        - имя (`str`) или `Path`/полный путь к yaml-файлу — загружается через
          `_load_preset` и применяется **целиком** (все трансформеры из файла);
        - готовый словарь `{transformer_name: params}` — применяется как есть,
          можно перечислить только нужные трансформеры со своими параметрами;
          пустой словарь `{}` — осознанный pass-through (колонка без наваренных
          фич).

    Если несколько групп называют одну и ту же колонку, наборы трансформеров
    объединяются — совпадающая пара (колонка, трансформер) с одинаковыми
    параметрами вычисляется в Phase A ровно один раз. Если при этом две группы
    просят один и тот же (колонка, трансформер) с РАЗНЫМИ параметрами — это
    настоящий конфликт, не тихий дедуп: поднимается `ValueError`.

    Args:
        df: Датасет, схема которого используется для резолва селекторов в columns.
        feature_spec: Список пар (columns, preset).

    Returns:
        {product_col: [(feature_name, module, params), ...]} — порядок ключей
        и порядок трансформеров внутри колонки соответствует порядку первого
        появления в `feature_spec`.

    Raises:
        ValueError: Если feature_spec пуст, если элемент не является парой
            (columns, preset), если preset не задан (`None`), если в нём
            встретилось неизвестное имя трансформера, если после резолва
            найдены колонки вне схемы df, или если один и тот же (колонка,
            трансформер) запрошен с разными параметрами в разных группах.
    """
    if not feature_spec:
        raise ValueError("feature_spec пуст — нечего наваривать")

    # col -> {transformer_name: (module, params)} — сохраняет порядок первого появления
    requested: dict[str, dict[str, tuple[Any, dict]]] = {}
    for entry in feature_spec:
        if len(entry) != 2:
            raise ValueError(
                f"feature_spec: каждый элемент должен быть парой (columns, preset), "
                f"получено {entry!r}"
            )
        columns_arg, preset_arg = entry
        if preset_arg is None:
            raise ValueError(
                "feature_spec: пресет обязателен и должен быть задан явно — "
                "передайте имя/путь пресета или словарь {transformer_name: params}. "
                "Автоматического пресета по умолчанию нет."
            )
        preset_dict = _load_preset(preset_arg)

        cols = _resolve_columns(df, columns_arg)
        if not cols:
            logger.warning("feature_spec: %r не резолвится ни в одну колонку — пропуск", columns_arg)
            continue

        selected = _transformers_from_preset(preset_dict)
        for col in cols:
            bucket = requested.setdefault(col, {})
            for name, module, params in selected:
                if name in bucket and bucket[name][1] != params:
                    raise ValueError(
                        f"feature_spec: колонка '{col}', трансформер '{name}' "
                        f"запрошен с разными параметрами в разных группах "
                        f"({bucket[name][1]!r} vs {params!r})"
                    )
                bucket[name] = (module, params)

    schema_names = set(df.collect_schema().names())
    unknown = set(requested) - schema_names
    if unknown:
        raise ValueError(f"feature_spec ссылается на колонки, отсутствующие в df: {sorted(unknown)}")

    return {
        col: [(name, module, params) for name, (module, params) in bucket.items()]
        for col, bucket in requested.items()
    }


def _col_name(product_col: str, feature: str, suffix: str) -> str:
    return f"{product_col}__{feature}__{suffix}" if suffix else f"{product_col}__{feature}"


@contextlib.contextmanager
def _temp_working_dir(tmp_dir: Path | str | None):
    """Контекстный менеджер для рабочей директории временных parquet-файлов.

    Если tmp_dir is None: создаёт TemporaryDirectory (удаляется автоматически
    при выходе из контекста). Если tmp_dir задан: создаёт директорию и отдаёт
    её путь без последующей очистки (файлы остаются для отладки).
    """
    if tmp_dir is None:
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)
    else:
        p = Path(tmp_dir)
        p.mkdir(parents=True, exist_ok=True)
        yield p


def _pearson_excluding_both_zero(left: np.ndarray, right: np.ndarray) -> float:
    """Корреляция Пирсона, исключая наблюдения, где оба значения равны нулю.

    Нули-оба не считаются «согласием» по спецификации — они убираются из выборки
    перед расчётом. Если после фильтрации осталось меньше двух точек или один из
    рядов константен — возвращает 0.0.
    """
    mask = ~((left == 0.0) & (right == 0.0))
    if mask.sum() < 2:
        return 0.0
    l_flt, r_flt = left[mask], right[mask]
    if np.std(l_flt) == 0.0 or np.std(r_flt) == 0.0:
        return 0.0
    return float(np.corrcoef(l_flt, r_flt)[0, 1])


def _is_already_sorted(df: pl.DataFrame, entity_col: str, ts_col: str) -> bool:
    """Возвращает True, если entity_col без разрывов и ts_col возрастает внутри блоков.

    Один проход select: O(n) против O(n log n) сортировки.
    """
    stats = df.select([
        (pl.col(entity_col) != pl.col(entity_col).shift(1))
        .fill_null(True)
        .sum()
        .alias("n_runs"),
        pl.col(entity_col).n_unique().alias("n_unique"),
        (
            (pl.col(ts_col) < pl.col(ts_col).shift(1))
            & (pl.col(entity_col) == pl.col(entity_col).shift(1))
        )
        .fill_null(False)
        .any()
        .alias("has_ts_inversion"),
    ])
    return stats["n_runs"][0] == stats["n_unique"][0] and not stats["has_ts_inversion"][0]


def _generate_candidate_features_to_parquets(
    df: pl.DataFrame | Path,
    entity_col: str,
    ts_col: str,
    product_cols: list[str],
    product_col_transformers: dict[str, list[tuple[str, Any, dict]]],
    tmp_dir: Path,
    name: str,
) -> tuple[Path, int, list[str], dict[str, Path]]:
    """Генерирует признаки-кандидаты, сохраняя на диск по (product × трансформер).

    Сортирует df по (entity_col, ts_col), записывает базовые колонки в
    {tmp_dir}/{name}_base.parquet, затем для каждой пары (product-колонка,
    трансформер) сохраняет результат в
    {tmp_dir}/{name}_{product_col}__{transformer_name}.parquet.
    В памяти в каждый момент — массивы только одного трансформера. float32
    применяется в numpy до записи.

    Если df — Path к уже отсортированному parquet, весь датасет не грузится в
    eager-память: entity-колонка читается один раз, каждая product-колонка —
    по одной за итерацию через columnar pruning.

    Args:
        df: Eager-датасет или Path к уже отсортированному parquet.
            Если Path — сортировка не выполняется, колонки читаются поштучно.
        entity_col: Имя колонки-идентификатора сущности.
        ts_col: Имя колонки с датой конца месяца.
        product_cols: Имена product-колонок (= list(product_col_transformers)).
        product_col_transformers: {product_col: [(feature_name, module, params), ...]} —
            свой список трансформеров на каждую колонку (может быть пустым списком —
            тогда колонка попадёт только как сырая, без кандидатов).
        tmp_dir: Директория для временных parquet-файлов.
        name: Метка ('subset' / 'holding') для имён файлов и логов.

    Returns:
        Кортеж (base_parquet_path, n_rows, all_candidate_cols, col_to_parquet_path).
    """
    if isinstance(df, Path):
        # Данные уже отсортированы на диске — грузим только нужные колонки
        n_rows = pq.read_metadata(df).num_rows
        base_path = df  # parquet содержит ровно [entity_col, ts_col] + product_cols
        entity_codes = (
            pl.read_parquet(df, columns=[entity_col])[entity_col]
            .rle_id()
            .to_numpy()
            .astype(np.int64)
        )
    else:
        if _is_already_sorted(df, entity_col, ts_col):
            df_sorted = df
        else:
            logger.info("(%s) Данные не отсортированы — запускаем sort([%s, %s])", name, entity_col, ts_col)
            df_sorted = df.sort([entity_col, ts_col])

        base_path = tmp_dir / f"{name}_base.parquet"
        df_sorted.select([entity_col, ts_col] + product_cols).write_parquet(base_path, row_group_size=_ROW_GROUP_SIZE)
        n_rows = df_sorted.height

        entity_codes = (
            df_sorted.select(pl.col(entity_col).rle_id().alias("_ec"))["_ec"]
            .to_numpy()
            .astype(np.int64)
        )

    position_within_entity = compute_position_within_entity(entity_codes)
    del entity_codes

    all_candidate_cols: list[str] = []
    col_to_file: dict[str, Path] = {}

    n_files = sum(len(product_col_transformers[c]) for c in product_cols)
    logger.info(
        "Наварка кандидатов (%s): %d product-колонок, %d parquet-файлов кандидатов",
        name, len(product_cols), n_files,
    )

    for product_col in tqdm(product_cols, desc=f"Наварка фич ({name})", unit="product"):
        if isinstance(df, Path):
            product_values = pl.read_parquet(df, columns=[product_col])[product_col].to_numpy().astype(np.float64)
        else:
            product_values = df_sorted[product_col].to_numpy().astype(np.float64)

        for transformer_name, module, params in product_col_transformers[product_col]:
            arrays, suffixes = module.compute(product_values, position_within_entity, params)

            group_arrays: dict[str, np.ndarray] = {}
            for suffix, arr in zip(suffixes, arrays):
                col = _col_name(product_col, transformer_name, suffix)
                group_arrays[col] = np.asarray(arr, dtype=np.float32)
                all_candidate_cols.append(col)

            tmp_path = tmp_dir / f"{name}_{product_col}__{transformer_name}.parquet"
            pl.DataFrame(group_arrays).write_parquet(tmp_path, row_group_size=_ROW_GROUP_SIZE)
            for col in group_arrays:
                col_to_file[col] = tmp_path

            logger.debug(
                "(%s) '%s' / '%s': %d кандидатов → %s",
                name, product_col, transformer_name, len(suffixes), tmp_path.name,
            )

    logger.info(
        "(%s) Сохранено %d кандидатов фич в %d parquet-файлах",
        name, len(all_candidate_cols), n_files,
    )
    return base_path, n_rows, all_candidate_cols, col_to_file


def _streaming_correlation_filter(
    candidate_cols: list[str],
    col_to_file: dict[str, Path],
    n_rows: int,
    threshold: float,
    max_rows: int | None,
    random_seed: int = 0,
) -> list[str]:
    """Жадный корреляционный фильтр с построчной загрузкой кандидатов из parquet.

    Загружает каждый кандидат по одной колонке (columnar pruning parquet),
    сравнивает со всеми уже принятыми признаками, хранящимися в памяти. Пиковая
    нагрузка: n_accepted × n_sample_rows float32 + 1 кандидат — против
    n_candidates × n_rows в прежнем подходе.

    Нули-оба исключаются перед расчётом корреляции (см. _pearson_excluding_both_zero).
    Порядок кандидатов влияет на итог (жадный алгоритм).

    Args:
        candidate_cols: Имена кандидатов в порядке рассмотрения.
        col_to_file: {имя_колонки -> путь к parquet}.
        n_rows: Полное число строк датасета.
        threshold: Порог |r|, выше которого кандидат считается избыточным.
        max_rows: Максимум строк для расчёта корреляции. None — все строки.
        random_seed: Сид генератора для воспроизводимого семплирования.

    Returns:
        Подмножество candidate_cols (в исходном порядке), принятое фильтром.
    """
    sample_idx: np.ndarray | None = None
    if max_rows is not None and n_rows > max_rows:
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(n_rows, size=max_rows, replace=False)
        logger.info(
            "Корреляционный фильтр: подвыборка %d из %d строк (random_seed=%d)",
            max_rows, n_rows, random_seed,
        )

    logger.info(
        "Корреляционный фильтр: %d кандидатов, threshold=%.3f",
        len(candidate_cols), threshold,
    )

    accepted_cols: list[str] = []
    accepted_arrays: dict[str, np.ndarray] = {}
    n_dropped = 0

    for candidate_col in tqdm(candidate_cols, desc="Корреляционный фильтр", unit="фича"):
        candidate_arr = (
            pl.scan_parquet(col_to_file[candidate_col])
            .select(candidate_col)
            .collect()[candidate_col]
            .to_numpy()
        )
        if sample_idx is not None:
            candidate_arr = candidate_arr[sample_idx]

        is_redundant = False
        for accepted_col in accepted_cols:
            corr = _pearson_excluding_both_zero(candidate_arr, accepted_arrays[accepted_col])
            if abs(corr) > threshold:
                is_redundant = True
                logger.debug(
                    "Отброшен '%s': |r|=%.3f с принятым '%s'",
                    candidate_col, corr, accepted_col,
                )
                break

        if is_redundant:
            n_dropped += 1
        else:
            accepted_cols.append(candidate_col)
            accepted_arrays[candidate_col] = candidate_arr

    logger.info(
        "Корреляционный фильтр завершён: принято %d, отброшено %d из %d",
        len(accepted_cols), n_dropped, len(candidate_cols),
    )
    return accepted_cols


def _build_output_from_parquets(
    base_path: Path,
    entity_col: str,
    ts_col: str,
    product_cols: list[str],
    accepted_cols: list[str],
    col_to_file: dict[str, Path],
    out_path: Path,
    min_ts: Any | None = None,
    max_ts: Any | None = None,
) -> None:
    """Записывает итоговый датасет в out_path, читая по одному row group за раз.

    Читает base_path и файлы признаков через pyarrow.ParquetFile.read_row_group —
    в памяти одновременно один срез (~_ROW_GROUP_SIZE строк × все принятые колонки)
    вместо полного датасета. Это возможно потому, что все временные parquet-файлы
    записаны с одинаковым row_group_size=_ROW_GROUP_SIZE, поэтому row group i в
    base соответствует row group i в каждом файле фич.

    Файлы признаков открываются по одному внутри цикла и сразу освобождаются:
    пиковое число открытых дескрипторов равно 2 (base + текущий файл фич).

    Args:
        base_path: Parquet с [entity_col, ts_col] + product_cols, отсортированный
            по (entity_col, ts_col), записанный с row_group_size=_ROW_GROUP_SIZE.
        entity_col: Имя колонки-идентификатора.
        ts_col: Имя колонки с датой конца месяца.
        product_cols: Включаются в выход рядом с фичами (приводятся к float32).
        accepted_cols: Принятые фичи в нужном порядке (уже float32 в parquet).
        col_to_file: {имя_колонки -> путь к parquet}.
        out_path: Путь для записи результата.
        min_ts: Нижняя граница по ts_col (включительно). None — без ограничения.
        max_ts: Верхняя граница по ts_col (включительно). None — без ограничения.
    """
    file_to_cols: dict[Path, list[str]] = {}
    for col in accepted_cols:
        file_to_cols.setdefault(col_to_file[col], []).append(col)

    base_file = pq.ParquetFile(base_path)
    n_row_groups = base_file.metadata.num_row_groups
    base_schema = base_file.schema_arrow
    ts_type = base_schema.field(ts_col).type

    out_schema = pa.schema(
        [base_schema.field(entity_col), base_schema.field(ts_col)]
        + [pa.field(c, pa.float32()) for c in product_cols]
        + [pa.field(c, pa.float32()) for c in accepted_cols]
    )

    with pq.ParquetWriter(out_path, schema=out_schema) as writer:
        for rg_idx in range(n_row_groups):
            base_batch = base_file.read_row_group(rg_idx, columns=[entity_col, ts_col] + product_cols)

            row_mask: pa.ChunkedArray | None = None
            if min_ts is not None or max_ts is not None:
                ts_arr = base_batch.column(ts_col)
                mask_parts = []
                if min_ts is not None:
                    mask_parts.append(pc.greater_equal(ts_arr, pa.scalar(min_ts, type=ts_type)))
                if max_ts is not None:
                    mask_parts.append(pc.less_equal(ts_arr, pa.scalar(max_ts, type=ts_type)))
                row_mask = mask_parts[0] if len(mask_parts) == 1 else pc.and_(mask_parts[0], mask_parts[1])
                base_batch = base_batch.filter(row_mask)

            if len(base_batch) == 0:
                continue

            chunk: dict[str, pa.ChunkedArray] = {
                entity_col: base_batch.column(entity_col),
                ts_col: base_batch.column(ts_col),
            }
            for col in product_cols:
                chunk[col] = pc.cast(base_batch.column(col), pa.float32())
            del base_batch

            for fpath, cols in file_to_cols.items():
                # Открываем файл, читаем один row group, файл закрывается сразу
                # (CPython: рефкаунт объекта ParquetFile упадёт до 0 после del)
                feat_batch = pq.ParquetFile(fpath).read_row_group(rg_idx, columns=cols)
                for col in cols:
                    arr = feat_batch.column(col)
                    chunk[col] = arr.filter(row_mask) if row_mask is not None else arr
                del feat_batch

            ordered = [entity_col, ts_col] + product_cols + accepted_cols
            writer.write_table(pa.table({col: chunk[col] for col in ordered}))
            del chunk


def generate_feature_groups(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    feature_spec: list[FeatureSpecEntry],
    out_path: Path | str,
    corr_threshold: float | None = None,
    min_output_ts_key: Any | None = None,
    max_output_ts_key: Any | None = None,
    max_rows_for_correlation: int | None = _DEFAULT_MAX_ROWS_FOR_CORRELATION,
    tmp_dir: Path | str | None = None,
    name: str = "dataset",
) -> list[str]:
    """Наваривает фичи по группам «колонки → пресет» и пишет на диск.

    Обобщение `select_features` на случай, когда разным product-колонкам нужны
    разные трансформеры и/или разные параметры (не единый набор для всех).
    Каждый элемент `feature_spec` — пара `(columns, preset)`:
        - `columns`: имя колонки (`str`), объект `polars.selectors`/`pl.Expr`, или
          список/кортеж любой их смеси — резолвится в конкретные имена колонок
          через схему `df` (без материализации данных).
        - `preset`: **обязателен**, автоматического пресета по умолчанию нет —
          это либо имя (`str`)/`Path`/полный путь к yaml-файлу (загружается через
          `_load_preset` и применяется целиком, все трансформеры из файла), либо
          готовый словарь `{transformer_name: params}` — можно перечислить
          только нужные трансформеры со своими параметрами. Пустой словарь `{}` —
          осознанный pass-through (колонка попадёт в выход как сырая, без
          наваренных фич).

    Если несколько элементов `feature_spec` называют одну и ту же колонку (в том
    числе через разные селекторы), наборы трансформеров объединяются, и совпадающая
    пара (колонка, трансформер) с одинаковыми параметрами вычисляется в Phase A
    ровно один раз — повторной наварки не происходит. Если при этом две группы
    просят один и тот же (колонка, трансформер) с разными параметрами — это
    конфликт, поднимается `ValueError` (см. `Raises`).

    В отличие от `select_features`, корреляционный фильтр (Phase B) по умолчанию
    **не запускается** (`corr_threshold=None`) — раз колонки и трансформеры выбраны
    явно, автоматическое отбрасывание «избыточных» фич могло бы молча отменить это
    явное решение. Передайте `corr_threshold`, чтобы включить тот же фильтр, что и
    в `select_features`.

    Терминальная для `df` функция: результат уходит в `out_path`, ничего не
    материализуется в Python-памяти целиком (Phase C стримит через pyarrow row
    group за row group). Возвращает список итоговых колонок (`accepted_cols`, если
    фильтр запускался, иначе — все кандидаты), чтобы можно было навариать тот же
    набор для другого датасета той же схемы через `apply_feature_groups` (проще
    всего — передать туда тот же самый `feature_spec`, тогда пресеты каждой группы
    автоматически совпадут между парными вызовами).

    Args:
        df: Датасет (eager или lazy) на уровне (entity_column_name,
            ts_column_name) с product-колонками.
        entity_column_name: Имя колонки-идентификатора сущности (клиент,
            холдинг, магазин — что угодно однородное внутри `df`).
        ts_column_name: Имя колонки с датой конца месяца.
        feature_spec: Список пар `(columns, preset)` — см. выше.
        out_path: Путь для итогового parquet-файла.
        corr_threshold: Порог |r| для корреляционного фильтра. По умолчанию
            `None` — фильтр не запускается, принимаются все наваренные кандидаты.
            Передайте значение (например, 0.9), чтобы включить фильтр.
        min_output_ts_key: Нижняя граница по ts_column_name (включительно).
            Применяется при записи — не обрезает историю для расчёта окон.
        max_output_ts_key: Верхняя граница по ts_column_name (включительно).
        max_rows_for_correlation: Максимум строк для расчёта корреляции (только
            если `corr_threshold` задан). Default: 100_000.
        tmp_dir: Путь для временных parquet-файлов с кандидатами фич.
            None → системная temp-директория (удаляется автоматически по выходу).
        name: Метка для временных файлов и логов (например, имя датасета в
            вызывающей задаче).

    Returns:
        Список итоговых колонок в порядке наварки (или отбора, если фильтр
        запускался).

    Raises:
        ValueError: Если `feature_spec` пуст, если элемент не является парой
            (columns, preset), если preset не задан (`None`), если в нём
            встретилось неизвестное имя трансформера, если резолв колонок дал
            имя вне схемы `df`, или если один и тот же (колонка, трансформер)
            запрошен с разными параметрами в разных группах `feature_spec`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    product_col_transformers = _resolve_feature_spec(df, feature_spec)
    product_cols = list(product_col_transformers)
    logger.info(
        "generate_feature_groups (%s): %d product-колонок, %d кандидатов трансформеров, threshold=%s",
        name, len(product_cols), sum(len(v) for v in product_col_transformers.values()), corr_threshold,
    )

    with _temp_working_dir(tmp_dir) as work_dir:
        if isinstance(df, pl.LazyFrame):
            _sorted_tmp = work_dir / f"_{name}.parquet"
            df.sort([entity_column_name, ts_column_name]).sink_parquet(_sorted_tmp, row_group_size=_ROW_GROUP_SIZE)
            # Передаём Path вместо чтения всего файла в память — Phase A читает
            # колонки поштучно через columnar pruning parquet.
            df = _sorted_tmp

        # Phase A: генерируем кандидатов — один parquet на (product × transformer)
        base_path, n_rows, candidate_cols, col_to_file = _generate_candidate_features_to_parquets(
            df=df,
            entity_col=entity_column_name,
            ts_col=ts_column_name,
            product_cols=product_cols,
            product_col_transformers=product_col_transformers,
            tmp_dir=work_dir,
            name=name,
        )
        del df

        # Phase B: стриминговый корреляционный фильтр (опционален в групповом режиме)
        if corr_threshold is None:
            accepted_cols = candidate_cols
            logger.info(
                "Корреляционный фильтр не запускается (corr_threshold=None): принято %d кандидатов",
                len(accepted_cols),
            )
        else:
            accepted_cols = _streaming_correlation_filter(
                candidate_cols=candidate_cols,
                col_to_file=col_to_file,
                n_rows=n_rows,
                threshold=corr_threshold,
                max_rows=max_rows_for_correlation,
            )
            logger.info(
                "Корреляционный фильтр: принято %d из %d (threshold=%.3f)",
                len(accepted_cols), len(candidate_cols), corr_threshold,
            )

        # Phase C: запись row group за row group через pyarrow
        _build_output_from_parquets(
            base_path=base_path,
            entity_col=entity_column_name,
            ts_col=ts_column_name,
            product_cols=product_cols,
            accepted_cols=accepted_cols,
            col_to_file=col_to_file,
            out_path=out_path,
            min_ts=min_output_ts_key,
            max_ts=max_output_ts_key,
        )
        logger.info("(%s) записан: %s", name, out_path)

    return accepted_cols


def select_features(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    product_cols: list[str],
    out_path: Path | str,
    corr_threshold: float | None = _DEFAULT_CORR_THRESHOLD,
    transformer_names: list[str] | None = None,
    min_output_ts_key: Any | None = None,
    max_output_ts_key: Any | None = None,
    max_rows_for_correlation: int | None = _DEFAULT_MAX_ROWS_FOR_CORRELATION,
    tmp_dir: Path | str | None = None,
    preset: Path | str | dict | None = None,
    name: str = "dataset",
) -> list[str]:
    """Наваривает фичи для `df`, отбирает их корреляционным фильтром и пишет на диск.

    Частный случай `generate_feature_groups` — один и тот же набор трансформеров
    применяется ко всем `product_cols` сразу: `transformer_names`+`preset` здесь
    заранее резолвятся в явный словарь `{transformer_name: params}` и передаются
    в `generate_feature_groups` одной группой (`feature_spec=[(product_cols,
    resolved)]`), а корреляционный фильтр включён по умолчанию (`corr_threshold=0.9`).
    Если разным колонкам нужны разные трансформеры, используйте
    `generate_feature_groups` напрямую.

    Терминальная для `df` функция: результат уходит в `out_path`, ничего не
    материализуется в Python-памяти целиком (Phase C стримит через pyarrow row
    group за row group). Возвращает `accepted_cols` — список принятых колонок,
    чтобы можно было навариать тот же набор для другого датасета той же схемы
    через `apply_selected_features` (без повторного запуска фильтра).

    Args:
        df: Датасет (eager или lazy) на уровне (entity_column_name,
            ts_column_name) с product-колонками.
        entity_column_name: Имя колонки-идентификатора сущности (клиент,
            холдинг, магазин — что угодно однородное внутри `df`).
        ts_column_name: Имя колонки с датой конца месяца.
        product_cols: Имена product-колонок, по которым считаются признаки.
        out_path: Путь для итогового parquet-файла.
        corr_threshold: Порог |r| для корреляционного фильтра. Default: 0.9.
            Передайте None, чтобы отключить фильтр и принять все кандидаты.
            Значения 0.85-0.95 оптимальны для баланса между размером модели и качеством.
        transformer_names: Подмножество AVAILABLE_TRANSFORMER_NAMES. None — все
            трансформеры из пресета.
        min_output_ts_key: Нижняя граница по ts_column_name (включительно).
            Применяется при записи — не обрезает историю для расчёта окон.
        max_output_ts_key: Верхняя граница по ts_column_name (включительно).
        max_rows_for_correlation: Максимум строк для расчёта корреляции.
            Default: 100_000. Если датасет больше, используется случайная выборка.
            Снизьте для очень больших датасетов (> 1B rows), увеличьте для нестабильных корреляций.
        tmp_dir: Путь для временных parquet-файлов с кандидатами фич.
            None → системная temp-директория (удаляется автоматически по выходу).
            Если задан — директория создаётся и файлы остаются после завершения
            (удобно для отладки).
        preset: Пресет параметров трансформеров — одно из:
            - None → monthly.yaml (по умолчанию)
            - "descriptive" / "trend" / "stability" / "lifecycle" → именованный пресет
            - Path / str с полным путём → точный yaml-файл
            - dict → готовый словарь {feature_name: params}
        name: Метка для временных файлов и логов (например, имя датасета в
            вызывающей задаче).

    Returns:
        Список принятых колонок (`accepted_cols`) в порядке отбора.

    Raises:
        ValueError: Если в transformer_names встретилось неизвестное имя.
    """
    preset_dict = _load_preset(preset)
    resolved = {n: p for n, _, p in _select_transformers(transformer_names, preset_dict)}

    return generate_feature_groups(
        df,
        entity_column_name=entity_column_name,
        ts_column_name=ts_column_name,
        feature_spec=[(product_cols, resolved)],
        out_path=out_path,
        corr_threshold=corr_threshold,
        min_output_ts_key=min_output_ts_key,
        max_output_ts_key=max_output_ts_key,
        max_rows_for_correlation=max_rows_for_correlation,
        tmp_dir=tmp_dir,
        name=name,
    )


def apply_feature_groups(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    feature_spec: list[FeatureSpecEntry],
    accepted_cols: list[str],
    out_path: Path | str,
    min_output_ts_key: Any | None = None,
    max_output_ts_key: Any | None = None,
    tmp_dir: Path | str | None = None,
    name: str = "dataset",
) -> None:
    """Наваривает для `df` только уже отобранные `accepted_cols` и пишет на диск.

    Групповой аналог `apply_selected_features` — парная функция к
    `generate_feature_groups`. `feature_spec` здесь должен описывать (как
    надмножество) те же группы «колонки → preset», что и вызов
    `generate_feature_groups`, который вернул `accepted_cols` — иначе часть
    `accepted_cols` не будет наварена и функция упадёт с `KeyError`. Проще всего
    передать сюда тот же самый `feature_spec`, что и в `generate_feature_groups` —
    тогда пресеты каждой группы автоматически совпадут, без риска разъехаться
    между парными вызовами. Не запускает корреляционный фильтр — это не его роль,
    фильтрация уже отражена в `accepted_cols`.

    Терминальная для `df` функция и по устройству симметрична
    `generate_feature_groups`: Phase A наваривает ВСЕ кандидатские фичи
    `feature_spec` на диск (`_generate_candidate_features_to_parquets`, стриминг
    column-at-a-time — ни весь `df`, ни весь набор кандидатов не оказываются в
    Python-памяти одновременно), Phase B (корреляционный фильтр) не запускается —
    вместо него сразу используется готовый `accepted_cols`, Phase C
    (`_build_output_from_parquets`) стримит результат в `out_path` row group за
    row group. `pl.LazyFrame` на входе никогда не коллектится целиком: сортируется
    и пишется в temp-parquet через `sink_parquet` — ровно тот же путь, что и в
    `generate_feature_groups`.

    Args:
        df: Датасет (eager или lazy) на уровне (entity_column_name,
            ts_column_name) с product-колонками.
        entity_column_name: Имя колонки-идентификатора сущности в `df`.
        ts_column_name: Имя колонки с датой конца месяца.
        feature_spec: Список пар `(columns, preset)` — см. `generate_feature_groups`.
            Должен покрывать все `accepted_cols`.
        accepted_cols: Список принятых фич из `generate_feature_groups` —
            определяет схему выходного датасета.
        out_path: Путь для итогового parquet-файла.
        min_output_ts_key: Нижняя граница по ts_column_name (включительно).
        max_output_ts_key: Верхняя граница по ts_column_name (включительно).
        tmp_dir: Путь для временных parquet-файлов с кандидатами фич.
            None → системная temp-директория (удаляется автоматически по выходу).
        name: Метка для логов/tqdm (например, имя датасета в вызывающей задаче).

    Raises:
        ValueError: Если `feature_spec` пуст, если элемент не является парой
            (columns, preset), если preset не задан (`None`), если в нём
            встретилось неизвестное имя трансформера, если резолв колонок дал
            имя вне схемы `df`, или если один и тот же (колонка, трансформер)
            запрошен с разными параметрами в разных группах `feature_spec`.
        KeyError: Если `feature_spec` не покрывает какую-то колонку из
            `accepted_cols` (см. описание выше).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    product_col_transformers = _resolve_feature_spec(df, feature_spec)
    product_cols = list(product_col_transformers)

    with _temp_working_dir(tmp_dir) as work_dir:
        if isinstance(df, pl.LazyFrame):
            _sorted_tmp = work_dir / f"_{name}.parquet"
            df.sort([entity_column_name, ts_column_name]).sink_parquet(_sorted_tmp, row_group_size=_ROW_GROUP_SIZE)
            df = _sorted_tmp

        base_path, _n_rows, _candidate_cols, col_to_file = _generate_candidate_features_to_parquets(
            df=df,
            entity_col=entity_column_name,
            ts_col=ts_column_name,
            product_cols=product_cols,
            product_col_transformers=product_col_transformers,
            tmp_dir=work_dir,
            name=name,
        )
        del df

        _build_output_from_parquets(
            base_path=base_path,
            entity_col=entity_column_name,
            ts_col=ts_column_name,
            product_cols=product_cols,
            accepted_cols=accepted_cols,
            col_to_file=col_to_file,
            out_path=out_path,
            min_ts=min_output_ts_key,
            max_ts=max_output_ts_key,
        )
        logger.info("(%s) записан: %s", name, out_path)


def apply_selected_features(
    df: pl.DataFrame | pl.LazyFrame,
    entity_column_name: str,
    ts_column_name: str,
    product_cols: list[str],
    accepted_cols: list[str],
    out_path: Path | str,
    min_output_ts_key: Any | None = None,
    max_output_ts_key: Any | None = None,
    transformer_names: list[str] | None = None,
    preset: Path | str | dict | None = None,
    tmp_dir: Path | str | None = None,
    name: str = "dataset",
) -> None:
    """Наваривает для `df` только уже отобранные `accepted_cols` и пишет на диск.

    Частный случай `apply_feature_groups` — тот же (уникформный) набор
    трансформеров, что и в парном `select_features`, применяется ко всем
    `product_cols` сразу. Не запускает корреляционный фильтр повторно —
    используется, когда набор фич уже выбран по другому датасету (см.
    `select_features`) и нужно применить тот же набор к `df`, чтобы оба датасета
    получили идентичную схему.

    Args:
        df: Датасет (eager или lazy) на уровне (entity_column_name,
            ts_column_name) с product-колонками.
        entity_column_name: Имя колонки-идентификатора сущности в `df`.
        ts_column_name: Имя колонки с датой конца месяца.
        product_cols: Имена product-колонок, по которым считаются признаки.
        accepted_cols: Список принятых фич из `select_features` — определяет
            схему выходного датасета.
        out_path: Путь для итогового parquet-файла.
        min_output_ts_key: Нижняя граница по ts_column_name (включительно).
        max_output_ts_key: Верхняя граница по ts_column_name (включительно).
        transformer_names: Подмножество AVAILABLE_TRANSFORMER_NAMES. Должно
            совпадать с тем, что передавалось в парный `select_features`, иначе
            параметры трансформеров разойдутся между датасетами.
        preset: Пресет параметров трансформеров (см. `select_features`). Должен
            совпадать с тем, что передавалось в парный `select_features`.
        tmp_dir: Путь для временных parquet-файлов с кандидатами фич.
            None → системная temp-директория (удаляется автоматически по выходу).
        name: Метка для логов/tqdm (например, имя датасета в вызывающей задаче).

    Raises:
        ValueError: Если в transformer_names встретилось неизвестное имя.
    """
    preset_dict = _load_preset(preset)
    resolved = {n: p for n, _, p in _select_transformers(transformer_names, preset_dict)}

    apply_feature_groups(
        df,
        entity_column_name=entity_column_name,
        ts_column_name=ts_column_name,
        feature_spec=[(product_cols, resolved)],
        accepted_cols=accepted_cols,
        out_path=out_path,
        min_output_ts_key=min_output_ts_key,
        max_output_ts_key=max_output_ts_key,
        tmp_dir=tmp_dir,
        name=name,
    )
