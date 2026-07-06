# ml_toolkit/feature_generation.py

Генерический движок наварки фич для **одного** датасета за раз. Не хардкодит имена колонок и не знает, сколько у вас датасетов и как они связаны (это уровень `tasks/`, см. `tasks/auto_kkp_classification/feature_generation.py`).

---

## У вас уже есть датасет — как навариться нужные фичи для нужных колонок

Датасет может быть **уже в памяти** (`pl.DataFrame`) или **лежать на диске** (`pl.LazyFrame`, например `pl.scan_parquet(...)`) — движок принимает и то, и другое одной и той же сигнатурой, разницы в вызове нет:

```python
import polars as pl
from ml_toolkit.feature_generation import generate_feature_groups

df = pl.scan_parquet("cltv_df.parquet")   # на диске — файл целиком в память не читается
# df = pl.read_parquet("cltv_df.parquet") # или уже в памяти как DataFrame — работает так же
# df = уже готовый pl.DataFrame из предыдущих шагов пайплайна — тоже ОК
```

Если `df` — `LazyFrame`, движок сам сортирует по `(entity_column_name, ts_column_name)` и стримит результат во временный parquet (`sink_parquet`) — Python-память не раздувается под весь датасет разом, даже если он большой.

Дальше — **`feature_spec`**: список пар `(колонки, что навариться)`. Именно он отвечает на вопрос "каким колонкам какие фичи навариваем":

```python
result_cols = generate_feature_groups(
    df,
    entity_column_name="id_key",       # колонка-идентификатор сущности (клиент/холдинг/...)
    ts_column_name="ts_key",           # колонка с датой конца месяца
    feature_spec=[
        ("trans_sum", {"slope": {"windows": [6, 12, 24]}, "streak": {}}),  # этой — slope и streak
        ("trans_cnt", {"ewma": {"alphas": [0.3]}}),                        # этой — только ewma
        ("fee_amount", {}),                                                # эту оставить как есть
    ],
    out_path="cltv_df_with_features.parquet",
)
```

Второй элемент каждой пары — **обязателен** и всегда явный: либо словарь `{имя_трансформера: параметры}` (можно перечислить только нужные трансформеры со своими параметрами — как выше), либо ссылка на пресет из файла (см. раздел про пресеты ниже). Автоматического пресета "по умолчанию" нет: если не задать второй элемент — это ошибка (`ValueError`), а не тихий fallback на какие-то параметры.

`out_path` получит: `id_key`, `ts_key`, все три raw-колонки (`trans_sum`, `trans_cnt`, `fee_amount`) плюс наваренные признаки — `trans_sum__slope__w6`, `trans_sum__streak__up`, ... `trans_cnt__ewma__...`. Для `fee_amount` признаков не будет — только сама колонка (осознанный pass-through через `{}`). `result_cols` — список наваренных имён (пригодится, если нужно применить тот же набор ко второму датасету, см. ниже).

### Одни и те же трансформеры для нескольких колонок сразу

Не обязательно писать по одной паре на колонку — можно сгруппировать:

```python
feature_spec = [
    (["trans_sum", "trans_cnt"], {"slope": {"windows": [6, 12, 24]}, "rolling_std": {"windows": [6, 12]}}),
    ("fee_amount", {"ewma": {"alphas": [0.3]}}),
]
```

### Указать колонки через `polars.selectors` (если колонок много/имена не фиксированы)

Вместо перечисления имён строками можно передать селектор — он резолвится против реальной схемы `df` (без чтения данных, даже если `df` — `LazyFrame` на диске):

```python
import polars.selectors as cs

SLOPE_AND_STREAK = {"slope": {"windows": [6, 12, 24]}, "streak": {}}

feature_spec = [
    (cs.starts_with("trans_"), SLOPE_AND_STREAK),                                  # все колонки "trans_*"
    (cs.starts_with("cnt_"), {"rolling_std": {"windows": [6, 12]}}),                # все "cnt_*"
    ([cs.starts_with("a_"), cs.ends_with("_pct")], {"zscore": {"windows": [12]}}),  # смесь нескольких селекторов
]
```

Если разные строки/селекторы в итоге называют одну и ту же колонку с одним и тем же трансформером **и одинаковыми параметрами** (например, явное имя пересекается с широким селектором) — он считается **ровно один раз**, повторной наварки не будет. Если параметры при этом различаются — это конфликт, см. ниже.

### Какие трансформеры доступны

```python
from ml_toolkit.feature_generation import AVAILABLE_TRANSFORMER_NAMES
```

Ключи словаря `{имя_трансформера: параметры}` должны быть именами из `AVAILABLE_TRANSFORMER_NAMES` (список — в разделе "Реестр трансформеров" ниже). Неизвестное имя — `ValueError`.

### Пресеты — способ не писать параметры вручную каждый раз

Если параметры трансформера уже есть в одном из готовых yaml-файлов репозитория (`ml_toolkit/transformers/presets/`) — не обязательно переписывать их в inline-словарь, можно сослаться на пресет по имени или пути. Второй элемент пары в `feature_spec` — это:

- `dict[str, dict]` — явный словарь `{transformer_name: params}`, как в примерах выше;
- `str` — имя пресета (например, `"descriptive"`, `"monthly"`), ищется в `ml_toolkit/transformers/presets/{имя}.yaml`, или полный путь строкой;
- `Path` — точный путь к yaml-файлу.

Пресет из файла применяется **целиком** — все трансформеры, перечисленные в нём:

```python
result_cols = generate_feature_groups(
    df, entity_column_name="id_key", ts_column_name="ts_key",
    feature_spec=[
        ("trans_sum", "monthly"),        # все трансформеры из monthly.yaml
        ("trans_cnt", {"ewma": {"alphas": [0.3]}}),  # а для этой колонки — только ewma вручную
    ],
    out_path="out.parquet",
)
```

Вот как выглядит запись `slope` в `ml_toolkit/transformers/presets/monthly.yaml`:

```yaml
slope:
  windows: [6, 12, 24]
```

Если нужен весь пресет, но с одним изменённым значением — загрузите yaml, поменяйте нужную запись и передайте получившийся словарь как обычный inline-словарь:

```python
import yaml
from pathlib import Path

preset = yaml.safe_load(Path("ml_toolkit/transformers/presets/monthly.yaml").read_text())
preset["slope"]["windows"] = [3, 6, 12]   # свои окна вместо [6, 12, 24]

result_cols = generate_feature_groups(
    df, entity_column_name="id_key", ts_column_name="ts_key",
    feature_spec=[("trans_sum", preset)],   # весь monthly.yaml, но с изменённым slope
    out_path="out.parquet",
)
```

Какие ключи принимает конкретный трансформер — смотрите в его модуле `ml_toolkit/transformers/kernels/{name}.py` (докстринг там всегда содержит раздел `Preset` с примером) или в самом `monthly.yaml`. Трансформеры без параметров (например, `streak`, `growth_since_start`) — это просто `{}`.

Если разные группы `feature_spec` называют одну и ту же колонку с одним и тем же трансформером, но **разными** параметрами — это конфликт: `ValueError` вместо тихого выбора одного из вариантов.

---

## Второй датасет с той же схемой фич (например, другая гранулярность)

Если у вас два датасета одной "формы" (одинаковые product-колонки, но разная сущность — клиент vs холдинг, магазин vs регион) и нужно, чтобы оба получили **идентичный** набор колонок — навариваете первый через `generate_feature_groups`, второй — через `apply_feature_groups` с тем же `feature_spec` и `result_cols`:

```python
from ml_toolkit.feature_generation import apply_feature_groups

apply_feature_groups(
    other_df,                          # тот же feature_spec, другой df/другая entity-колонка
    entity_column_name="agreement_primary_key",
    ts_column_name="ts_key",
    feature_spec=feature_spec,          # тот же список, что и в generate_feature_groups
    accepted_cols=result_cols,          # что навариться — из первого вызова
    out_path="other_df_with_features.parquet",
)
```

`feature_spec` здесь должен покрывать (как надмножество) все `result_cols` — иначе часть колонок не будет наварена (`KeyError`). Проще всего передавать один и тот же `feature_spec`, что и в первом вызове — тогда параметры каждой группы автоматически совпадают, без риска разъехаться между парными вызовами.

---

## Нужен автоматический отбор фич (корреляционный фильтр)?

По умолчанию `generate_feature_groups` навариваете именно то, что попросили в `feature_spec`, — ничего не отбрасывается. Если хотите дополнительно прогнать жадный корреляционный фильтр (отбросить кандидатов, которые почти дублируют друг друга), передайте `corr_threshold`:

```python
result_cols = generate_feature_groups(
    df, entity_column_name="id_key", ts_column_name="ts_key",
    feature_spec=feature_spec, out_path="out.parquet",
    corr_threshold=0.9,   # |r| > 0.9 → кандидат отбрасывается
)
```

Фильтр — жадный Пирсон, читает кандидатов из временных parquet по одной колонке за раз (не грузит весь датасет фич разом), нули-в-обоих исключаются перед расчётом корреляции. `max_rows_for_correlation` (default 100 000) ограничивает число строк для расчёта на больших датасетах.

### Частный случай: один и тот же набор трансформеров на все колонки

Если разных наборов не нужно и вы хотите **всегда** прогонять корреляционный фильтр (включён по умолчанию, `corr_threshold=0.9`) — есть более короткая uniform-обёртка `select_features`/`apply_selected_features`. В отличие от группового API, здесь `transformer_names`/`preset` — обычные необязательные параметры с понятным дефолтом (весь `monthly.yaml`), это исторически сложившийся более простой интерфейс:

```python
from ml_toolkit.feature_generation import select_features, apply_selected_features

accepted_cols = select_features(
    df,
    entity_column_name="id_key",
    ts_column_name="ts_key",
    product_cols=["trans_sum", "trans_cnt"],   # одни и те же трансформеры на обе колонки
    out_path="out.parquet",
    transformer_names=["slope", "ewma"],        # None — все трансформеры пресета
    preset=None,                                 # None — monthly.yaml
)
```

Внутри это тонкая обёртка: `transformer_names`+`preset` резолвятся в явный словарь `{transformer_name: params}` и передаются в `generate_feature_groups(feature_spec=[(product_cols, resolved)], corr_threshold=0.9, ...)`.

---

## Параметры, общие для всех четырёх функций

| Параметр | Смысл |
|---|---|
| `entity_column_name` | Колонка-идентификатор сущности (клиент, холдинг — что угодно однородное внутри `df`). |
| `ts_column_name` | Колонка с датой конца месяца. |
| `out_path` | Путь итогового parquet. |
| `min_output_ts_key` / `max_output_ts_key` | Границы по `ts_column_name` (включительно), применяются **после** наварки — не обрезают историю, на которой считаются окна. |
| `tmp_dir` | Только у `select_features`/`generate_feature_groups` — папка для временных parquet с кандидатами. `None` → системная temp (авто-удаление). Задайте явно, чтобы файлы остались для отладки. |
| `name` | Метка для логов/tqdm — например, имя датасета в вызывающей задаче ("subset" / "holding"). |

`preset`/параметры трансформеров — не общий параметр всех четырёх функций: в `select_features`/`apply_selected_features` это отдельные kwargs `transformer_names`+`preset` с дефолтом на весь `monthly.yaml`; в `generate_feature_groups`/`apply_feature_groups` это второй, обязательный элемент каждой пары в `feature_spec` (без дефолта — см. выше).

---

## Реестр трансформеров

81 трансформер, сгруппированных тематически (полный список — `ml_toolkit/transformers/__init__.py`, параметры каждого — `ml_toolkit/transformers/presets/monthly.yaml`):

- *trend*: `slope`, `slope_ratio`, `momentum`, `direction_flag`, `max_abs_jump`, `streak`, `growth_since_start`
- *volatility*: `rolling_std`, `rolling_cv`, `rolling_min_max`, `extreme_share`, `skew_proxy`
- *tenure/activity*: `active_months`, `active_run_count`, `activity_rate`, `client_age`, `inactive_streak`, `longest_active_run`, `recency`, `tenure`, `zero_share`
- *dynamics ratios*: `ewma`, `lag1_diff`, `mean_median_gap`, `recent_share`, `rolling_sum`, `zscore`
- *trend change*: `accel`, `max_drawdown`, `peak_trough_timing`, `run_above_mean`, `sign_change_count`, `time_weighted_momentum`, `trend_flip`, `volatility_of_diff`
- *relative position*: `cumulative_share`, `distance_to_global_max`, `half_ratio`, `lag_growth_ratio`, `level_ratio`, `local_extrema`, `log1p_level`, `pct_of_max`, `rank_in_window`, `trough_to_current`, `volatility_trend`
- *structural signals*: `corr_with_time`, `cusum`, `entropy`, `gini`
- *autocorr & seasonal*: `autocorr`, `seasonal_autocorr`
- *log growth*: `geometric_return`, `log_level`, `log_slope`, `log_slope_ratio`, `log_volatility`
- *smoothness*: `alternation_rate`, `roughness_ratio`, `total_variation`
- *distribution moments*: `kurtosis_proxy`
- *прочее*: `burstiness`, `cross_window_momentum`, `extreme_events`, `flow_regularity`, `growth_quality`, `lag_comparison`, `lifecycle_phase`, `mean_deviation_shape`, `microstructure`, `nonlinearity`, `plateau`, `quantile_persistence`, `recovery_dynamics`, `regime_change`, `trend_consistency`, `value_clustering`, `window_mean`, `window_median`, `window_volatility_ratios`, `zero_clustering`

Именование выходных колонок: `{product_col}__{feature}__{suffix}` (или без `__{suffix}`, если у трансформера нет суффиксов, например `growth_since_start`).

---

## Ошибки и edge cases

| Ситуация | Поведение |
|---|---|
| `feature_spec` пуст (`[]`) | `ValueError` |
| Элемент `feature_spec` — не пара `(columns, preset)` | `ValueError` |
| Второй элемент пары не задан (`None`) | `ValueError` — автоматического пресета по умолчанию нет |
| Неизвестное имя трансформера (ключ словаря) | `ValueError` со списком `AVAILABLE_TRANSFORMER_NAMES` |
| Строка/селектор в `feature_spec` резолвится в колонку вне схемы `df` | `ValueError` с именами отсутствующих колонок |
| Селектор ни во что не резолвится (например, датасет без опциональных колонок) | `logger.warning`, группа молча пропускается |
| `(columns, {})` | Осознанный pass-through — колонка в выходе, фич по ней нет |
| Одна и та же колонка + трансформер запрошены в разных группах `feature_spec` с разными параметрами | `ValueError` — конфликт, не тихий выбор одного из вариантов |
| `accepted_cols` в `apply_feature_groups` содержит колонку, не покрытую `feature_spec` | `KeyError` — `feature_spec` должен быть надмножеством |

---

## Как это устроено под капотом (кратко)

1. **Наварка кандидатов** — для каждой пары (колонка, трансформер) признаки считаются и сразу пишутся во временный parquet (не держим весь широкий датасет фич в памяти разом).
2. **Корреляционный фильтр** (только если задан `corr_threshold`) — жадный Пирсон, читает кандидатов по одной колонке за раз.
3. **Сборка выхода** — только нужные колонки читаются из временных parquet, по одному row group за раз, и пишутся в `out_path`.

`apply_feature_groups`/`apply_selected_features` (второй датасет) идут другим, более коротким путём: считают только то, что нужно для покрытия `accepted_cols`, без промежуточных parquet-файлов.
