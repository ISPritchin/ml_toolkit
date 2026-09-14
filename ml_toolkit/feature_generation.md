# ml_toolkit/feature_generation.py

Считает признаки по временным рядам датасета и докладывает их прямо в `df`. Лишние колонки (таргет, метаданные — всё, чего нет в `feature_spec`) остаются в выходе как есть, join не нужен.

---

## Датасет в памяти → тот же датасет с фичами

```python
import polars as pl
from ml_toolkit.feature_generation import generate_feature_groups_df

df = pl.DataFrame({
    "client_id": [1, 1, 1, 2, 2, 2],
    "ts_key":    [1, 2, 3, 1, 2, 3],
    "trans_sum": [10., 20., 15., 5., 5., 5.],
    "trans_cnt": [1., 2., 1., 1., 1., 1.],
    "region":    ["msk", "msk", "msk", "spb", "spb", "spb"],
    "target":    [0, 1, 0, 0, 0, 1],
})

out = generate_feature_groups_df(
    df, entity_column_name="client_id", ts_column_name="ts_key",
    feature_spec=[("trans_sum", {"slope": {"windows": [3]}, "streak": {}})],
)
print(out)
```

```text
shape: (6, 9)
┌───────────┬────────┬───────────┬───────────┬────────┬────────┬───────────────────────┬────────────────────────┬──────────────────────────┐
│ client_id ┆ ts_key ┆ trans_sum ┆ trans_cnt ┆ region ┆ target ┆ trans_sum__slope__w3 ┆ trans_sum__streak__up ┆ trans_sum__streak__down │
│ i64       ┆ i64    ┆ f32       ┆ f64       ┆ str    ┆ i64    ┆ f32                   ┆ f32                    ┆ f32                      │
╞═══════════╪════════╪═══════════╪═══════════╪════════╪════════╪═══════════════════════╪════════════════════════╪══════════════════════════╡
│ 1         ┆ 1      ┆ 10.0      ┆ 1.0       ┆ msk    ┆ 0      ┆ 0.0                   ┆ 0.0                    ┆ 0.0                      │
│ 1         ┆ 2      ┆ 20.0      ┆ 2.0       ┆ msk    ┆ 1      ┆ 10.0                  ┆ 1.0                    ┆ 0.0                      │
│ 1         ┆ 3      ┆ 15.0      ┆ 1.0       ┆ msk    ┆ 0      ┆ 2.5                   ┆ 0.0                    ┆ 1.0                      │
│ 2         ┆ 1      ┆ 5.0       ┆ 1.0       ┆ spb    ┆ 0      ┆ 0.0                   ┆ 0.0                    ┆ 0.0                      │
│ 2         ┆ 2      ┆ 5.0       ┆ 1.0       ┆ spb    ┆ 0      ┆ 0.0                   ┆ 0.0                    ┆ 0.0                      │
│ 2         ┆ 3      ┆ 5.0       ┆ 1.0       ┆ spb    ┆ 1      ┆ 0.0                   ┆ 0.0                    ┆ 0.0                      │
└───────────┴────────┴───────────┴───────────┴────────┴────────┴───────────────────────┴────────────────────────┴──────────────────────────┘
```

`trans_cnt`/`region`/`target` не в `feature_spec` — вернулись как есть, даже `trans_cnt` не приведён к `float32` (к нему приводятся только реальные product-колонки и наваренные фичи).

Никакого `out_path`: `_df` создаёт временный parquet, читает обратно, удаляет.

---

## Разные трансформеры для разных колонок

```python
feature_spec = [
    ("trans_sum", "minimum"),                     # пресет целиком — ml_toolkit/transformers/presets/minimum.yaml
    ("trans_cnt", {"ewma": {"alphas": [0.3]}}),    # inline-словарь вручную
]
generate_feature_groups_df(df, entity_column_name="client_id", ts_column_name="ts_key", feature_spec=feature_spec).columns
```

```text
['client_id', 'ts_key', 'trans_sum', 'trans_cnt', 'region', 'target',
 'trans_sum__slope__w6', 'trans_sum__slope__w12', 'trans_sum__slope__w24',
 'trans_cnt__ewma__a30', 'trans_cnt__ewma__diff_a30']
```

Пресет с одним изменённым параметром — просто словарь:

```python
import yaml
from pathlib import Path

preset = yaml.safe_load(Path("ml_toolkit/transformers/presets/minimum.yaml").read_text())
preset["slope"]["windows"] = [1, 2]
feature_spec = [("trans_sum", preset)]
```

Один набор трансформеров сразу на несколько колонок — списком имён или `polars.selectors`:

```python
feature_spec = [(["trans_sum", "trans_cnt"], {"slope": {"windows": [3]}})]

import polars.selectors as cs
feature_spec = [(cs.starts_with("trans_"), {"slope": {"windows": [3]}})]  # то же самое
```

```text
['client_id', 'ts_key', 'trans_sum', 'trans_cnt', 'region', 'target', 'trans_sum__slope__w3', 'trans_cnt__slope__w3']
```

`preset` обязателен всегда — `None` даёт `ValueError`, автопресета нет. `AVAILABLE_TRANSFORMER_NAMES` — полный список имён трансформеров.

---

## На диске → на диске

Та же функция без `_df`, с явным `out_path`; `df` может быть `pl.scan_parquet(...)` — не читается в память целиком:

```python
from ml_toolkit.feature_generation import generate_feature_groups

result_cols = generate_feature_groups(
    pl.scan_parquet("df.parquet"), entity_column_name="client_id", ts_column_name="ts_key",
    feature_spec=feature_spec, out_path="df_with_features.parquet",
)
print(result_cols)
```

```text
['trans_sum__slope__w6', 'trans_sum__slope__w12', 'trans_sum__slope__w24',
 'trans_cnt__ewma__a30', 'trans_cnt__ewma__diff_a30']
```

---

## Второй датасет, та же схема

Тот же `feature_spec` и `result_cols` с первого вызова → идентичный набор колонок на другом датасете:

```python
from ml_toolkit.feature_generation import apply_feature_groups

holding_df = pl.DataFrame({
    "holding_id": [9, 9, 9], "ts_key": [1, 2, 3],
    "trans_sum": [3., 6., 9.], "trans_cnt": [1., 2., 3.],
})

apply_feature_groups(
    holding_df, entity_column_name="holding_id", ts_column_name="ts_key",
    feature_spec=feature_spec, accepted_cols=result_cols,
    out_path="holding_with_features.parquet",
)
print(pl.read_parquet("holding_with_features.parquet"))
```

```text
shape: (3, 9)
┌────────────┬────────┬───────────┬───────────┬───────────────────────┬────────────────────────┬────────────────────────┬───────────────────────┬────────────────────────────┐
│ holding_id ┆ ts_key ┆ trans_sum ┆ trans_cnt ┆ trans_sum__slope__w6 ┆ trans_sum__slope__w12 ┆ trans_sum__slope__w24 ┆ trans_cnt__ewma__a30 ┆ trans_cnt__ewma__diff_a30 │
│ i64        ┆ i64    ┆ f32       ┆ f32       ┆ f32                   ┆ f32                    ┆ f32                    ┆ f32                   ┆ f32                        │
╞════════════╪════════╪═══════════╪═══════════╪═══════════════════════╪════════════════════════╪════════════════════════╪═══════════════════════╪════════════════════════════╡
│ 9          ┆ 1      ┆ 3.0       ┆ 1.0       ┆ 0.0                   ┆ 0.0                    ┆ 0.0                    ┆ 1.0                   ┆ 0.0                        │
│ 9          ┆ 2      ┆ 6.0       ┆ 2.0       ┆ 3.0                   ┆ 3.0                    ┆ 3.0                    ┆ 1.3                   ┆ 0.7                        │
│ 9          ┆ 3      ┆ 9.0       ┆ 3.0       ┆ 3.0                   ┆ 3.0                    ┆ 3.0                    ┆ 1.81                  ┆ 1.19                       │
└────────────┴────────┴───────────┴───────────┴───────────────────────┴────────────────────────┴────────────────────────┴───────────────────────┴────────────────────────────┘
```

`feature_spec` здесь должен покрывать все `result_cols` как надмножество — иначе `KeyError`. Проще всего передать тот же `feature_spec`.

---

## Отсев дублирующихся фич

```python
kw = dict(df=df, entity_column_name="client_id", ts_column_name="ts_key",
          feature_spec=[("trans_sum", {"slope": {"windows": [3, 6, 12]}})])

print([c for c in generate_feature_groups_df(**kw).columns if "__" in c])
print([c for c in generate_feature_groups_df(**kw, corr_threshold=0.9).columns if "__" in c])
```

```text
['trans_sum__slope__w3', 'trans_sum__slope__w6', 'trans_sum__slope__w12']
['trans_sum__slope__w3']
```

Один и тот же набор на все `product_cols` сразу, фильтр включён по умолчанию (`0.9`) — `select_features_df`:

```python
from ml_toolkit.feature_generation import select_features_df

select_features_df(
    df, entity_column_name="client_id", ts_column_name="ts_key",
    product_cols=["trans_sum", "trans_cnt"], preset={"slope": {"windows": [3]}},
).columns
```

```text
['client_id', 'ts_key', 'trans_sum', 'trans_cnt', 'region', 'target', 'trans_sum__slope__w3']
```

`trans_cnt__slope__w3` отфильтровался — с `trans_sum__slope__w3` он коррелирует выше 0.9 (фильтр включён по умолчанию, в отличие от группового API). Дисковые/парные аналоги — `select_features`/`apply_selected_features` (та же пара ролей, что у `generate_feature_groups`/`apply_feature_groups`, но с единым `product_cols` вместо `feature_spec`).

---

## Трансформеры

```python
from ml_toolkit.feature_generation import AVAILABLE_TRANSFORMER_NAMES
```

93 штуки, параметры каждого — в докстринге `ml_toolkit/transformers/kernels/{имя}.py`, раздел `Preset entry`:

- *simple statistics*: `window_mean`, `window_median`
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
- *остальное*: `burstiness`, `cross_window_momentum`, `extreme_events`, `flow_regularity`, `growth_quality`, `lag_comparison`, `lifecycle_phase`, `mean_deviation_shape`, `microstructure`, `nonlinearity`, `plateau`, `quantile_persistence`, `recovery_dynamics`, `regime_change`, `trend_consistency`, `value_clustering`, `window_volatility_ratios`, `zero_clustering`
- *tsfresh-inspired*: `c3`, `permutation_entropy`, `change_quantiles`, `energy_ratio_by_chunks`, `index_mass_quantile`, `agg_autocorrelation`
- *catch22/tsfel-inspired*: `dfa`, `transition_matrix`, `acf_characteristic_scale`, `automutual_info`, `auto_period`
- *сегментация как фича*: `segment_gap` (обычно не навариваете сами — подключается через `segment:` в params другого трансформера, см. CLAUDE.md → «Segmentation»)

Имя выходной колонки: `{product_col}__{feature}__{suffix}` (без `__{suffix}`, если у трансформера один безымянный выход, например `growth_since_start`).

Нужны не все выходы трансформера, а только часть из них — `include_suffixes`/`exclude_suffixes` в params. Сегментация по разрывам активности и заполнение NaN — `segment`/`fill_nan`. Подробности — CLAUDE.md → «Segmentation», «Filling NaN», «Selecting output suffixes».

---

## NaN от сегментации → `fill_known_nan`

```python
df_gap = pl.DataFrame({
    "client_id": [1] * 6,
    "ts_key":    [1, 2, 3, 4, 5, 6],
    "trans_sum": [0., 0., 9., 12., 18., 15.],   # молчал, потом начал
})

out = generate_feature_groups_df(
    df_gap, entity_column_name="client_id", ts_column_name="ts_key",
    feature_spec=[("trans_sum", {
        "window_mean": {"windows": [3], "segment": {"strategy": "zero_gap", "gap_threshold": 2}},
    })],
)
print(out)
print(fill_known_nan(out))
```

```text
shape: (6, 4)                                              shape: (6, 4)
┌───────────┬────────┬───────────┬───────────────────┐     ┌───────────┬────────┬───────────┬────────────────────┐
│ client_id ┆ ts_key ┆ trans_sum ┆ ...window_mean__w3 │     │ client_id ┆ ts_key ┆ trans_sum ┆ ...window_mean__w3 │
│ i64       ┆ i64    ┆ f32       ┆ f32                │     │ i64       ┆ i64    ┆ f32       ┆ f32                 │
╞═══════════╪════════╪═══════════╪════════════════════╡     ╞═══════════╪════════╪═══════════╪═════════════════════╡
│ 1         ┆ 1      ┆ 0.0       ┆ NaN                │     │ 1         ┆ 1      ┆ 0.0       ┆ -1.0000e30          │
│ 1         ┆ 2      ┆ 0.0       ┆ NaN                │     │ 1         ┆ 2      ┆ 0.0       ┆ -1.0000e30          │
│ 1         ┆ 3      ┆ 9.0       ┆ 9.0                │     │ 1         ┆ 3      ┆ 9.0       ┆ 9.0                 │
│ 1         ┆ 4      ┆ 12.0      ┆ 10.5               │     │ 1         ┆ 4      ┆ 12.0      ┆ 10.5                │
│ 1         ┆ 5      ┆ 18.0      ┆ 13.0               │     │ 1         ┆ 5      ┆ 18.0      ┆ 13.0                │
│ 1         ┆ 6      ┆ 15.0      ┆ 15.0               │     │ 1         ┆ 6      ┆ 15.0      ┆ 15.0                │
└───────────┴────────┴───────────┴────────────────────┘     └───────────┴────────┴───────────┴─────────────────────┘
```

`zero_gap` безусловно исключает ведущие нули (клиента ещё не было) — это `NaN`, не `0` (`0` = «мало истории»). `fill_known_nan(df)` заполняет их сама, без `preset`: смотрит на имя колонки, находит кернель по токену `transformer`, берёт его `FILL_NAN`. Тут `-1e30`-сентинел, не `-1.0` — `window_mean` считает среднее в масштабе исходной колонки без гарантированной границы, `-1.0` мог бы оказаться настоящим значением.

---

## Частые ошибки

| Ситуация | Что произойдёт |
|---|---|
| `feature_spec` пуст (`[]`) | `ValueError` |
| Элемент `feature_spec` — не пара `(columns, preset)` | `ValueError` |
| Второй элемент пары (или `preset`) не задан (`None`) | `ValueError` — автопресета нет нигде |
| Неизвестное имя трансформера | `ValueError` со списком `AVAILABLE_TRANSFORMER_NAMES` |
| Колонка/селектор вне схемы `df` | `ValueError` с именами отсутствующих колонок |
| Селектор ни во что не резолвится | `logger.warning`, группа пропускается |
| `(columns, {})` | Осознанный пропуск — колонка в выходе, фич по ней нет |
| Одна колонка+трансформер, разные параметры в разных группах | `ValueError` — конфликт, не тихий выбор |
| `accepted_cols` в `apply_*` содержит колонку вне `feature_spec`/`product_cols` | `KeyError` |

---

## Общие параметры

| Параметр | Смысл |
|---|---|
| `entity_column_name` / `ts_column_name` | Колонка-идентификатор сущности / колонка с датой. |
| `min_output_ts_key` / `max_output_ts_key` | Границы по `ts_column_name` (включительно), режут выход **после** наварки — история для окон не обрезается. |
| `name` | Метка для логов/tqdm. |
| `out_path` / `tmp_dir` | Только у функций без `_df` — путь к результату / к временным parquet-кандидатам. `tmp_dir=None` → авто-temp. |

---

## Под капотом

1. `df` (отсортированный по entity/ts) целиком пишется во временный parquet — отсюда сквозные колонки в выходе.
2. Для каждой пары (колонка, трансформер) фичи считаются и сразу пишутся в свой parquet — широкий набор фич разом в памяти не держится.
3. Корреляционный фильтр (если задан `corr_threshold`) — жадный Пирсон, кандидат за кандидатом; сквозных колонок не касается.
4. Сборка выхода читает всё по одному row group за раз: product-колонки и фичи → `float32`, сквозные колонки — как есть.

`apply_feature_groups`/`apply_selected_features` — тот же путь без шага 3, сразу по готовому `accepted_cols`.
