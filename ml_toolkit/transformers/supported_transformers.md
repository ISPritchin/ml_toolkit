# Справочник трансформеров фич

**81 трансформер · 292 выходные колонки на product-колонку** (число приведено для референсной конфигурации со всеми 81 трансформерами и параметрами, показанными в секциях `Preset` ниже — no library-shipped "full" preset with this configuration exists; preset обязателен и всегда явный, автоматического дефолта нет, см. CLAUDE.md → Preset system. Другой набор трансформеров/параметров даёт другое число — см. таблицу в конце документа).

Каждый трансформер — отдельный файл в `ml_toolkit/transformers/kernels/{name}.py`, экспортирующий `FEATURE: str` и `compute(values, position, params) -> (arrays, suffixes)`. Все кернелы работают за **один последовательный проход** по массиву, отсортированному по `(entity, ts_key)`; группы `entity` идут подряд, поэтому достаточно знать 0-индексную позицию строки внутри своей сущности (`position_within_entity`).

Документ сгенерирован по актуальным докстрингам кернелов (секции Signal/Formula/Outputs/Preset/Interpretation/Example, см. правило в `CLAUDE.md`) — формулы и примеры здесь идентичны тем, что лежат в коде.

---

## Общие обозначения

| Символ | Смысл |
|--------|-------|
| `v[t]` | Значение product-колонки в текущей строке |
| `v[t-k]` | Значение $k$ строк назад (внутри той же сущности) |
| `w` | Запрошенный размер окна (6, 12, 24 и т.д.) |
| `ws` (эффективное окно) | `ws = min(pos+1, w)`, где `pos` — 0-индексная позиция строки в сущности |
| `mean_w`, `std_w` | Среднее/стандартное отклонение окна (std смещённое, делитель `ws`, без коррекции Бесселя) |
| `eps` | `EPS = 1e-9` (защита от деления на ноль) |

**Название выходной колонки:** `{product_col}__{FEATURE}__{suffix}`, либо `{product_col}__{FEATURE}` без суффикса, если у трансформера один безусловный выход (двойное подчёркивание — обязательная часть конвенции, см. `_col_name` в `ml_toolkit/feature_generation.py`).

Окно `[t-w+1, t]` включительно; при `pos < w-1` оно сужается до `[t-pos, t]` (`resolve_window_size`). Каждый пример ниже использует свой минимальный числовой ряд — тот, что приведён в докстринге соответствующего кернела.

### Контракты (`ml_toolkit/transformers/_windowing.py`)

- **Неотрицательность.** `product_values` ожидаются неотрицательными (денежные объёмы). Кернелы не падают на отрицательных значениях, но лог-признаки (`log1p(|v|)`) и ratio-признаки теряют знак/интерпретацию.
- **0 = «недостаточно истории».** Ноль в выходе — конвенция для случая `pos < требуемого лага/окна`; она неотличима от легитимного нулевого сигнала. Учитывать при интерпретации значений, близких к нулю у молодых сущностей.
- **`safe_ratio(num, den)`** — единственная разрешённая форма нормировки отношений: `0.0` при `|den| <= EPS`, иначе `num / |den|` с клампом в `[-RATIO_CAP, RATIO_CAP]` (`RATIO_CAP = 1e6`). Старый паттерн `x / (|y| + eps)` при нулевом знаменателе давал выбросы ~1e10, ломавшие float32-каст и корреляционный фильтр — «отношение не определено» (нулевая база) везде трактуется как `0`, а не как взрыв.
- **Медиана/квантили из отсортированного окна** (`sorted_median`, `sorted_quantile`) — единая конвенция для всех кернелов: честная медиана (среднее двух центральных элементов при чётном `ws`), квантиль — `sorted[int(q * (ws - 1))]` (симметрично для p25/p75, p10/p90).

---

## 1. `window_mean` / `window_median` — простая статистика окна

**Файлы:** `window_mean.py`, `window_median.py`

**Сигнал:** базовые среднее и медиана окна — компоненты для многих других трансформеров (CV, zscore, mean_median_gap и т.д.), но и самостоятельные признаки уровня.

| Колонка | Формула |
|---------|---------|
| `window_mean__w3`, `__w6`, `__w12` | `sum(v[t-w+1..t]) / w` |
| `window_median__w3`, `__w6`, `__w12` | Медиана отсортированного окна (честная — среднее двух центральных при чётном `w`) |

**Пример (`window_mean`, ряд `[10, 20, 30, 40]`, w=3):** `mean = (20+30+40)/3 = 30.0` → `window_mean__w3 = 30.0`.

**Пример (`window_median`, ряд `[10, 40, 20, 30]`, w=3):** окно `[40, 20, 30]` → сортировка `[20, 30, 40]` → `median = 30.0`.

---

## 2. `trend` — направление и сила тренда

**Файлы:** `slope.py`, `slope_ratio.py`, `momentum.py`, `direction_flag.py`, `max_abs_jump.py`, `streak.py`, `growth_since_start.py`

**Сигнал:** куда идёт клиент и насколько устойчиво.

| Колонка | Формула |
|---------|---------|
| `slope__w6`, `__w12`, `__w24` | OLS-наклон: `(n·Σ(i·v) − Σi·Σv) / (n·Σi² − (Σi)²)` по позициям `i=0..n-1` окна |
| `slope_ratio__w6_w12`, `__w12_w24` | `slope_short / (\|slope_long\| + eps)` |
| `momentum__h3`, `__h6` | `mean(последние h) / (\|mean(предыдущие h)\| + eps) − 1` |
| `direction_flag__w6`, `__w12` | `sign(slope_w)` ∈ {-1, 0, +1} |
| `max_abs_jump__w6`, `__w12` | `max(\|v[i] - v[i-1]\|)` в окне |
| `streak__up`, `__down` | Длина текущей серии строгого роста/падения (running state) |
| `growth_since_start` (без суффикса) | `(v[t] - first_nonzero) / (\|first_nonzero\| + eps)` (running state) |

**Интерпретация:**
- `slope_w6 > slope_w12 > 0` — ускорение роста (последнее полугодие круче).
- `direction_flag_w6 = +1, direction_flag_w12 = -1` — краткосрочный разворот вверх на фоне долгосрочного снижения.
- `momentum_h6 > 0, h3 < 0` — среднесрочное ускорение при краткосрочном откате.

**Пример (`slope`, ряд `[10, 20, 30, 40, 50]`, w=5):** `Σi=10, Σv=150, Σ(iv)=400, Σi²=30` → `slope = (5·400−10·150)/(5·30−100) = 500/50 = 10.0`.

**Пример (`momentum`, ряд `[10, 20, 30, 40, 50, 60]`, h=3):** `recent=(40+50+60)/3=50`, `prior=(10+20+30)/3=20` → `momentum__h3 = 50/20−1 = 1.5`.

---

## 3. `volatility` — разброс и изменчивость

**Файлы:** `rolling_std.py`, `rolling_cv.py`, `rolling_min_max.py`, `extreme_share.py`, `skew_proxy.py`

**Сигнал:** насколько непредсказуем клиент.

| Колонка | Формула |
|---------|---------|
| `rolling_std__w6`, `__w12`, `__w24` | `sqrt(mean((v[i]-mean_w)²))`, смещённое std |
| `rolling_cv__w6`, `__w12`, `__w24` | `std_w / (\|mean_w\| + eps)` |
| `rolling_min_max__min_w6`, `__max_w6`, `__min_w12`, `__max_w12` | Min/max окна |
| `extreme_share__extreme_w6`, `__balance_w6`, `__extreme_w12`, `__balance_w12` | `extreme = count(\|v-mean\|>1.5σ)/ws`; `balance = count(v>mean)/ws − 0.5` |
| `skew_proxy__w6`, `__w12` | `(mean_w - lo_w) / (hi_w - lo_w + eps)` |

**Интерпретация:**
- `rolling_cv = 0` — абсолютно стабильный ряд; `> 1` — экстремальная нестабильность.
- `extreme_share__extreme_w12 > 0.3` — более 30% месяцев выходят за 1.5σ: нестабильный клиент.
- `skew_proxy ≈ 0.5` — среднее посередине между min/max (симметрия); `< 0.3` — правосторонняя асимметрия (редкие пики на низком фоне).

**Пример (`rolling_cv`, ряд `[10,10,10,10,10,40]`, w=6):** `mean=15, std=11.18` → `rolling_cv__w6 = 11.18/15 = 0.745`.

**Пример (`extreme_share`, тот же ряд):** `1.5σ=16.77`; экстремален только `40` (`|40-15|=25`) → `extreme_w6 = 1/6 = 0.167`, `balance_w6 = 1/6 − 0.5 = −0.333`.

---

## 4. `tenure_activity` — стаж и активность

**Файлы:** `tenure.py`, `recency.py`, `inactive_streak.py`, `active_months.py`, `active_run_count.py`, `activity_rate.py`, `longest_active_run.py`, `client_age.py`, `zero_share.py`

**Сигнал:** как давно клиент существует, как долго молчал, насколько регулярен.

| Колонка | Формула |
|---------|---------|
| `tenure__tenure_months`, `__first_active_flag` | Месяцев с первой ненулевой транзакции (running); флаг самой строки первой активации |
| `recency__recency_gap` | `current_pos - last_active_pos` (`-1`, если активности ещё не было) |
| `inactive_streak__current`, `__max` | Текущая/исторический максимум серии нулей (running) |
| `active_months__w6`, `__w12`, `__w24` | `count(v[i] != 0)` в окне |
| `active_run_count__w6`, `__w12` | Число переходов `0 → ненулевое` в окне |
| `activity_rate__share_of_tenure_active` | `count_active_total / tenure` (running) |
| `longest_active_run__w6`, `__w12` | Наибольшая непрерывная серия ненулевых в окне |
| `client_age__new_client_flag`, `__months_since_start_norm` | `pos < 3`; `pos / (pos + 12)` ∈ [0,1) |
| `zero_share__w3`, `__w6`, `__w12` | `count(v[i]==0) / w` |

**Интерпретация:**
- `active_months_w12 = 12` — потоковый клиент; `3–5` — проектный B2B с редкими крупными поступлениями.
- `inactive_streak__max = 6, current = 0` — когда-то выпадал на полгода, сейчас восстановился.
- `activity_rate > 0.9` — практически без пропусков; `< 0.3` — редкий/проектный клиент.

**Пример (общий ряд `B = [10, 0, 0, 30, 0]`, t=4):** `recency__recency_gap`: последняя активность на `pos=3` → `gap = 4-3 = 1`. `active_months__w6` (ряд `[10,0,5,0,8,3]`): ненулевые `10,5,8,3` → `4`. `tenure__tenure_months` (ряд `[0,0,10,20,30]`, первая активность `pos=2`): `4-2+1 = 3`.

---

## 5. `dynamics_ratios` — MoM-динамика и масштабированные метрики

**Файлы:** `ewma.py`, `lag1_diff.py`, `mean_median_gap.py`, `recent_share.py`, `rolling_sum.py`, `zscore.py`

**Сигнал:** как меняется значение месяц к месяцу и относительно своей истории.

| Колонка | Формула |
|---------|---------|
| `ewma__a30`, `__diff_a30` | `EWMA[t] = 0.3·v[t] + 0.7·EWMA[t-1]`; `diff = v[t] - EWMA[t]` |
| `lag1_diff__diff`, `__log_diff`, `__pct_change` | `v[t]-v[t-1]`; `log1p(\|v[t]\|)-log1p(\|v[t-1]\|)`; `(v[t]-v[t-1])/(\|v[t-1]\|+eps)` |
| `mean_median_gap__w6`, `__w12` | `(mean_w - median_w) / (\|mean_w\| + eps)` |
| `recent_share__r3_w12`, `__r6_w24` | `sum_short / (\|sum_long\| + eps)` |
| `rolling_sum__w3`, `__w6`, `__w12` | `sum(v[t-w+1..t])` |
| `zscore__w6`, `__w12`, `__w24` | `(v[t]-mean_w)/(std_w+eps)` |

**Интерпретация:**
- `ewma_diff > 0` — текущий месяц выше сглаженного тренда (позитивный импульс).
- `recent_share__r3_w12 = 0.40` при равномерном ожидании `0.25` — квартал даёт больше нормы (рост).
- `zscore_w12 > 2` — аномально высокий месяц (разовая сделка/сезонный пик).
- `mean_median_gap > 0.3` — сильная правосторонняя асимметрия (нули + редкие всплески), типично для B2B.

**Пример (`ewma`, ряд `[100,80,120,90]`, alpha=0.3):** `EWMA: 100 → 94.0 → 101.8 → 98.26` → `ewma__a30=98.26`, `diff_a30 = 90-98.26 = -8.26`.

**Пример (`zscore`, ряд `[10,10,10,10,10,40]`, w=6):** `mean=15, std=11.18` → `zscore__w6 = (40-15)/11.18 = 2.236`.

---

## 6. `trend_change` — изменения тренда и просадки

**Файлы:** `accel.py`, `max_drawdown.py`, `peak_trough_timing.py`, `run_above_mean.py`, `sign_change_count.py`, `time_weighted_momentum.py`, `trend_flip.py`, `volatility_of_diff.py`

**Сигнал:** ускоряется/тормозит ли тренд, случались ли развороты и просадки.

| Колонка | Формула |
|---------|---------|
| `accel` (без суффикса) | `v[t] - 2·v[t-1] + v[t-2]` (вторая разность) |
| `max_drawdown__w6`, `__w12`, `__w24` | `max((running_peak - v[j]) / (\|running_peak\|+eps))` в окне |
| `peak_trough_timing__peak_w6/12`, `__trough_w6/12` | Месяцев с максимума/минимума окна |
| `run_above_mean__w12` | Текущая серия месяцев подряд выше скользящего среднего (running) |
| `sign_change_count__w6`, `__w12` | Число смен знака приращения в окне |
| `time_weighted_momentum__w6`, `__w12` | `Σ(i·d[i]) / (\|Σv\|+eps)` — приращения, взвешенные позицией в окне |
| `trend_flip__flag`, `__slope_change_lag6_w6`, `__slope_change_lag12_w12` | Флаг смены знака slope vs `lag` месяцев назад; численная дельта наклона |
| `volatility_of_diff__w6`, `__w12` | `std(v[i]-v[i-1])` в окне |

**Интерпретация:**
- `accel > 0` — тренд ускоряется («разгон»); `< 0` — тормозит («усталость тренда»).
- `max_drawdown = 0.9375` — потеря ~94% от локального пика окна (глубокая просадка).
- `trend_flip__flag = 1` — тренд поменял знак относительно `lag` месяцев назад (разворот).
- `run_above_mean = 12` — весь год выше среднего: аномальная активность/рост.

**Пример (`accel`, ряд `[10,15,25,30]`):** `(30-25)-(25-15) = 5-10 = -5` (тормозит).

**Пример (`max_drawdown`, ряд `[10,80,40,20,5,30]`, w=6):** пик `80`, минимум после него `5` → `(80-5)/80 = 0.9375`.

---

## 7. `relative_position` — положение значения в истории

**Файлы:** `cumulative_share.py`, `distance_to_global_max.py`, `half_ratio.py`, `lag_growth_ratio.py`, `level_ratio.py`, `local_extrema.py`, `log1p_level.py`, `pct_of_max.py`, `rank_in_window.py`, `trough_to_current.py`, `volatility_trend.py`

**Сигнал:** где текущее значение находится относительно своего максимума/минимума/истории.

| Колонка | Формула |
|---------|---------|
| `cumulative_share` (без суффикса) | `v[t] / (\|cum_sum[t]\|+eps)` — доля в накопленной сумме (running) |
| `distance_to_global_max` (без суффикса) | `(v[t]-running_max)/(\|running_max\|+eps)`, всегда `<= 0` (running) |
| `half_ratio__w6`, `__w12` | `sum(вторая половина окна) / (\|sum(первая половина)\|+eps)` |
| `lag_growth_ratio__lag3`, `__lag6`, `__lag12` | `v[t]/(\|v[t-k]\|+eps) - 1` |
| `level_ratio__w3_w12`, `__w6_w24` | `mean_short/(\|mean_long\|+eps)` |
| `local_extrema__w6`, `__w12` | Число локальных пиков+впадин в окне |
| `log1p_level` (без суффикса) | `sign(v[t])·log1p(\|v[t]\|)` |
| `pct_of_max__w6`, `__w12`, `__w24` | `safe_ratio(v[t], max_w)` |
| `rank_in_window__w6`, `__w12`, `__w24` | `count(v[i] <= v[t]) / ws` |
| `trough_to_current__w6`, `__w12` | `safe_ratio(v[t], min_w)` |
| `volatility_trend__w3_w12`, `__w6_w12` | `std_short - std_long` |

**Интерпретация:**
- `distance_to_global_max = 0` — сущность на своём историческом максимуме прямо сейчас; `< -0.5` — упала более чем на 50% от пика.
- `pct_of_max_w6 = 1, pct_of_max_w24 = 0.4` — краткосрочный локальный пик, но далеко от долгосрочного максимума.
- `trough_to_current > 2` — удвоение оборота от минимума окна (сильное восстановление).
- `local_extrema_w12 = 0` — монотонный тренд без разворотов; `≈ 5-6` — сильная осцилляция.

**Пример (`distance_to_global_max`, ряд `[10,30,20,25]`):** `running_max=30` → `(25-30)/30 = -0.167` (на ~17% ниже исторического пика).

**Пример (`pct_of_max`, ряд `[10,20,90,40,30,45]`, w=6):** `hi=90` → `45/90 = 0.5`.

---

## 8. `structural_signals` — структурные сигналы и неравномерность

**Файлы:** `corr_with_time.py`, `cusum.py`, `entropy.py`, `gini.py`

**Сигнал:** линейность тренда, накопленные отклонения, концентрация/неравномерность.

| Колонка | Формула |
|---------|---------|
| `corr_with_time__w6`, `__w12` | Pearson({0..ws-1}, окно значений) |
| `cusum__pos_w6`, `__neg_w6`, `__pos_w12`, `__neg_w12` | Накопленные `max(0, v[i]-mean_w)` / `min(0, v[i]-mean_w)` (в единицах колонки) |
| `entropy__w6`, `__w12` | Нормированная энтропия Шеннона `-Σ(p·ln p)/ln(ws)` ∈ [0,1], только по `v[i]>0` |
| `gini__w6`, `__w12` | Коэффициент Джини по окну (через отсортированный буфер, O(ws·log ws)) |

**Интерпретация:**
- `|corr_with_time| > 0.9` — очень чистый линейный тренд.
- `entropy ≈ 1.0` — доход равномерно распределён по месяцам; `≈ 0` — весь доход в одном месяце.
- `gini > 0.6` — экстремальная концентрация (1-2 месяца дают почти весь доход).
- `cusum_pos_w12 >> |cusum_neg_w12|` — распределение смещено вправо (проектный B2B-профиль).

**Пример (`corr_with_time`, ряд `[10,30,20,40,30,50]`, w=6):** `r = 0.832` (растущий, но зубчатый тренд).

**Пример (`gini`, ряд `[0,10,30,60]`, w=4):** `gini = 400/(2·4·100) = 0.5`.

---

## 9. `autocorrelation` — память ряда

**Файл:** `autocorr.py`

**Сигнал:** насколько текущий месяц похож на предыдущий/позапрошлый/на месяц три назад.

| Колонка | Формула |
|---------|---------|
| `autocorr__lag1`, `__lag2`, `__lag3` | Expanding Pearson по парам `(v[t-k], v[t])` с начала истории |
| `autocorr__lag1_w12`, `__lag2_w12` | Windowed Pearson по парам внутри окна 12 |
| `autocorr__partial_lag2` | Частичная автокорреляция лага 2 (Yule-Walker): `(r2-r1²)/(1-r1²+eps)` |

**Интерпретация:**
- `lag1 ≈ +1` — сильная инерция (рост следует за ростом); `lag1 ≈ -1` — жёсткая осцилляция.
- `partial_lag2 ≈ 0` при высоком `lag2` — лаг-2 корреляция полностью объяснена лагом-1.

**Пример (ряд `[10,20,15,25,20]`):** `lag1 = -100/316.23 = -0.316` (лёгкая осцилляция).

---

## 10. `seasonal_proxy` — сезонность без дат

**Файл:** `seasonal_autocorr.py`

**Сигнал:** повторяющиеся паттерны с полугодовым/годовым периодом.

| Колонка | Формула |
|---------|---------|
| `seasonal_autocorr__lag6`, `__lag12` | Expanding Pearson по парам `(v[t-6], v[t])` / `(v[t-12], v[t])` |
| `seasonal_autocorr__lag6_w24`, `__lag12_w24` | Windowed Pearson в окне 24 |
| `seasonal_autocorr__quarter_cv_w12` | `std(Q1..Q4)/|mean(Q1..Q4)|`, где Qi — средние 4 троек окна 12 |
| `seasonal_autocorr__even_odd_w12` | `mean(чётные позиции внутри сущности) / |mean(нечётные)|` |
| `seasonal_autocorr__amplitude_w12` | `(max(Q)-min(Q))/|mean_w12|` |

**Интерпретация:**
- `lag12 ≈ +1` — сильная годовая сезонность; `lag6 ≈ +1` — полугодовая периодичность (бюджетные клиенты).
- `even_odd = 0.333` — нечётные месяцы втрое сильнее чётных (бимесячный ритм).

Чётность в `even_odd` считается от позиции **внутри сущности** (`pos % 2`), а не от начала окна — иначе при сдвиге окна на месяц чётности инвертируются и признак осциллирует на стабильном бимесячном паттерне.

**Пример (ряд `[10,30,10,30,10,30]`):** чётные `→ mean=10`, нечётные `→ mean=30` → `even_odd_w12 = 10/30 = 0.333`.

---

## 11. `log_growth` — мультипликативная динамика

**Файлы:** `geometric_return.py`, `log_level.py`, `log_slope.py`, `log_slope_ratio.py`, `log_volatility.py`

**Сигнал:** темп и устойчивость роста в лог-шкале — устойчивы к экспоненциальному распределению оборотов.

| Колонка | Формула |
|---------|---------|
| `geometric_return__w6`, `__w12` | `exp(mean(log1p(\|v[i]\|)-log1p(\|v[i-1]\|))) - 1` |
| `log_level__w6`, `__w12` | `log1p(\|mean_w\|)` |
| `log_slope__w6`, `__w12`, `__w24` | OLS-наклон по `log1p(\|v\|)` окна |
| `log_slope_ratio__w6_w12` | `log_slope_short / (\|log_slope_long\|+eps)` |
| `log_volatility__w6`, `__w12` | `std(log_diff)` в окне |

**Интерпретация:**
- `geometric_return ≈ +0.19` — рост ≈19% в месяц.
- `log_slope ≈ 0` — стагнация в log-шкале; в паре с `log_volatility` высоким — рост с высоким риском.
- `log_slope_ratio > 1` — краткосрочный log-темп роста выше долгосрочного (ускорение).

**Пример (`geometric_return`, ряд `[10,20,40,80]`, w=4 — удвоение каждый месяц):** `mean(log_diff) ≈ 0.665` → `geometric_return__w4 = exp(0.665)-1 = 0.946` (≈+95%/мес).

**Пример (`log_slope`, тот же ряд):** OLS-наклон по `log1p` ≈ `0.666`/мес.

---

## 12. `smoothness` — шероховатость ряда

**Файлы:** `alternation_rate.py`, `roughness_ratio.py`, `total_variation.py`

**Сигнал:** насколько «рваным» выглядит движение ряда.

| Колонка | Формула |
|---------|---------|
| `alternation_rate__alt_rate_w6/12` | Доля смен знака приращения из всех пар окна |
| `alternation_rate__max_jump_share_w6/12` | `max(\|d[i]\|) / (TV_w+eps)` |
| `alternation_rate__mean_abs_jump_w6/12` | `TV_w / n_pairs` |
| `roughness_ratio__w6_w12` | `TV_norm_short / (TV_norm_long+eps)` |
| `total_variation__w6`, `__norm_w6`, `__w12`, `__norm_w12` | `Σ|v[i]-v[i-1]|` (в ед. колонки) и нормированная на `mean_w` версия |

**Интерпретация:**
- `alt_rate = 1.0` — каждый месяц смена направления (пилообразный ряд); `= 0` — монотонный тренд.
- `roughness_ratio > 2` — краткосрочный период вдвое «рваней» долгосрочного.
- `TV_norm = 0` — абсолютно гладкий ряд.

**Пример (`alternation_rate`, ряд `[10,30,20,40,30,50]`, w=6):** `d = +20,-10,+20,-10,+20`, `TV=80`, все 4 смежные пары меняют знак → `alt_rate_w6=1.0`, `max_jump_share_w6=20/80=0.25`, `mean_abs_jump_w6=80/5=16.0`.

---

## 13. `distribution_moments` — форма распределения в окне

**Файл:** `kurtosis_proxy.py`

**Сигнал:** тяжёлые хвосты (редкие крупные месяцы) vs равномерное распределение.

| Колонка | Формула |
|---------|---------|
| `kurtosis_proxy__kurt_w6/12` | `mean(((v-mean)/std)^4) - 3` |
| `kurtosis_proxy__p75_p25_w6/12`, `__p90_p10_w6/12` | Отношения квантилей окна |
| `kurtosis_proxy__upper_tail_w6/12`, `__lower_tail_w6/12` | Доля суммы, приходящаяся на верхний/нижний квартиль |

**Интерпретация:**
- `kurt > 3` — тяжёлые хвосты (проектный B2B-паттерн); `< -1` — плоское распределение (потоковый клиент).
- `upper_tail > 0.8` — верхние 25% месяцев дают 80% объёма (крайняя концентрация).

**Пример (ряд `[10,10,10,10,10,70]`, w=6):** `mean=20, std=22.36` → `Σz⁴=25.2` → `kurt_w6 = 25.2/6 - 3 = 1.2` (тяжёлый правый хвост).

---

## 14. `burstiness` — проектный vs потоковый клиент

**Файл:** `burstiness.py`

**Сигнал:** редкие крупные всплески с долгими паузами vs равномерный поток.

| Колонка | Формула |
|---------|---------|
| `burstiness__peak_mean_w12` | `max_w / (\|mean_w\|+eps)` |
| `burstiness__peak_med_w12` | `max_w / (\|median_w\|+eps)` |
| `burstiness__gap_mean_w12` | `zero_count / max(burst_count, 1)` |
| `burstiness__burst_count_w12` | Число переходов `0 → ненулевое` |
| `burstiness__burst_dur_w12` | Средняя длина ненулевой серии |
| `burstiness__burst_cv_w12` | `std(длины серий) / (mean_длины+eps)` |
| `burstiness__calm_share_w12` | `zero_count / ws` |

**Интерпретация:** `peak_mean > 4, calm_share > 0.5` — типичный проектный B2B-клиент; `peak_mean ≈ 1.1, calm_share ≈ 0` — потоковый клиент без всплесков.

**Пример (ряд `[0,40,30,0,20,0]`, w=6):** `mean=15, median=20, max=40`; серии `[40,30]` и `[20]` → `burst_count=2, burst_dur=1.5, gap_mean=1.5, calm_share=0.5`; `peak_mean_w6=40/15=2.667`, `peak_med_w6=40/20=2.0`.

---

## 15. `flow_regularity` — регулярность поступлений

**Файл:** `flow_regularity.py`

**Сигнал:** насколько равномерен ритм активности (интервалы между всплесками).

| Колонка | Формула |
|---------|---------|
| `flow_regularity__gap_mean_w12` | Средняя длина нулевой серии между активными сериями |
| `flow_regularity__gap_std_w12` | σ промежутков |
| `flow_regularity__gap_cv_w12` | `gap_std / (gap_mean+eps)` |
| `flow_regularity__is_monthly_w12` | Флаг: активен почти каждый месяц |
| `flow_regularity__cadence_shift_w12` | Флаг замедления ритма (последняя половина промежутков заметно длиннее) |
| `flow_regularity__active_len_cv_w12` | CV длин активных серий |

**Интерпретация:** `gap_cv = 0` — идеально ритмичный клиент; `cadence_shift = 1` — клиент «замолчал» дольше обычного (риск оттока).

**Пример (ряд `[5,0,0,8,0,0]`, w=6):** одна пауза длиной 2 между сериями → `gap_mean_w6=2.0, gap_cv_w6=0.0` (идеально равномерный ритм).

---

## 16. `recovery_dynamics` — восстановление после просадки

**Файл:** `recovery_dynamics.py`

**Сигнал:** насколько и как быстро клиент восстановился от минимума окна.

| Колонка | Формула |
|---------|---------|
| `recovery_dynamics__completeness_w12` | `(v[t]-w_min)/(w_max-w_min+eps)` ∈ [0,1] |
| `recovery_dynamics__drawdown_dur_w12` | `count(v[i] < w_max)` в окне |
| `recovery_dynamics__post_trough_gain_w12` | `(v[t]-w_min)/(\|mean_w\|+eps)` |
| `recovery_dynamics__trough_is_recent_w12` | Флаг: дно было ≤3 мес назад |
| `recovery_dynamics__speed_w12` | `(v[t]-w_min)/(months_since_trough+1)` |
| `recovery_dynamics__is_recovering_now` | Флаг: 3 месяца подряд роста, но ещё ниже среднего |

**Интерпретация:** `completeness = 1.0` — полное восстановление от минимума до максимума окна; `is_recovering_now = 1` — активная, но ещё не завершённая фаза роста.

**Пример (ряд `[10,80,40,20,5,30]`, w=6):** `w_min=5, w_max=80, v[t]=30` → `completeness = 25/75 = 0.333`, `months_since_trough=1`, `speed = 25/2 = 12.5`.

---

## 17. `growth_quality` — органичность роста

**Файл:** `growth_quality.py`

**Сигнал:** много равных небольших приростов («органичный» рост) vs один крупный скачок («проектный»).

| Колонка | Формула |
|---------|---------|
| `growth_quality__best_share_w12` | `max(Δ+) / (Σ(Δ+)+eps)` |
| `growth_quality__consist_score_w12` | `count(Δ+>0) / max(active_months,1)` |
| `growth_quality__pos_count_w12` | Число месяцев с положительным приростом |
| `growth_quality__growth_gini_w12` | Gini по положительным приростам |
| `growth_quality__organic_w12` | `1 - best_share` |
| `growth_quality__neg_sum_share_w12` | Доля суммарных потерь к обороту |

**Интерпретация:** `organic ≈ 0.91` — рост равномерно распределён по месяцам; `organic ≈ 0` — весь рост от одного скачка/контракта.

**Пример (ряд `[10,20,30,40,50,60]`, w=6):** пять равных приростов `+10` → `best_share = 10/50 = 0.2`, `organic_w6 = 0.8`, `pos_count_w6 = 5`, `consist_score_w6 = 1.0`.

---

## 18. `lag_comparison` — лаговые сравнения (QoQ, 9m, YoY)

**Файл:** `lag_comparison.py`

**Сигнал:** сравнение с кварталом, тремя кварталами и годом назад.

| Колонка | Формула |
|---------|---------|
| `lag_comparison__lag3_ratio`, `__lag9_ratio`, `__lag12_ratio` | `v[t]/(\|v[t-k]\|+eps) - 1` |
| `lag_comparison__lag3_trend` | `mean(lag3_ratio[t], [t-1], [t-2])` |
| `lag_comparison__lag12_consistency` | `std(lag12_ratio[t], [t-1], [t-2])` |
| `lag_comparison__yoy_accel` | `lag12_ratio[t] - lag12_ratio[t-6]` |

**Интерпретация:** `lag12_ratio = +1.0` — доход удвоился год к году; `yoy_accel > 0` — YoY-рост ускоряется.

**Пример (ряд `[10,20,30,40]`, доступен только лаг 3):** `lag3_ratio = 40/10 - 1 = 3.0` (рост ×4 за квартал); lag9/lag12 = 0 (мало истории).

---

## 19. `regime_change` — структурный сдвиг уровня

**Файл:** `regime_change.py`

**Сигнал:** резкий переход с одного уровня дохода на другой (перебор всех точек разбиения окна).

| Колонка | Формула |
|---------|---------|
| `regime_change__magnitude_w12` | `max_k(\|mean_left(k)-mean_right(k)\|/(std_w+eps))` |
| `regime_change__split_pos_w12` | Оптимальная точка разрыва `k*` |
| `regime_change__flag_w12` | `1 если magnitude > 2.0` |
| `regime_change__late_vs_early_w12` | `(mean_поздняя_половина - mean_ранняя)/(std_w+eps)` |
| `regime_change__asymmetry_w12` | `mean(последние 3) / (\|mean(первые 3)\|+eps)` |
| `regime_change__current_regime_len` | Running: месяцев без сдвига флага |

**Интерпретация:** `magnitude=2.33, flag=1` — явная смена режима (контракт/расторжение); `current_regime_len=0` — сдвиг обнаружен прямо сейчас.

**Пример (ряд `[0,0,100,100,100,100]`, w=6):** оптимальный разрыв на `k=2` → `mean_left=0, mean_right=100, std=47.14` → `magnitude=2.121 > 2 → flag_w6=1`.

---

## 20. `nonlinearity` — выпуклость и кривизна тренда

**Файл:** `nonlinearity.py`

**Сигнал:** U-образный / дугообразный тренд vs равномерное ускорение (метод третей окна).

| Колонка | Формула |
|---------|---------|
| `nonlinearity__quad_proxy_w6/12` | `(Q1 - 2·Q2 + Q3) / (\|mean_w\|+eps)`, Qi — средние третей окна |
| `nonlinearity__convexity_sign_w6/12` | `sign(quad_proxy)` |
| `nonlinearity__mean_accel_w6/12` | Средняя вторая разность `v[i]-2v[i-1]+v[i-2]` |
| `nonlinearity__accel_std_w6/12` | σ вторых разностей |
| `nonlinearity__frac_concave_w6/12` | Доля шагов с `accel[i] < 0` |

**Интерпретация:** `quad_proxy < 0` — дугообразный (пик посередине); `> 0` — U-образный (просадка → восстановление).

**Пример (ряд `[10,20,30,20,10,5]`, w=6):** `Q1=15, Q2=25, Q3=7.5, mean=15.833` → `quad_proxy_w6 = (15-50+7.5)/15.833 = -1.737` (пик посередине).

---

## 21. `trend_consistency` — чистота тренда

**Файл:** `trend_consistency.py`

**Сигнал:** насколько «чист» тренд — согласованность знаков приращений с общим направлением, доля объяснённой дисперсии (R²).

| Колонка | Формула |
|---------|---------|
| `trend_consistency__dir_consistency_w6/12` | `count(sign(Δ)==sign(slope)) / (w-1)` |
| `trend_consistency__noise_signal_w6/12` | RMSE остатков / (`\|slope\|·w+eps`) |
| `trend_consistency__clean_streak_w6/12` | Длиннейшая серия шагов, согласованных со знаком тренда |
| `trend_consistency__sub_sign_consist_w6/12` | Доля 3-месячных подокон с тем же знаком наклона |
| `trend_consistency__r_squared_w6/12` | `1 - SS_res/(SS_tot+eps)` |

**Интерпретация:** `dir_consistency=1.0, r_squared=1.0` — идеальная монотонность, весь тренд линеен без шума.

**Пример (ряд `[10,20,30,40,50,60]`, w=6):** все 5 приростов `+10`, точки строго на линии → `dir_consistency_w6=1.0`, `r_squared_w6=1.0`, `clean_streak_w6=5`.

---

## 22. `window_volatility_ratios` — эволюция волатильности

**Файл:** `window_volatility_ratios.py`

**Сигнал:** на каком горизонте (3/6/12/24 мес) сконцентрирован «хаос».

| Колонка | Формула |
|---------|---------|
| `window_volatility_ratios__cv_ratio_w3_w6`, `__cv_ratio_w3_w12`, `__cv_ratio_w6_w24` | `CV_short / (CV_long+eps)` |
| `window_volatility_ratios__vol_accel` | `(std_3-std_6) - (std_6-std_12)` |
| `window_volatility_ratios__short_excess` | `(CV_3-CV_12)/(CV_12+eps)` |
| `window_volatility_ratios__regime_flag` | `1 если CV_3 > 2·CV_12` |

**Интерпретация:** `regime_flag=1` — экстремальный краткосрочный режим нестабильности (CV_3 более чем вдвое выше CV_12).

**Пример (ряд `[20,20,20,10,40,10]`):** `CV_3=0.707, CV_6=0.5` → `cv_ratio_w3_w6=1.414`, `short_excess=(0.707-0.5)/0.5=0.414`.

---

## 23. `cross_window_momentum` — зум-спектр средних

**Файл:** `cross_window_momentum.py`

**Сигнал:** иерархия соотношений «текущее к краткосрочному → краткосрочное к среднесрочному → среднесрочное к долгосрочному» — «зум-спектр» ускорения.

| Колонка | Формула |
|---------|---------|
| `cross_window_momentum__ratio_w1_w3` | `v[t] / (\|mean_w3\|+eps)` |
| `cross_window_momentum__ratio_w3_w6` | `mean_w3 / (\|mean_w6\|+eps)` |
| `cross_window_momentum__ratio_w6_w24` | `mean_w6 / (\|mean_w24\|+eps)` |
| `cross_window_momentum__all_accel` | Флаг: все три ratio > 1 |
| `cross_window_momentum__all_decel` | Флаг: все три ratio < 1 |
| `cross_window_momentum__horizon_spread` | `ln(\|mean_w3\| / \|mean_w24\|)` |

**Интерпретация:** `all_accel=1` — «бычья» структура на всех горизонтах.

**Пример (ряд `[10,20,30,40,50,60]`):** `mean_w3=50, mean_w6=mean_w24=35, v[t]=60` → `ratio_w1_w3=1.2`, `ratio_w3_w6=1.429`, `ratio_w6_w24=1.0`, `horizon_spread=0.357`.

---

## 24. `quantile_persistence` — устойчивость квантильной позиции

**Файл:** `quantile_persistence.py`

**Сигнал:** стабильно ли клиент удерживает «хорошие»/«плохие» позиции в своей истории.

| Колонка | Формула |
|---------|---------|
| `quantile_persistence__above_med_w12` | `count(v[i] > median) / ws` |
| `quantile_persistence__top_q_w12`, `__bot_q_w12` | Доля месяцев в верхнем/нижнем квартиле |
| `quantile_persistence__rank_trend_w12` | OLS slope перцентильных рангов последней половины окна |
| `quantile_persistence__q_stability_w12` | `1 - CV(rank[i])` |
| `quantile_persistence__above_ewma_w12` | Доля месяцев выше текущего EWMA(alpha=0.3) |

**Интерпретация:** `top_q=0.9, q_stability>0.8` — стабильно в верхнем квартиле весь год.

**Пример (ряд `[10,20,30,40,50,60]`, w=6):** `median=35` → выше медианы `40,50,60` → `above_med_w6=0.5`; ранги растут монотонно → `rank_trend_w6 > 0`.

---

## 25. `lifecycle_phase` — жизненный цикл клиента

**Файл:** `lifecycle_phase.py`

**Сигнал:** фаза жизненного цикла — разгон (не достиг пика), зрелость (≥80% пика), снижение (ниже пика).

| Колонка | Формула |
|---------|---------|
| `lifecycle_phase__peak_age_share` | `pos_at_peak / (current_pos+1)` |
| `lifecycle_phase__post_peak_share` | `1 - peak_age_share` |
| `lifecycle_phase__completeness` | `safe_ratio(v[t], running_max)` |
| `lifecycle_phase__ramp_norm` | Нормированное время разгона до половины исторического максимума |
| `lifecycle_phase__is_new_peak` | Флаг нового исторического пика |
| `lifecycle_phase__phase_flag` | `0` разгон / `1` зрелость (`v >= 0.8·max`) / `2` снижение |
| `lifecycle_phase__post_peak_slope_w12` | Скорость снижения от пика |

**Интерпретация:** `phase_flag=2, post_peak_share=0.6` — на спаде уже 60% всей истории.

**Пример (ряд `[10,20,30,40,50,60]`):** `running_max=60` на `pos=5` → `completeness=1.0` (≥0.8 → зрелость), `peak_age_share=5/6=0.833`, `is_new_peak=1`.

---

## 26. `mean_deviation_shape` — асимметрия отклонений

**Файл:** `mean_deviation_shape.py`

**Сигнал:** больше ли случается крупных подъёмов (`σ_up > σ_down`) или глубоких провалов; как часто ряд пересекает своё среднее.

| Колонка | Формула |
|---------|---------|
| `mean_deviation_shape__up_semi_w6/12` | `sqrt(mean((v-mean)² для v>=mean))` |
| `mean_deviation_shape__down_semi_w6/12` | `sqrt(mean((mean-v)² для v<mean))` |
| `mean_deviation_shape__semi_ratio_w6/12` | `up_semi / (down_semi+eps)` |
| `mean_deviation_shape__max_up_z_w6/12`, `__max_down_z_w6/12` | Лучший/худший месяц в единицах σ |
| `mean_deviation_shape__dev_asym_w6/12` | `Σmax(0,v-mean)/(Σ\|v-mean\|+eps) - 0.5` |
| `mean_deviation_shape__cross_count_w6/12` | Число пересечений среднего |

**Интерпретация:** `semi_ratio > 1` — «приятные сюрпризы» сильнее «неприятных».

**Пример (ряд `[10,10,10,10,10,40]`, w=6):** `mean=15` → `up_semi=25, down_semi=5` → `semi_ratio_w6 = 5.0`.

---

## 27. `microstructure` — предсказуемость текущего месяца

**Файл:** `microstructure.py`

**Сигнал:** насколько «типичен» текущий месяц для клиента.

| Колонка | Формула |
|---------|---------|
| `microstructure__snr_w12` | `std_short / (std_long+eps)`, short_w = `max(w//4,1)` |
| `microstructure__surprise_w12` | `\|v[t]-mean_w\| / (std_w+eps)` |
| `microstructure__predictability_w12` | `1 / (1 + CV_w)` ∈ (0,1] |
| `microstructure__cond_mean_w12` | `mean_w / (active_rate+eps)` |
| `microstructure__vs_cond_mean_w12` | `v[t] / (\|cond_mean\|+eps)` |
| `microstructure__surprise_dir` | `sign(v[t]-mean_w)` |

**Интерпретация:** `predictability ≈ 0.99` — почти идеально предсказуемый клиент; `≈ 0.40` — высокая нестабильность.

**Пример (ряд `[20,20,20,20,20,30]`, w=6):** `mean=21.667, std=3.727` → `surprise_w6=2.236`, `predictability_w6=0.853`.

---

## 28. `plateau` — стагнация и плоские периоды

**Файл:** `plateau.py`

**Сигнал:** доля «плоских» шагов (`|Δ| < 5% локального среднего`), длина плато, давность выхода.

| Колонка | Формула |
|---------|---------|
| `plateau__flat_share_w6/12` | `count(is_flat) / (ws-1)` |
| `plateau__longest_flat_w6/12` | Наибольшая непрерывная плоская серия |
| `plateau__near_mean_w6/12` | `count(\|v-mean\| < 0.1·\|mean\|) / ws` |
| `plateau__current_flat_streak` | Running: текущая непрерывная плоская серия |
| `plateau__plateau_exit_recency` | Running: месяцев с последнего выхода из плато (`-1` — плато ещё не завершалось) |

**Интерпретация:** `flat_share_w12=1.0` — полное плато за весь год.

**Пример (ряд `[100,101,100,101,100,101]`, w=6):** каждый шаг `|Δ|=1 < 5%·~100` → все 5 шагов плоские → `flat_share_w6=1.0, longest_flat_w6=5`.

---

## 29. `extreme_events` — всплески и обвалы

**Файл:** `extreme_events.py`

**Сигнал:** редкие аномальные месяцы — всплески (`z > 2σ`) и обвалы (падение > 50% MoM).

| Колонка | Формула |
|---------|---------|
| `extreme_events__spike_count_w12` | `count(z[i] > 2)` |
| `extreme_events__max_spike_z_w12` | `max(z[i])` |
| `extreme_events__crash_count_w12` | `count(относительное_падение > 0.5)` |
| `extreme_events__max_drop_w12` | `max(относительное_падение)` |
| `extreme_events__recency_w12` | Месяцев с последнего экстремума |
| `extreme_events__balance_w12` | `spike_count - crash_count` |
| `extreme_events__is_spike_now` | Флаг текущего всплеска |

**Интерпретация:** `spike_count=1, max_spike_z=3.5` — один мощный выброс за год.

**Пример (ряд `[10,10,10,10,10,10,100]`, w=7):** `mean=22.857, std=31.493` → `z=2.449 > 2` → `spike_count_w7=1, is_spike_now=1, recency_w7=0`.

---

## 30. `value_clustering` — концентрация ценности в нескольких месяцах

**Файл:** `value_clustering.py`

**Сигнал:** доля оборота, приходящаяся на топ-1/топ-3 месяца; индекс Херфиндаля.

| Колонка | Формула |
|---------|---------|
| `value_clustering__top1_share_w12` | `safe_ratio(max_v, total_w)` |
| `value_clustering__top3_share_w12`, `__bot3_share_w12` | Доли топ-3/нижних-3 значений в сумме |
| `value_clustering__concentration_w12` | `safe_ratio(top3_sum, bot3_sum)` |
| `value_clustering__density_w12` | `safe_ratio(total_w, n_active·max_v)` |
| `value_clustering__herfindahl_w12` | `Σ(v_i/total)²` |

**Интерпретация:** `herfindahl ≈ 1/12` — идеально равномерное распределение по месяцам; `> 0.25` — сильная концентрация.

**Пример (ряд `[10,10,10,10,10,50]`, w=6):** `total=100` → `top1_share_w6=0.5`, `herfindahl_w6 = 5·0.01+0.25 = 0.3`.

---

## 31. `zero_clustering` — паттерны нулей внутри окна

**Файл:** `zero_clustering.py`

**Сигнал:** архитектура нулей — слиплись в длинную серию или рассеяны по окну; давность последнего нуля.

| Колонка | Формула |
|---------|---------|
| `zero_clustering__max_zero_run_w12` | Длиннейшая непрерывная серия нулей |
| `zero_clustering__zero_run_count_w12` | Число отдельных нулевых серий |
| `zero_clustering__recent_vs_long_w12` | `zero_share(последние 3)/(zero_share_w12+eps)` |
| `zero_clustering__last_zero_rec_w12` | Месяцев с последнего нуля (`0` = сейчас, `w` = нулей не было) |
| `zero_clustering__front_back_w12` | `zero_share(первая половина)/(zero_share(вторая половина)+eps)` |
| `zero_clustering__zero_after_active` | Флаг: прошлый месяц активен, текущий — ноль |

**Интерпретация:** `max_zero_run=6` — клиент полгода не совершал транзакций.

**Пример (ряд `[10,0,0,10,0,10]`, w=6):** серии нулей `idx1-2` (длина 2) и `idx4` (длина 1) → `max_zero_run_w6=2, zero_run_count_w6=2, last_zero_rec_w6=1`.

---

## Итоговая таблица

Число выходных колонок приведено для референсной конфигурации (все 81 трансформер, параметры как в секциях `Preset` выше) — это не пресет, который реально поставляется библиотекой (см. CLAUDE.md → Preset system: единственный пресет в `presets/` сейчас — `minimum.yaml`, только `slope`). Любой реальный пресет с другим набором трансформеров/параметров даёт другое число.

| Группа | Трансформеры | Колонок |
|--------|--------------|--------:|
| Простая статистика | `window_mean`, `window_median` | 6 |
| Тренд | `slope`, `slope_ratio`, `momentum`, `direction_flag`, `max_abs_jump`, `streak`, `growth_since_start` | 14 |
| Волатильность | `rolling_std`, `rolling_cv`, `rolling_min_max`, `extreme_share`, `skew_proxy` | 16 |
| Стаж и активность | `tenure`, `recency`, `inactive_streak`, `active_months`, `active_run_count`, `activity_rate`, `longest_active_run`, `client_age`, `zero_share` | 18 |
| Динамика MoM | `ewma`, `lag1_diff`, `mean_median_gap`, `recent_share`, `rolling_sum`, `zscore` | 15 |
| Изменения тренда | `accel`, `max_drawdown`, `peak_trough_timing`, `run_above_mean`, `sign_change_count`, `time_weighted_momentum`, `trend_flip`, `volatility_of_diff` | 18 |
| Положение в истории | `cumulative_share`, `distance_to_global_max`, `half_ratio`, `lag_growth_ratio`, `level_ratio`, `local_extrema`, `log1p_level`, `pct_of_max`, `rank_in_window`, `trough_to_current`, `volatility_trend` | 22 |
| Структурные сигналы | `corr_with_time`, `cusum`, `entropy`, `gini` | 10 |
| Автокорреляция | `autocorr` | 6 |
| Сезонность | `seasonal_autocorr` | 7 |
| Лог-рост | `geometric_return`, `log_level`, `log_slope`, `log_slope_ratio`, `log_volatility` | 10 |
| Шероховатость | `alternation_rate`, `roughness_ratio`, `total_variation` | 11 |
| Форма распределения | `kurtosis_proxy` | 10 |
| Пульсирующее поведение | `burstiness` | 7 |
| Регулярность потока | `flow_regularity` | 6 |
| Восстановление | `recovery_dynamics` | 6 |
| Качество роста | `growth_quality` | 6 |
| Лаговые сравнения | `lag_comparison` | 6 |
| Смена режима | `regime_change` | 6 |
| Нелинейность | `nonlinearity` | 10 |
| Чистота тренда | `trend_consistency` | 10 |
| Эволюция волатильности | `window_volatility_ratios` | 6 |
| Зум-спектр средних | `cross_window_momentum` | 6 |
| Квантильная устойчивость | `quantile_persistence` | 6 |
| Жизненный цикл | `lifecycle_phase` | 7 |
| Асимметрия отклонений | `mean_deviation_shape` | 14 |
| Предсказуемость | `microstructure` | 6 |
| Стагнация | `plateau` | 8 |
| Экстремальные события | `extreme_events` | 7 |
| Концентрация ценности | `value_clustering` | 6 |
| Паттерны нулей | `zero_clustering` | 6 |
| **Итого (81 трансформер)** | | **292** |

---

## Технические замечания

### Обработка коротких историй

`resolve_window_size(pos, w) = w if (pos+1) >= w else (pos+1)` — все признаки корректны уже с первого наблюдения (`pos=0`); при `pos < w-1` окно сужается до фактически доступной глубины. Признаки на лагах `k` (например, `lag_growth_ratio`, `autocorr`) равны нулю при `pos < k` — это конвенция «недостаточно истории», а не легитимный нулевой сигнал.

### Running state vs windowed

Часть трансформеров — **running state**: значение накапливается с начала истории сущности и сбрасывается только при `position_within_entity == 0` (новая сущность). Сюда относятся: `tenure`, `recency`, `inactive_streak`, `activity_rate`, `growth_since_start`, `cumulative_share`, `distance_to_global_max`, `streak`, `run_above_mean`, `lifecycle_phase` (частично), `plateau__current_flat_streak` / `__plateau_exit_recency`, `client_age`, `autocorr`/`seasonal_autocorr` (expanding-часть). Остальные — **windowed**: пересчитываются заново на каждом шаге по последним `w` значениям и не зависят от истории за пределами окна.

### Единицы измерения

Большинство признаков безразмерны или нормированы (ratio/CV/gini/entropy и т.п.) — не требуют масштабирования перед линейными моделями. Исключения — признаки в единицах исходной product-колонки: `rolling_sum`, `rolling_std`, `rolling_min_max__min/max`, `cusum__pos/neg`, `total_variation` (без `norm_` версии), `max_abs_jump`, `volatility_of_diff`, `time_weighted_momentum` — если денежные суммы клиентов сильно различаются по масштабу, эти признаки стоит логарифмировать/нормировать отдельно.

### Пресеты

Параметры (`windows`/`pairs`/`lags`/`half_windows`/`alphas` и т.д.) вынесены из кода в `ml_toolkit/transformers/presets/*.yaml`. `monthly.yaml` — полный набор (использован для чисел в таблице выше); `descriptive.yaml`, `trend.yaml`, `stability.yaml`, `lifecycle.yaml`, `midterm.yaml` — более узкие подмножества под конкретные задачи. См. раздел «Preset system» в `CLAUDE.md`.
