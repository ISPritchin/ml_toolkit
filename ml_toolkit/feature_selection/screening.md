# Многоступенчатый пре-фильтр признаков (`screening.py`)

`FeatureScreener` отсеивает заведомо бесполезные признаки до обучения модели. Цель — снизить число фич без потери качества: удалить пропуски, константы, дубликаты и слабосигнальные столбцы.

Импорт:

```python
from ml_toolkit.feature_selection import FeatureScreener
```

---

## Стадии фильтрации

Фильтры применяются **последовательно**. Признак выбывает на первом сработавшем фильтре — это его `removed_by` в отчёте.

| # | Причина | Условие удаления | Параметр |
|---|---|---|---|
| 1 | `high_null_rate` | доля пропусков > порога | `max_null_rate` (default 0.95) |
| 2 | `low_variance` | дисперсия < порога | `min_variance` (default 1e-5) |
| 3 | `quasi_constant` | доля доминирующего значения > порога | `max_quasi_constant_rate` (default 0.95) |
| 4 | `duplicate` | точный дубликат ранее принятого признака | `drop_duplicates=True` (default False) |
| 5 | `low_auc` | однофакторный ROC-AUC < порога | `min_univariate_auc` (default 0.52) |
| 6 | `low_mutual_info` | взаимная информация < порога | `min_mutual_info` (default None — отключён) |

**Стадии 1–3** работают без целевой переменной и почти не имеют риска ошибки: константный столбец никогда не несёт сигнала.

**Стадии 5–6** зависят от `y` и могут ошибочно удалить признак, у которого сигнал проявляется только в сочетании с другими (feature interaction). Подробнее — в разделе «Риски AUC-фильтра» ниже.

---

## Параметры

```python
FeatureScreener(
    max_null_rate: float = 0.95,          # стадия 1
    min_variance: float = 1e-5,           # стадия 2
    max_quasi_constant_rate: float = 0.95,# стадия 3
    min_univariate_auc: float = 0.52,     # стадия 5; 0.0 = отключить
    auc_subsample: int | None = None,     # подвыборка для ускорения AUC на больших данных
    min_mutual_info: float | None = None, # стадия 6; None = отключить
    drop_duplicates: bool = False,        # стадия 4
)
```

Чтобы **полностью отключить AUC-фильтр** (только структурная очистка):

```python
screener = FeatureScreener(min_univariate_auc=0.0)
```

---

## API

### `fit(X, y) → self`

Вычисляет статистики и применяет все активные фильтры. `X` — `pd.DataFrame`, `y` — бинарная целевая переменная (AUC и MI считаются для бинарной задачи; для регрессии/мультикласса бинаризуйте `y` вручную — см. раздел ниже).

### `selected_features_` (property)

Список имён признаков, прошедших все фильтры. Поднимает `RuntimeError` до вызова `fit()`.

### `transform(X) → pd.DataFrame`

Возвращает `X[selected_features_]`. Можно применять к `X_train`, `X_val`, `X_test` после одного `fit()` на обучающей выборке.

### `fit_transform(X, y) → pd.DataFrame`

Эквивалент `fit(X, y).transform(X)`.

### `report() → pd.DataFrame`

Полная таблица по каждому признаку. Индекс — имена столбцов.

| Колонка | Тип | Описание |
|---|---|---|
| `null_rate` | float | Доля строк с NaN |
| `variance` | float | Дисперсия (ddof=0) по ненулевым значениям |
| `quasi_constant_rate` | float | Доля доминирующего значения среди ненулевых |
| `univariate_auc` | float | max(AUC, 1−AUC) по ненулевым строкам |
| `mutual_info` | float | Взаимная информация (NaN если стадия 6 отключена) |
| `kept` | bool | True = признак принят |
| `removed_by` | str\|None | Причина удаления или None |

```python
report = screener.report()
# Все удалённые с причиной
report[~report["kept"]][["univariate_auc", "removed_by"]]
# Отсортировать по AUC
report.sort_values("univariate_auc", ascending=False)
```

### `removal_summary() → pd.DataFrame`

Агрегированная сводка: сколько признаков удалено каждым фильтром.

```
          причина  признаков
   high_null_rate          8
     low_variance         15
  quasi_constant           4
          low_auc         23
    ИТОГО удалено         50
```

Колонки: `причина`, `признаков`. Последняя строка — `'ИТОГО удалено'` (итог).

---

## Типичный сценарий использования

```python
screener = FeatureScreener(
    max_null_rate=0.90,
    max_quasi_constant_rate=0.95,
    min_univariate_auc=0.52,
)

# fit только на обучающей выборке
screener.fit(X_train, y_train)

print(screener.removal_summary())
print(f"Признаков осталось: {len(screener.selected_features_)}")

X_train_clean = screener.transform(X_train)
X_val_clean   = screener.transform(X_val)
X_test_clean  = screener.transform(X_test)
```

---

## Бинаризация целевой для регрессии и мультикласса

AUC-фильтр рассчитан на бинарную `y`. Для других задач нужна ручная бинаризация перед `fit()`:

```python
# Регрессия → выше медианы
y_bin = (y_train > y_train.median()).astype(int)

# Мультикласс → мода vs остальные
mode = y_train.mode().iloc[0]
y_bin = (y_train == mode).astype(int)

screener.fit(X_train, y_bin)

# Обучение модели — с оригинальным y_train
model.fit(X_train_clean, y_train)
```

---

## Риски AUC-фильтра

Однофакторный AUC не видит **синергий между признаками** — сигнал, который проявляется только в паре с другим признаком, может быть удалён как «шумовой».

Рекомендации по применению:

| Ситуация | Рекомендация |
|---|---|
| Данные с очевидным мусором (константы, пропуски, случайные поля) | Стадии 1–3 обязательно, AUC-фильтр опционально |
| Данные по природе высокоразмерные (взаимодействие признаков) | Только стадии 1–3; AUC-фильтр отключить |
| Нужен максимальный скор | Стадии 1–3 + важность признаков после первого обучения модели |
| Нужна скорость / production latency | Агрессивный скрининг со всеми стадиями |

Альтернатива AUC-фильтру для максимального качества — двухэтапный отбор через модель:

```python
# 1. Обучить на очищенных структурно данных (стадии 1–3)
screener = FeatureScreener(min_univariate_auc=0.0).fit_transform(X_train, y_bin)
model.fit(X_clean, y_train)

# 2. Убрать признаки с нулевой важностью
importances = pd.Series(model.feature_importances_, index=X_clean.columns)
keep = importances[importances > 0].index.tolist()
X_final = X_clean[keep]
```

---

## Детали реализации

**Дубликаты** (`drop_duplicates=True`) определяются через `X[surviving_cols].T.duplicated()` — транспонируем, чтобы pandas сравнивал строки (= столбцы оригинала). NaN-безопасно. Из пары сохраняется первый встреченный признак.

**AUC** считается как `max(auc, 1 − auc)` — признак с обратной корреляцией (`auc = 0.4`) не удаляется, его AUC приводится к `0.6`.

**auc_subsample** задаёт максимальное число строк при расчёте AUC. Полезно при `n > 100k` — снижает время `fit()` без существенной потери точности отбора.

**mutual_info** использует `sklearn.feature_selection.mutual_info_classif` с `random_state=42`. Требует бинарную `y`. Запускается после AUC-фильтра — удаляет признаки с достаточным AUC, но низкой MI (редкий случай).
