# Drift-фильтр признаков (`drift_filter.py`)

Модуль решает задачу **train/valid distribution shift**: обнаруживает признаки, по которым распределение в обучающей и валидационной выборках значимо расходится, и удаляет их перед обучением модели.

Два независимых инструмента с разным соотношением скорость/точность:

| Инструмент | Подход | Требует модель | Когда использовать |
|---|---|---|---|
| `compute_psi` | Population Stability Index | Нет | Быстрая диагностика, большое число признаков |
| `AdversarialDriftFilter` | Adversarial validation | CatBoost | Точное устранение drift перед обучением |

Импорт:

```python
from ml_toolkit.feature_selection import AdversarialDriftFilter, compute_psi
# или напрямую:
from ml_toolkit.feature_selection.drift_filter import AdversarialDriftFilter, compute_psi
```

---

## `compute_psi` — Population Stability Index

### Что измеряет

PSI количественно выражает, насколько сильно изменилось распределение признака между двумя выборками. Формула для числового признака, разбитого на K бинов:

```
PSI = Σ (q_i − p_i) × ln(q_i / p_i)
```

где `p_i` — доля наблюдений train в бине `i`, `q_i` — доля valid.

Для **категориальных признаков** вместо бинов используются частоты уникальных значений.

### Интерпретация

| PSI | Статус | Что делать |
|---|---|---|
| < 0.10 | `stable` | Смещения нет, признак можно использовать |
| 0.10–0.25 | `moderate` | Умеренное смещение, наблюдать |
| > 0.25 | `high` | Критическое смещение, кандидат на удаление |

PSI симметрична по смыслу («насколько valid отличается от train»), но **не симметрична математически**: `PSI(train→valid) ≠ PSI(valid→train)`. Стандартное соглашение — считать относительно train.

### API

```python
compute_psi(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    n_bins: int = 10,
    min_bin_count: int = 5,
) -> pd.DataFrame
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `n_bins` | 10 | Число равноширинных бинов для числовых признаков. Границы строятся по train, применяются к valid. |
| `min_bin_count` | 5 | Бины с числом наблюдений < порога в любой из выборок исключаются из расчёта. Защита от нестабильных PSI на разреженных бинах. |

**Возвращает:** `pd.DataFrame` с колонками `feature`, `psi`, `drift_level`, отсортированный по убыванию PSI.

### Примеры

```python
from ml_toolkit.feature_selection import compute_psi

psi_report = compute_psi(X_train, X_valid)

# Только критические признаки
high_drift = psi_report[psi_report['drift_level'] == 'high']
print(high_drift)

# Топ-10 самых нестабильных
print(psi_report.head(10))

# Список признаков для исключения (PSI > 0.25)
drift_features = high_drift['feature'].tolist()
X_train_clean = X_train.drop(columns=drift_features)
```

### Ограничения PSI

PSI работает по одному признаку независимо — **не видит смещение в совместном распределении** нескольких признаков. Признак с PSI = 0.05 сам по себе стабилен, но в паре с другим может создавать drift, который PSI не поймает. Для таких случаев используйте `AdversarialDriftFilter`.

---

## `AdversarialDriftFilter` — Adversarial Validation

### Идея

Если модель умеет отличать train от valid лучше случайного угадывания (ROC-AUC > 0.5), значит, между выборками есть измеримое смещение. Признаки, которые модель использует для дискриминации — это и есть источники drift.

**Алгоритм:**

```
1. Объединяем X_train (y=0) и X_valid (y=1) → adversarial датасет.
2. Разбиваем 70/30 (stratified), обучаем CatBoost на 70%, оцениваем AUC на 30%.
3. Если AUC > target_auc:
       удаляем top-K признаков по adversarial feature importance
       переходим к шагу 2 на оставшихся признаках.
4. Останавливаемся при AUC ≤ target_auc или достижении лимита удалений.
```

Чем выше adversarial AUC — тем сильнее drift. После удаления признаков-виновников AUC должен упасть к ~0.5.

### Параметры

```python
AdversarialDriftFilter(
    target_auc: float = 0.55,
    max_features_to_drop: int | None = None,
    remove_per_step: int = 1,
    cat_features: list[str] | None = None,
    cb_iterations: int = 300,
    cb_max_depth: int = 4,
    cb_learning_rate: float = 0.05,
    random_seed: int = 42,
)
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `target_auc` | 0.55 | Целевой adversarial AUC. Итерации идут, пока AUC > target. Чем ближе к 0.5, тем жёстче фильтрация и больше признаков удаляется. |
| `max_features_to_drop` | None | Жёсткий лимит на число удаляемых признаков. None — без ограничения. Полезно, когда drift сильный, но нельзя терять слишком много сигнала. |
| `remove_per_step` | 1 | Сколько признаков удалять за одну итерацию. `1` — консервативно, точно. `3–5` — быстро, когда drift-признаков много. |
| `cat_features` | None | Категориальные признаки для CatBoost adversarial модели. |
| `cb_iterations` | 300 | Число итераций CatBoost. Небольшое значение предпочтительно — adversarial модель намеренно «слабая», чтобы не переобучиться на малом числе drift-признаков. |
| `cb_max_depth` | 4 | Глубина деревьев. Мелкие деревья менее склонны к переобучению, что важно для корректной оценки AUC. |
| `cb_learning_rate` | 0.05 | Learning rate CatBoost adversarial модели. |
| `random_seed` | 42 | Фиксирует результат train/test сплита и CatBoost. |

### API

#### `fit(X_train, X_valid) → self`

Запускает итеративное удаление drift-признаков. Принимает только X — целевая переменная `y` не нужна.

```python
adf = AdversarialDriftFilter(target_auc=0.55)
adf.fit(X_train, X_valid)
```

#### `transform(X) → pd.DataFrame`

Возвращает X с отобранными признаками. Применяется одинаково к X_train, X_valid и X_test.

```python
X_train_clean = adf.transform(X_train)
X_valid_clean = adf.transform(X_valid)
X_test_clean  = adf.transform(X_test)
```

#### `fit_transform(X_train, X_valid) → pd.DataFrame`

Эквивалент `fit(X_train, X_valid).transform(X_train)`.

#### `report() → pd.DataFrame`

Детальный отчёт по всем признакам из последнего adversarial run.

Колонки: `feature`, `adversarial_importance`, `removed_by_drift`.

```python
print(adf.report())
```

```
feature  adversarial_importance  removed_by_drift
     f0                   51.20              True
     f1                   23.25              True
     f3                   12.40             False
     f2                    8.10             False
```

### Атрибуты после `fit`

| Атрибут | Тип | Описание |
|---|---|---|
| `selected_features_` | `list[str]` | Признаки без значимого drift |
| `removed_features_` | `list[str]` | Удалённые признаки в порядке удаления |
| `adversarial_auc_history_` | `list[float]` | AUC после каждой итерации: `[auc_0, auc_1, ..., auc_final]` |
| `feature_importances_` | `pd.Series` | Adversarial важность признаков из последнего запуска CatBoost |

### Примеры

**Базовый сценарий:**

```python
from ml_toolkit.feature_selection import AdversarialDriftFilter

adf = AdversarialDriftFilter(target_auc=0.55)
adf.fit(X_train, X_valid)

print(f"Начальный adversarial AUC: {adf.adversarial_auc_history_[0]:.4f}")
print(f"Финальный adversarial AUC: {adf.adversarial_auc_history_[-1]:.4f}")
print(f"Удалено признаков: {len(adf.removed_features_)}: {adf.removed_features_}")

X_train_clean = adf.transform(X_train)
X_valid_clean = adf.transform(X_valid)
X_test_clean  = adf.transform(X_test)
```

**Ускоренный режим (много дрейфующих признаков):**

```python
adf = AdversarialDriftFilter(
    target_auc=0.55,
    remove_per_step=5,          # удалять по 5 признаков за итерацию
    max_features_to_drop=20,    # не удалять больше 20 признаков
)
adf.fit(X_train, X_valid)
```

**Жёсткая фильтрация (максимальное подавление drift):**

```python
adf = AdversarialDriftFilter(
    target_auc=0.50,    # целиться в случайное угадывание
    cb_iterations=500,  # более сильная adversarial модель
    cb_max_depth=5,
)
adf.fit(X_train, X_valid)
```

**Диагностика без удаления (1 итерация):**

```python
adf = AdversarialDriftFilter(max_features_to_drop=0)
adf.fit(X_train, X_valid)

print(f"Adversarial AUC = {adf.adversarial_auc_history_[0]:.4f}")
# AUC > 0.7 → серьёзный drift
# AUC < 0.55 → drift минимален

# Посмотреть, какие признаки дискриминируют train от valid
print(adf.report().head(10))
```

---

## Выбор инструмента

```
Есть подозрение на drift?
├── Нужна быстрая диагностика без CatBoost → compute_psi()
│       PSI > 0.25 у конкретных признаков → удалить вручную или передать в ADF
│
└── Нужно автоматическое устранение → AdversarialDriftFilter
        Много признаков (> 50) → remove_per_step=3..5
        Мало признаков / важен каждый → remove_per_step=1, max_features_to_drop=N
```

Оба инструмента хорошо сочетаются: PSI — для быстрого обзора, ADF — для точного удаления. Включить обе стадии в один пайплайн позволяет `FeatureSelectionPipeline` (см. `pipeline.py`).

---

## Теоретические основы

### Почему adversarial AUC — надёжный индикатор

Если признаки в train и valid имеют одинаковое распределение, классификатор «train vs valid» не может работать лучше случайного → AUC = 0.5. Любое устойчивое превышение над 0.5 означает обнаруживаемое смещение.

Adversarial validation обнаруживает **любой тип смещения**: сдвиг среднего (covariate shift), изменение дисперсии, появление новых категорий, временной дрейф — в отличие от PSI, который чувствителен только к маргинальным распределениям.

### Почему важность признаков, а не AUC по признаку

AUC отдельного признака показывает, насколько хорошо он разделяет классы **целевой задачи**. Adversarial importance показывает, насколько сильно именно этот признак **отличает train от valid**. Это разные вещи: признак с высоким AUC по целевой задаче может иметь нулевой drift, и наоборот.

### Ограничения

**Необходима валидационная выборка.** Метод требует X_valid с теми же признаками, что X_train. Если valid недоступен (например, test неразмечен), используйте `compute_psi` как альтернативу.

**Нестабильность при малых выборках.** При `len(X_valid) < 200` adversarial AUC оценивается нестабильно. Увеличьте `cb_iterations` и уменьшите `cb_max_depth` для сглаживания.

**Потеря сигнала.** Признак, вызывающий drift, может нести ценный предиктивный сигнал. Удаление устраняет проблему обобщения, но ценой части информации. В ситуации выбора «drift или сигнал» используйте `max_features_to_drop` как ограничение и проверяйте качество модели на задаче после удаления.

**Нестабильность feature importance между запусками.** При `remove_per_step=1` на каждой итерации сбрасывается adversarial модель и пересчитывается importance — порядок удаления детерминирован только при фиксированном `random_seed`.
