# Генетический алгоритм отбора признаков (`genetic.py`)

Точка входа — `select_features_genetic()`. Модель не зашита: GA принимает произвольный `scorer` callable — замыкание, которое обучает нужную модель и возвращает float для минимизации. Для CatBoost есть готовая фабрика `make_catboost_scorer`.

---

## Представление задачи

Каждая **особь** — бинарный вектор длиной `N` (число всех признаков). `1` означает «признак включён», `0` — «выброшен». Задача — найти особь с минимальным значением `scorer` на валидации.

---

## Scorer-интерфейс

```python
ScorerFn = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float]
```

GA нарезает `X_train` и `X_valid` по выбранным признакам и вызывает:

```python
score = scorer(X_train[selected], y_train, X_valid[selected], y_valid)
```

Scorer должен вернуть **float для минимизации**. Для метрик «выше — лучше» нужно вернуть отрицательное значение.

### Примеры scorer-замыканий

**LightGBM, pr_auc:**
```python
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

def lgbm_scorer(X_tr, y_tr, X_va, y_va):
    model = LGBMClassifier(n_estimators=300, verbose=-1).fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)[:, 1]
    return -average_precision_score(y_va, proba)
```

**sklearn RandomForest, mae:**
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def rf_scorer(X_tr, y_tr, X_va, y_va):
    model = RandomForestRegressor(n_estimators=100).fit(X_tr, y_tr)
    return float(np.abs(model.predict(X_va) - y_va).mean())
```

**CatBoost через фабрику:**
```python
from ml_toolkit.feature_selection import make_catboost_scorer

scorer = make_catboost_scorer(
    task='classification',
    metric='pr_auc',
    model_params={'iterations': 300, 'verbose': 0},
    cat_features=['region'],
    baseline_train=X_train['fee_nds_amount'],
    baseline_valid=X_valid['fee_nds_amount'],
)
```

---

## `make_catboost_scorer` — фабрика CatBoost-скорера

```python
make_catboost_scorer(
    task,           # 'classification' | 'regression'
    metric,         # строка или callable(y_true, y_score) -> float
    model_params,   # dict для CatBoostClassifier / CatBoostRegressor
    cat_features=None,       # список категориальных признаков
    baseline_train=None,     # pd.Series / np.ndarray — смещение предиктов
    baseline_valid=None,
    postprocess_fn=None,     # callable(pred: np.ndarray) -> np.ndarray
) -> ScorerFn
```

`baseline_train` / `baseline_valid` задаются как предвычисленные массивы (например `X_train['fee_nds_amount']`), а не именем колонки — CatBoost использует их как фиксированное смещение при residual learning. `postprocess_fn` применяется к предиктам регрессора до расчёта метрики; если нужны доп. колонки — замкните их снаружи.

Строковые метрики (регрессия): `'mae'`, `'rmse'`, `'median_ae'`, `'mape'`, `'smape'`, `'r2'`. Строковые метрики (классификация): `'pr_auc'`, `'roc_auc'`, `'f1'`, `'balanced_accuracy'`, `'logloss'`, `'brier'`, `'mcc'`, `'accuracy'`.

Callable-метрика: `fn(y_true: np.ndarray, y_score: np.ndarray) -> float` — по конвенции возвращает значение для минимизации.

### Примеры пользовательских метрик (callable)

**MedianAPE** (регрессия, минимизируется):
```python
def median_ape(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100)

scorer = make_catboost_scorer(task='regression', metric=median_ape, ...)
```

**Precision@k** (классификация, максимизируется → нужен минус):
```python
def precision_at_k(k):
    def _metric(y_true, proba):
        top_k = np.argsort(proba)[-k:]
        return -float(y_true[top_k].mean())
    return _metric

scorer = make_catboost_scorer(task='classification', metric=precision_at_k(10), ...)
```

---

## Инициализация популяции

Способ задаётся **ровно одним** из двух взаимоисключающих ключей в `gen_params`:

| Ключ | Тип | Поведение |
|---|---|---|
| `start_probability_to_include_feature` | `float` | Каждый бит включается независимо с этой вероятностью. |
| `start_n_features` | `int` | Ровно N признаков выбираются случайно (без возврата). |

Если в `gen_params` задан `must_be_included` — соответствующие позиции принудительно выставляются в `1` после генерации. После любого способа вызывается `repair_individual`.

### Валидация `gen_params`

| Условие | Реакция |
|---|---|
| `n_features_without_penalty > max_features` | `ValueError` |
| `len(must_be_included) > max_features` | `ValueError` |
| Индекс в `must_be_included` вне `[0, N-1]` | `ValueError` |
| Имя в `must_be_included` не в `feature_names` | `logger.warning`, пропуск |
| `population_size < 2` / `n_generations < 1` / `max_features < 1` | `ValueError` |
| Оба / ни одного ключа инициализации популяции | `ValueError` |

---

## Функция оценки (`_compute_fitness`)

1. Нарезает `X_train[selected]` и `X_valid[selected]`.
2. Вызывает `scorer(X_train_sel, y_train, X_valid_sel, y_valid)` → `score`.
3. Добавляет штраф за лишние признаки:

```
extra = max(0, n_selected - n_features_without_penalty)
score >= 0:  score *= (1 + extra * penalty_for_extra_feature)
score <  0:  score *= (1 - extra * penalty_for_extra_feature)
```

Знак автоматически определяет правильное направление штрафа — независимо от модели.

---

## Генетические операторы

**Скрещивание (`cx_uniform_with_repair`)** — uniform crossover: каждый бит с вероятностью `0.5` обменивается между двумя родителями. После — `repair_individual`.

**Мутация (`mut_flip_bit_with_repair`)** — каждый бит флипается с вероятностью `indpb=0.05`. После — `repair_individual`.

**Репарация (`repair_individual`)** — гарантирует валидность особи:
- если выбрано `0` признаков — добавляется один случайный;
- если выбрано больше `max_features` — случайные лишние удаляются (must-признаки защищены).

**Отбор** — турнирный (`tournsize=3`).

---

## Цикл эволюции

```
for gen in 1..n_generations:
    offspring = varAnd(pop, cross_probability, mutation_probability)
    оценить каждую особь в offspring (кэш по хромосоме)
    обновить best_overall (HallOfFame)
    elite = копия best_overall
    pop = [elite] + tournament_select(offspring, k=len(pop) - 1)
    проверить early stopping
```

`best_overall` хранит **абсолютно лучшую особь за всю историю** — защита от деградации популяции в поздних поколениях.

---

## Early stopping

| Ключ | Тип | Семантика |
|---|---|---|
| `min_improvement` | float | Абсолютное улучшение за `n_epoch_for_min_improvement` поколений. |
| `min_improvement_in_percents` | float | Процент от текущего значения. |
| `n_epoch_for_min_improvement` | int | Окно сравнения (default 1). |

Оба ключа одновременно не допускаются.

---

## Продакшн-параметры (пример)

| Параметр | Регрессия | Классификация |
|---|---|---|
| `population_size` | 40 | 20 |
| `n_generations` | 10 | 10 |
| `max_features` | 30 | 30 |
| `n_features_without_penalty` | 20 | 20 |
| `penalty_for_extra_feature` | 0.005 | 0.005 |
| `start_probability_to_include_feature` | 0.1 | 0.1 |
| `must_be_included` | `'fee_nds_amount'` | `'fee_nds_amount'` |
| Early stopping | 1% за 1 эпоху | 0.002 за 2 эпохи |
