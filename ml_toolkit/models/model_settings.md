# Настройка моделей через model_settings

Все параметры передаются через словарь `model_settings`, который при обучении передаётся в `train_regression_model` и `train_classification_model`. Дополнительные параметры специфичны для конкретной модели (см. `supported_models.md`); здесь описаны **общие параметры**, работающие во всех адаптерах.

---

## Метрика Optuna (`reg_metric` / `cls_metric`)

По умолчанию регрессия оптимизирует **MAE**, классификация — **PR-AUC**. Оба можно переопределить.

### Именованные пресеты

```python
# Регрессия
model_settings = {'name': 'catboost', 'reg_metric': 'mae'}    # по умолчанию
model_settings = {'name': 'catboost', 'reg_metric': 'rmse'}
model_settings = {'name': 'catboost', 'reg_metric': 'mape'}
model_settings = {'name': 'catboost', 'reg_metric': 'smape'}

# Классификация
model_settings = {'name': 'catboost', 'cls_metric': 'pr_auc'}  # по умолчанию
model_settings = {'name': 'catboost', 'cls_metric': 'roc_auc'}
model_settings = {'name': 'catboost', 'cls_metric': 'f1'}
```

### Произвольная функция

```python
def my_metric(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))   # MedianAE

model_settings = {
    'name': 'lightgbm',
    'reg_metric': my_metric,
    'reg_metric_direction': 'minimize',   # направление для callable без кортежа
}
```

### Кортеж (функция, направление)

```python
from sklearn.metrics import r2_score

model_settings = {
    'name': 'xgboost',
    'reg_metric': (r2_score, 'maximize'),
}
```

---

## Параметризованные метрики

Для метрик, требующих дополнительного параметра, используются фабричные функции из `ml_toolkit.models._utils`.

### Классификация

```python
from ml_toolkit.models._utils import make_precision_at_k, make_recall_at_k

# precision@k — доля позитивных среди топ-k по скору
model_settings = {'name': 'catboost', 'cls_metric': make_precision_at_k(k=100)}

# k как доля выборки (топ 5%)
model_settings = {'name': 'catboost', 'cls_metric': make_precision_at_k(k=0.05)}

# recall@k
model_settings = {'name': 'lightgbm', 'cls_metric': make_recall_at_k(k=200)}
```

### Регрессия

```python
from ml_toolkit.models._utils import make_quantile_loss

# Pinball loss — оптимизирует конкретный квантиль
model_settings = {'name': 'catboost', 'reg_metric': make_quantile_loss(q=0.75)}
# q=0.5 близко к MAE; q>0.5 штрафует недооценку; q<0.5 — переоценку
```

### Параметр `k`

| Тип | Интерпретация | Пример |
|-----|---------------|--------|
| `int` | Абсолютное число объектов | `k=100` → топ-100 |
| `float ∈ (0, 1]` | Доля от размера выборки | `k=0.1` → топ-10% |

---

## Кодирование категориальных признаков (`cat_encoder`)

CatBoost и LightGBM поддерживают категориальные признаки нативно. Остальные адаптеры используют `cat_encoder` для кодирования.

### Пресеты

```python
# OrdinalEncoder — по умолчанию; безопасен для деревьев
model_settings = {'name': 'random_forest', 'cat_encoder': 'ordinal'}

# OneHotEncoder — рекомендуется для линейных моделей
model_settings = {'name': 'elasticnet', 'cat_encoder': 'onehot'}
```

### Произвольный sklearn-трансформер

```python
from sklearn.preprocessing import TargetEncoder

model_settings = {
    'name': 'hist_gbm',
    'cat_encoder': TargetEncoder(target_type='continuous'),
}
```

Трансформер всегда обучается на обучающей выборке, затем применяется к валидационной и инференс-выборкам.

### Поведение при OneHotEncoder

Столбцы с категориальными признаками заменяются на расширенные (`col__value`). Список `selected_features` автоматически обновляется внутри адаптера.

### Адаптеры со своим кодированием

| Адаптер | Кодирование |
|---------|-------------|
| `catboost` | CatBoost Pool (нативно) |
| `lightgbm` | `category` dtype (нативно) |
| `lama` | LAMA AutoML (внутри) |
| `tabm` | OrdinalEncoder в `_Preprocessor` |
| `gaminet` | только числовые признаки |

Для этих адаптеров `cat_encoder` игнорируется.

---

## Baseline (`baseline_col`)

Имя столбца-бейзлайна для адаптеров, поддерживающих residual learning. Передаётся через `model_settings`, а не как отдельный аргумент.

```python
model_settings = {
    'name': 'catboost',
    'baseline_col': 'fee_nds_amount',   # по умолчанию
}
```

| Адаптер | Использование |
|---------|---------------|
| `catboost` | CatBoost Pool `baseline=` |
| `lightgbm` | Residual learning: `y - baseline`, затем `pred + baseline` |
| `lama` | Добавляется к признакам если отсутствует |
| `linear` | Добавляется к числовым признакам если отсутствует |
| Остальные | Игнорируется |

В `run_inference.py` и `ml_toolkit/main.py` `baseline_col` вычисляется автоматически:
- `'fee_nds_amount_current_month'` для продуктов типа *current month*
- `'fee_nds_amount'` для остальных

---

## Своё пространство поиска Optuna (`param_space`)

По умолчанию `catboost`/`lightgbm`/`xgboost` (регрессоры и классификаторы) тюнят фиксированный
набор гиперпараметров, зашитый в адаптере. `param_space` подменяет его целиком.

```python
def my_space(trial):
    return {
        'iterations': trial.suggest_int('iterations', 200, 600, step=50),
        'depth': trial.suggest_int('depth', 4, 6),
    }

model_settings = {'name': 'catboost', 'param_space': my_space}
```

Правила:
- Сигнатура: `Callable[[optuna.Trial], dict]`. Возвращать нужно только тюнируемые параметры —
  служебные ключи (`loss_function`/`objective`, `eval_metric`, `verbose`, `random_seed`/`random_state`,
  `early_stopping_rounds`, `enable_categorical`) подставляются адаптером автоматически и имеют приоритет
  над одноимёнными ключами из `param_space`.
- Для `lightgbm` `param_space` может (но не обязан) вернуть `'boosting_type'` — если не передан,
  используется `'gbdt'`.
- Не применяется к рангерам (`*_ranker`) и остальным моделям — только `catboost`, `lightgbm`, `xgboost`.
- Работает независимо от `undersample_majority` — сэмплирование (если включено) применяется до
  обучения, `param_space` только определяет тюнируемые гиперпараметры.

---

## Отключение undersampling мажоритарного класса (`undersample_majority`)

Классификаторы `catboost`/`lightgbm`/`xgboost` умеют урезать мажоритарный класс внутри Optuna-тюнинга
(бинарный случай — `majority_fraction`, мультикласс, только `catboost`, — `balance_fraction`).
Финальная модель всегда обучается на том же сэмпле, что и лучший trial (не на полных данных) —
иначе гиперпараметры оценивались бы на одном объёме данных, а обучение шло бы на другом.

```python
model_settings = {'name': 'catboost', 'undersample_majority': False}   # без сэмплирования, всегда полные данные
```

| Адаптер | По умолчанию | Примечание |
|---------|--------------|------------|
| `catboost` | `True` | |
| `lightgbm` | `True` | при `True` `is_unbalance` автоматически выключается (не комбинируется с сэмплированием) |
| `xgboost` | `True` | поддерживает только бинарную классификацию |

---

## Сводная таблица ключей

| Ключ | Тип | Где работает | По умолчанию |
|------|-----|--------------|--------------|
| `reg_metric` | `str \| callable \| (callable, str)` | все регрессоры | `'mae'` |
| `cls_metric` | `str \| callable \| (callable, str)` | все классификаторы | `'pr_auc'` |
| `reg_metric_direction` | `'minimize' \| 'maximize'` | регрессоры (при callable) | `'minimize'` |
| `cls_metric_direction` | `'minimize' \| 'maximize'` | классификаторы (при callable) | `'maximize'` |
| `cat_encoder` | `None \| str \| TransformerMixin` | все кроме нативных | `None` → ordinal |
| `baseline_col` | `str` | catboost, lightgbm, lama, linear | `'fee_nds_amount'` |
| `param_space` | `Callable[[optuna.Trial], dict] \| None` | catboost, lightgbm, xgboost | `None` → дефолтное пространство |
| `undersample_majority` | `bool` | catboost, lightgbm, xgboost (классификаторы) | `True` (catboost) / `False` (lightgbm, xgboost) |
