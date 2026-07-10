# Настройка моделей через model_settings

Все параметры передаются через словарь `model_settings`, который при создании адаптера передаётся в конструктор `XxxRegressor(model_settings=...)`/`XxxClassifier(model_settings=...)`. Дополнительные параметры специфичны для конкретной модели (см. `supported_models.md`); здесь описаны **общие параметры**, работающие во всех адаптерах.

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

Имя столбца-бейзлайна для адаптеров, поддерживающих residual learning. Передаётся через `model_settings`, а не как отдельный аргумент. `ml_toolkit` не хардкодит имя колонки — **дефолт всегда `None` (бейзлайн не используется)**, пока не передано явно.

```python
model_settings = {
    'name': 'catboost',
    'baseline_col': 'my_baseline_column',
}
```

| Адаптер | Использование | Если столбца нет в `X` |
|---------|---------------|------------------------|
| `catboost` | CatBoost Pool `baseline=` | бейзлайн не применяется (`baseline=None`) |
| `lightgbm` | Residual learning: `y - baseline`, затем `pred + baseline` | обучение идёт на `y` напрямую |
| `xgboost` | Residual learning: `y - baseline`, затем `pred + baseline` (тот же контракт, что у `lightgbm`) | обучение идёт на `y` напрямую |
| `lama` | Добавляется к признакам (только регрессия) | бейзлайн не добавляется |
| `linear` (регрессия) | Добавляется к числовым признакам | бейзлайн не добавляется |
| Остальные | Игнорируется | — |

Для `catboost`/`lightgbm`/`xgboost` `baseline_col` работает и внутри самого Optuna-тюнинга
(`params=None`): каждый trial оценивается на предсказаниях с уже прибавленным baseline
(и уже применённым `postprocess_fn`, если задан) — гиперпараметры подбираются под финальную,
а не промежуточную метрику.

Конкретное имя столбца (например, `'fee_nds_amount'`) и логика его автоматического выбора — забота вызывающего бизнес-пайплайна (например `auto_kkp_classification`), а не `ml_toolkit`.

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

## Урезание мажоритарного класса внутри Optuna (`undersample_majority`)

Классификаторы `catboost`/`lightgbm`/`xgboost` умеют урезать классы внутри Optuna-тюнинга
(бинарный случай — `majority_fraction`, мультикласс, все три адаптера, — `balance_fraction`).
Финальная модель всегда обучается на том же сэмпле, что и лучший trial (не на полных данных) —
иначе гиперпараметры оценивались бы на одном объёме данных, а обучение шло бы на другом.
По умолчанию отключено — Optuna тюнит гиперпараметры на полных данных.

```python
model_settings = {'name': 'catboost', 'undersample_majority': True}   # включить сэмплирование в Optuna-триалах
```

| Адаптер | По умолчанию | Примечание |
|---------|--------------|------------|
| `catboost` | `False` | поддерживает и бинарную, и мультикласс классификацию |
| `lightgbm` | `False` | при `True` `is_unbalance` автоматически выключается (не комбинируется с сэмплированием) |
| `xgboost` | `False` | поддерживает и бинарную, и мультикласс классификацию |

---

## Ограничение времени тюнинга и прунинг (`optuna_timeout` / `optuna_pruner` / `optuna_verbose`)

Все адаптеры, использующие Optuna (`catboost`, `lightgbm`, `xgboost` — включая `*_ranker`-варианты
— `tabm`, а также все sklearn-подобные адаптеры без staged-обучения: `random_forest`, `extra_trees`,
`hist_gbm`, `quantile_forest`, `oblique_forest`, `mondrian`, `decision_tree`, `linear_tree`, `ebm`,
`pygam`, `mars`, `rulefit`, `figs`, `skope_rules`, `brl`, `ripper`, `soft_decision_tree`,
`locally_linear_forest`, `gaminet`, линейные модели) читают эти ключи из `model_settings`.

```python
model_settings = {
    'name': 'catboost',
    'optuna_timeout': 600,       # секунд на весь study.optimize; None (по умолч.) — без лимита
    'optuna_pruner': 'hyperband',
    'optuna_verbose': False,     # True — не форсировать WARNING-уровень логов Optuna
}
```

### `optuna_timeout`

Секунды на весь `study.optimize(...)`. Останавливает тюнинг по первому из условий:
`n_optuna_trials` trials или истечение `optuna_timeout` — текущий trial всегда доучивается до
конца, обрезки посреди trial не бывает. `None` (по умолчанию) — только по числу trials.

### `optuna_pruner`

`None` (по умолч.) → `MedianPruner()`. Строковые алиасы: `'median'`, `'hyperband'`,
`'percentile'` (25-й перцентиль), `'successive_halving'`, `'none'` (отключает прунинг —
`NopPruner`). Либо готовый экземпляр `optuna.pruners.BasePruner`.

Прунер реально отсекает бесперспективные trials только там, где есть промежуточные отчёты о
качестве по ходу обучения одного trial:

| Адаптер | Прунинг по | Метрика отчёта |
|---------|-------------|----------------|
| `catboost`, `catboost_ranker` | итерациям бустинга | `eval_metric` (через колбэк, `after_iteration`) |
| `lightgbm`, `lightgbm_ranker` | итерациям бустинга | первая метрика `eval_set` |
| `xgboost`, `xgboost_ranker` | итерациям бустинга | первая метрика `eval_set` |
| `tabm` | эпохам | `reg_metric`/`cls_metric` на валидации |
| остальные (sklearn-подобные, без staged-обучения) | — | `optuna_pruner` принимается, но не подключается — прунинг не имеет смысла без промежуточных отчётов внутри trial |
| `lama` | не через Optuna | LAMA управляет тюнингом сама; см. `model_settings['timeout']` (сек, по умолч. `n_optuna_trials * 60`) |

### `optuna_verbose`

`False` (по умолч.) — форсирует `optuna.logging.WARNING` на время `fit()` (глушит INFO-логи по
каждому trial). `True` — не трогает текущий уровень логирования Optuna.

---

## Сводная таблица ключей

| Ключ | Тип | Где работает | По умолчанию |
|------|-----|--------------|--------------|
| `reg_metric` | `str \| callable \| (callable, str)` | все регрессоры | `'mae'` |
| `cls_metric` | `str \| callable \| (callable, str)` | все классификаторы | `'pr_auc'` |
| `reg_metric_direction` | `'minimize' \| 'maximize'` | регрессоры (при callable) | `'minimize'` |
| `cls_metric_direction` | `'minimize' \| 'maximize'` | классификаторы (при callable) | `'maximize'` |
| `cat_encoder` | `None \| str \| TransformerMixin` | все кроме нативных | `None` → ordinal |
| `baseline_col` | `str \| None` | catboost, lightgbm, xgboost, lama (regressor), linear (regressor) | `None` → бейзлайн не используется |
| `param_space` | `Callable[[optuna.Trial], dict] \| None` | catboost, lightgbm, xgboost | `None` → дефолтное пространство |
| `undersample_majority` | `bool` | catboost, lightgbm, xgboost (классификаторы) | `False` для всех трёх |
| `optuna_timeout` | `float \| None` (секунды) | все Optuna-адаптеры | `None` → без лимита времени |
| `optuna_pruner` | `None \| str \| optuna.pruners.BasePruner` | все Optuna-адаптеры (реально отсекает trials только в catboost/lightgbm/xgboost/*_ranker/tabm) | `None` → `MedianPruner()` |
| `optuna_verbose` | `bool` | все Optuna-адаптеры | `False` → форсирует WARNING-уровень логов Optuna |
