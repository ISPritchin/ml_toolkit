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

Для `catboost`/`lightgbm`/`xgboost` `baseline_col` работает и внутри самого Optuna-тюнинга (`params=None`): каждый trial оценивается на предсказаниях с уже прибавленным baseline (и уже применённым `postprocess_fn`, если задан) — гиперпараметры подбираются под финальную, а не промежуточную метрику.

Конкретное имя столбца (например, `'fee_nds_amount'`) и логика его автоматического выбора — забота вызывающего бизнес-пайплайна (например `auto_kkp_classification`), а не `ml_toolkit`.

---

## Своё пространство поиска Optuna (`param_space`)

По умолчанию `catboost`/`lightgbm`/`xgboost` (регрессоры и классификаторы) тюнят фиксированный набор гиперпараметров, зашитый в адаптере. `param_space` подменяет его целиком.

```python
def my_space(trial):
    return {
        'iterations': trial.suggest_int('iterations', 200, 600, step=50),
        'depth': trial.suggest_int('depth', 4, 6),
    }

model_settings = {'name': 'catboost', 'param_space': my_space}
```

Правила:
- Сигнатура: `Callable[[optuna.Trial], dict]`. Возвращать нужно только тюнируемые параметры — служебные ключи (`loss_function`/`objective`, `eval_metric`, `verbose`, `random_seed`/`random_state`, `early_stopping_rounds`, `enable_categorical`) подставляются адаптером автоматически и имеют приоритет над одноимёнными ключами из `param_space`.
- Для `lightgbm` `param_space` может (но не обязан) вернуть `'boosting_type'` — если не передан, используется `'gbdt'`.
- Не применяется к рангерам (`*_ranker`) и остальным моделям — только `catboost`, `lightgbm`, `xgboost`.
- Работает независимо от `undersample_majority` — сэмплирование (если включено) применяется до обучения, `param_space` только определяет тюнируемые гиперпараметры.

---

## Кастомный лосс с тюнингом параметров (`loss_spec`)

Единый механизм для `catboost`/`lightgbm`/`xgboost` (регрессоры и классификаторы, включая мультикласс) — подменяет TRAINING loss на кастомный (`calc_ders_range`/`calc_ders_multi`-совместимый класс из `ml_toolkit.losses` или `ml_toolkit.presets.*`) и тюнит его собственные параметры через Optuna вместе с гиперпараметрами модели. Реализация — `ml_toolkit/models/_loss_spec.py`.

```python
from ml_toolkit.losses import FocalLoss
from ml_toolkit.models._loss_spec import LossSpec

model_settings = {
    'loss_spec': LossSpec(
        name='focal',                                          # для логов/best_params_
        loss_cls=FocalLoss,                                    # calc_ders_range/calc_ders_multi-класс
        param_bounds={'gamma': (1.0, 5.0), 'alpha': (0.1, 0.9)},  # trial.suggest_float по каждому ключу
    ),
}
model = CatBoostClassifier(n_optuna_trials=50, model_settings=model_settings)   # то же — LightGBM*, XGBoost*, *Regressor
```

Правила:
- `loss_cls` определяется по наличию `calc_ders_multi` (мультикласс) vs `calc_ders_range` (бинарная классификация/регрессия) — `is_multiclass_loss`, никакого отдельного флага передавать не нужно.
- `param_bounds={}` — у лосса нет тюнящихся параметров, конструируется с дефолтами класса.
- После `fit()` `best_params_['loss_name']`/`best_params_['loss_params']` — человекочитаемые имя и фактически выбранные параметры лосса лучшего trial (сам объект-лосс в `best_params_[...]` под ключом `loss_function`/`objective` — живой, нужен для реконструкции модели, но для логов неинформативен).
- CatBoost принимает `calc_ders_range`/`calc_ders_multi`-объекты нативно; при кастомном `loss_function` CatBoost **требует** `eval_metric` явно и падает собственной `CatBoostError`, если `model_settings['eval_metric']` не задан (см. раздел «Метрика раннего останова» ниже — у `eval_metric` нет дефолта в адаптере); для мультикласса дополнительно передаётся `classes_count`.
- LightGBM/XGBoost (sklearn-обёртки) используют свою объектную сигнатуру (`objective(y_true, y_pred) -> grad, hess`, см. `_loss_spec.py`); гессиан мультикласса там — только диагональ (ограничение custom objective API обоих фреймворков, не этого модуля). **LightGBM и XGBoost ведут себя по-разному на `predict_proba()`** при кастомном objective: LightGBM документированно/эмпирически возвращает raw margins вместо вероятностей (с warning'ом) — адаптер применяет sigmoid/softmax вручную поверх raw score. **XGBoost, наоборот, уже возвращает настоящие вероятности сам** — Booster помнит `'multi:softprob'`/аналог для бинарного на уровне learner config независимо от того, что градиенты считала Python-функция (верифицировано вживую на xgboost==3.2.0 через `uv run --with xgboost`, т.к. пакет не входит в обязательные зависимости проекта) — адаптер **не** применяет к нему ничего дополнительно; более ранняя версия этого кода ошибочно копировала LightGBM-паттерн и для XGBoost тоже, что на практике ломало `predict_proba()` двойным sigmoid/softmax (см. `tests/models/_tabular/_boosting/test_xgboost.py::TestXGBoostLossSpec` — регрессионный тест на этот конкретный баг).
- Мультиклассовая форма `y_pred` в custom objective (2D `(n, n_classes)`, без флэттенинга) верифицирована вживую на xgboost==3.2.0.
- Несовместим с явным `loss_function`/`objective` в `param_space` — `loss_spec`, если задан, имеет приоритет (переопределяет его).

---

## Метрика раннего останова и Optuna-пруинга (`eval_metric`)

В тюнинге `catboost`/`lightgbm`/`xgboost` участвуют **три независимые функции**, которые не обязаны совпадать друг с другом:

1. **training loss** (`loss_function`/`objective`) — то, по чему считаются градиент/гессиан, то есть то, что реально растит каждое дерево. Кастомизируется через `model_settings['loss_spec']` (см. выше).
2. **`eval_metric`** (этот раздел) — считается на валидации **на каждой итерации бустинга внутри одного обучения**. Решает две вещи: когда остановить обучение (early stopping) и когда прибить бесперспективный Optuna-trial досрочно (pruning, через `make_catboost_pruning_callback`/`make_lgb_pruning_callback`/`make_xgb_pruning_callback` — все три берут «какая метрика пришла первой», без привязки к конкретному имени, так что кастомная метрика pruning не ломает).
3. **`reg_metric`/`cls_metric`** — считается один раз **после того как весь trial обучился**, на финальных предсказаниях; по ней Optuna сравнивает trial'ы и выбирает лучший (см. раздел «Метрика Optuna» выше). Уже полностью кастомизируема (строка/callable/`(callable, direction)`).

Рабочий пример, где все три разные: train на `FocalLoss` (`loss_spec`, градиенты под дисбаланс) → early stopping по `logloss` (`eval_metric`, стабильная метрика на итерацию) → лучший trial Optuna выбирает по `PR-AUC` (`cls_metric`, то, что важно бизнесу).

**У `eval_metric` нет дефолта в адаптере** (ни в одном из трёх фреймворков) — `model_settings.get('eval_metric')` без запасного значения; не задан → ключ просто не попадает в params, и framework сам решает:
- CatBoost: без кастомного `loss_function` — подставляет метрику, соответствующую `loss_function`, сам; с кастомным (`loss_spec`) — требует `eval_metric` явно и падает `CatBoostError`, если его нет (не маскируется).
- LightGBM: выводит метрику из строки `objective` (`'mae'` → `'l1'`, `'binary'` → `'binary_logloss'`, и т.п.); при `objective=callable` (т.е. `loss_spec` без явного `eval_metric`) молча откатывается на общий дефолт по типу задачи (`'l2'` для регрессии) — это поведение LightGBM, не ml_toolkit, и оно может быть не связано с тем, что реально тюнится кастомным лоссом. Указывайте `eval_metric` явно при использовании `loss_spec`.
- XGBoost: аналогично LightGBM — выводит метрику из `objective`, если `eval_metric` не задан явно (верифицировано вживую на xgboost==3.2.0). **Важно про направление**: если `eval_metric` — строка, XGBoost сам верно определяет maximize/minimize по встроенному вайтлисту префиксов имени (`'auc'`, `'aucpr'`, `'pre'`, `'map'`, `'ndcg'` → maximize, иначе minimize). Для callable этот же вайтлист матчится по имени ФУНКЦИИ — `sklearn.metrics.roc_auc_score` под него не попадает и тихо считается minimize, из-за чего early stopping останавливается на первой же итерации (`best_iteration=0`, полностью сломанная модель, без единой ошибки — проверено вживую). Поэтому **для XGBoost голый callable не принимается** — только `(callable, direction)` с явным `'minimize'`/`'maximize'` (см. пример ниже); строки по-прежнему можно передавать как есть.

```python
model_settings = {
    'loss_spec': LossSpec(name='focal', loss_cls=FocalLoss, param_bounds={'gamma': (1.0, 5.0)}),
    'eval_metric': 'Logloss',   # CatBoost: обязателен при кастомном loss_function
}
```

Можно передать и callable (не только строку) — сигнатура и способ указать направление зависят от фреймворка:
- CatBoost: собственный объект метрики (`is_max_optimal()`/`evaluate(approxes, target, weight)`/`get_final_error(error, weight)`) — направление уже внутри объекта, передаётся как есть.
- LightGBM: `(y_true, y_pred[, weight[, group]]) -> (name, value, is_higher_better)` — направление явно в возвращаемом кортеже, передаётся как есть. Уходит в `.fit(eval_metric=...)` (не в конструктор — LightGBM принимает callable только там); адаптер дополнительно ставит `'metric': 'None'` в конструкторе, чтобы подавить автоматический вывод LightGBM — иначе он добавился бы ВТОРЫМ метрикой рядом с вашей, и pruning-колбэк (берёт метрику с индексом 0) мог бы молча использовать не ту.
- XGBoost: **только `(callable, direction)`**, `direction` ∈ `{'minimize', 'maximize'}`, сам `callable` — обычная sklearn-метрика `(y_true, y_pred) -> float` (например, `sklearn.metrics.roc_auc_score`, `mean_absolute_error` как есть, без обёрток). Голый callable (без направления) отклоняется `ValueError` — см. предупреждение выше про то, почему угадать direction по имени функции нельзя. Реализовано через явный `xgboost.callback.EarlyStopping(maximize=...)` вместо встроенного шортката `early_stopping_rounds=<int>`.

```python
# LightGBM
def custom_metric(y_true, y_pred):
    return 'custom', float(((y_true - y_pred) ** 2).mean()), False   # is_higher_better=False

model_settings = {'eval_metric': custom_metric}

# XGBoost — направление обязательно, функция возвращает голый float
from sklearn.metrics import roc_auc_score
model_settings = {'eval_metric': (roc_auc_score, 'maximize')}
```

---

## Урезание мажоритарного класса внутри Optuna (`undersample_majority`)

Классификаторы `catboost`/`lightgbm`/`xgboost` умеют урезать классы внутри Optuna-тюнинга (бинарный случай — `majority_fraction`, мультикласс, все три адаптера, — `balance_fraction`). Финальная модель всегда обучается на том же сэмпле, что и лучший trial (не на полных данных) — иначе гиперпараметры оценивались бы на одном объёме данных, а обучение шло бы на другом. По умолчанию отключено — Optuna тюнит гиперпараметры на полных данных.

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

Все адаптеры, использующие Optuna (`catboost`, `lightgbm`, `xgboost` — включая `*_ranker`-варианты — `tabm`, а также все sklearn-подобные адаптеры без staged-обучения: `random_forest`, `extra_trees`, `hist_gbm`, `quantile_forest`, `oblique_forest`, `mondrian`, `decision_tree`, `linear_tree`, `ebm`, `pygam`, `mars`, `rulefit`, `figs`, `skope_rules`, `brl`, `ripper`, `soft_decision_tree`, `locally_linear_forest`, `gaminet`, линейные модели) читают эти ключи из `model_settings`.

```python
model_settings = {
    'name': 'catboost',
    'optuna_timeout': 600,       # секунд на весь study.optimize; None (по умолч.) — без лимита
    'optuna_pruner': 'hyperband',
    'optuna_verbose': False,     # True — не форсировать WARNING-уровень логов Optuna
}
```

### `optuna_timeout`

Секунды на весь `study.optimize(...)`. Останавливает тюнинг по первому из условий: `n_optuna_trials` trials или истечение `optuna_timeout` — текущий trial всегда доучивается до конца, обрезки посреди trial не бывает. `None` (по умолчанию) — только по числу trials.

### `optuna_pruner`

`None` (по умолч.) → `MedianPruner()`. Строковые алиасы: `'median'`, `'hyperband'`, `'percentile'` (25-й перцентиль), `'successive_halving'`, `'none'` (отключает прунинг — `NopPruner`). Либо готовый экземпляр `optuna.pruners.BasePruner`.

Прунер реально отсекает бесперспективные trials только там, где есть промежуточные отчёты о качестве по ходу обучения одного trial:

| Адаптер | Прунинг по | Метрика отчёта |
|---------|-------------|----------------|
| `catboost`, `catboost_ranker` | итерациям бустинга | `eval_metric` (через колбэк, `after_iteration`) |
| `lightgbm`, `lightgbm_ranker` | итерациям бустинга | первая метрика `eval_set` |
| `xgboost`, `xgboost_ranker` | итерациям бустинга | первая метрика `eval_set` |
| `tabm` | эпохам | `reg_metric`/`cls_metric` на валидации |
| остальные (sklearn-подобные, без staged-обучения) | — | `optuna_pruner` принимается, но не подключается — прунинг не имеет смысла без промежуточных отчётов внутри trial |
| `lama` | не через Optuna | LAMA управляет тюнингом сама; см. `model_settings['timeout']` (сек, по умолч. `n_optuna_trials * 60`) |

### `optuna_verbose`

`False` (по умолч.) — форсирует `optuna.logging.WARNING` на время `fit()` (глушит INFO-логи по каждому trial). `True` — не трогает текущий уровень логирования Optuna.

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
| `loss_spec` | `LossSpec \| None` | catboost, lightgbm, xgboost | `None` → фиксированный training loss адаптера |
| `eval_metric` | `str \| callable \| (callable, direction) \| None` | catboost, lightgbm, xgboost | `None` → фреймворк сам выводит метрику из loss_function/objective (см. «Метрика раннего останова»; `(callable, direction)` — только для XGBoost, голый callable там запрещён) |
| `undersample_majority` | `bool` | catboost, lightgbm, xgboost (классификаторы) | `False` для всех трёх |
| `optuna_timeout` | `float \| None` (секунды) | все Optuna-адаптеры | `None` → без лимита времени |
| `optuna_pruner` | `None \| str \| optuna.pruners.BasePruner` | все Optuna-адаптеры (реально отсекает trials только в catboost/lightgbm/xgboost/*_ranker/tabm) | `None` → `MedianPruner()` |
| `optuna_verbose` | `bool` | все Optuna-адаптеры | `False` → форсирует WARNING-уровень логов Optuna |

---

## Примеры вызова: пять сценариев возрастающей сложности

Один и тот же `CatBoostClassifier` на одних и тех же `X_train`/`y_train`/`X_valid`/`y_valid` — различается только то, что передано в конструктор. `LightGBMClassifier`/`XGBoostClassifier`/соответствующие `*Regressor` читают те же ключи `model_settings` без изменений (различаются только имена гиперпараметров внутри `params` в сценарии 1 — у каждого фреймворка свой нативный словарь).

### 1. Нативный бустинг — без Optuna, явные параметры

```python
from ml_toolkit.models import CatBoostClassifier

model = CatBoostClassifier(params={'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'verbose': 0})
model.fit(X_train, y_train, X_valid, y_valid)
proba = model.predict_proba(X_valid)
```

Сами гиперпараметры дерева (`params`) уходят в CatBoost как есть, без Optuna — `model.best_params_ == params`. Но это не то же самое, что вызвать `catboost.CatBoostClassifier` напрямую: адаптер всё равно сам определяет binary/multiclass по `y_train`, собирает `Pool` из `cat_features` за вас, после `fit()` фитит изотоническую калибровку вероятностей на валидации (`predict_proba()` уже калиброван — сырой CatBoost вероятности не калибрует) и даёт тот же `.fit()`/`.predict_proba()`/`.save()`/`.load()` контракт, что у `LightGBMClassifier`/`XGBoostClassifier`/прочих адаптеров. Разница с примерами 2–5 ниже — только в том, что `params` фиксирован и Optuna не участвует; разница с самим CatBoost — во всём перечисленном.

### 2. Бустинг с Optuna — тюнинг гиперпараметров дерева

```python
model = CatBoostClassifier(n_optuna_trials=50)
model.fit(X_train, y_train, X_valid, y_valid)
proba = model.predict_proba(X_valid)
print(model.best_params_)
```

`params=None` (дефолт конструктора) запускает Optuna по дефолтному search space (`iterations`/`depth`/`learning_rate`/...). Лучший trial выбирается по `cls_metric` (дефолт `'pr_auc'`) — ничего об этом знать не обязательно, если дефолт подходит.

### 3. + перебор параметров лосса (`loss_spec`)

```python
from ml_toolkit.losses import FocalLoss
from ml_toolkit.models._loss_spec import LossSpec

model = CatBoostClassifier(
    n_optuna_trials=50,
    model_settings={
        'loss_spec': LossSpec(name='focal', loss_cls=FocalLoss,
                               param_bounds={'gamma': (1.0, 5.0), 'alpha': (0.1, 0.9)}),
        'eval_metric': 'Logloss',   # у CatBoost обязателен при кастомном loss_function, см. ниже
    },
)
model.fit(X_train, y_train, X_valid, y_valid)
print(model.best_params_['loss_name'], model.best_params_['loss_params'])
```

Теперь Optuna тюнит не только `iterations`/`depth`/..., но и собственные параметры лосса (`gamma`/`alpha` у `FocalLoss`) — training loss каждого trial'а свой. `eval_metric` здесь не опционален: CatBoost не умеет сам подобрать метрику раннего останова под кастомный `loss_function` и падает `CatBoostError`, если её не дать (у LightGBM/XGBoost в этой же ситуации падения нет — молчаливый откат на дефолт по типу задачи, см. раздел «Метрика раннего останова» выше).

### 4. + своя метрика раннего останова (`eval_metric`), без кастомного лосса

```python
model = CatBoostClassifier(n_optuna_trials=50, model_settings={'eval_metric': 'AUC'})
model.fit(X_train, y_train, X_valid, y_valid)
proba = model.predict_proba(X_valid)
```

Training loss — обычный `'Logloss'` (дефолт адаптера), но early stopping и Optuna-пруинг внутри каждого trial'а теперь следят за `AUC`, а не за тем, что CatBoost подставил бы сам. Независимо от сценария 2 — `cls_metric`, который выбирает лучший trial *между* trial'ами, как и раньше, не трогали.

### 5. Всё вместе — перебор параметров лосса + своя метрика Optuna + своя метрика раннего останова

```python
from sklearn.metrics import roc_auc_score

model = CatBoostClassifier(
    n_optuna_trials=50,
    model_settings={
        'loss_spec': LossSpec(name='focal', loss_cls=FocalLoss,
                               param_bounds={'gamma': (1.0, 5.0), 'alpha': (0.1, 0.9)}),
        'eval_metric': 'AUC',                         # (2) ранний останов/пруинг внутри trial'а
        'cls_metric': (roc_auc_score, 'maximize'),     # (3) выбор лучшего trial между trial'ами
    },
)
model.fit(X_train, y_train, X_valid, y_valid)
proba = model.predict_proba(X_valid)
print(model.best_params_['loss_name'], model.best_params_['loss_params'])
```

Три независимые функции в одном вызове: `FocalLoss` растит деревья (1 — training loss), `AUC` решает, когда остановить бустинг и когда прибить trial досрочно (2 — `eval_metric`), `roc_auc_score` сравнивает уже обученные trial'ы между собой и выбирает лучший (3 — `cls_metric`). Ничто не обязывает их совпадать — см. «Метрика раннего останова» выше про то, почему это осмысленно разделено.
