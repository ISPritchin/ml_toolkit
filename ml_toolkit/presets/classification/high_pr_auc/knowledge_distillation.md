# Дистилляция ансамбля в один скорер (`knowledge_distillation.py`)

`KnowledgeDistillationPreset` решает задачу **compression**: у вас уже есть тяжёлый, но качественный ансамбль (`EasyEnsembleClassifier`, `BoostedEnsemble`, `HeterogeneousStacking`, `SubsampleStacking`, ...), а в проде нужен один быстрый CatBoost — без 10–20 моделей в памяти и без накладных расходов на инференс каждой из них.

Импорт:

```python
from ml_toolkit.presets.classification.high_pr_auc import KnowledgeDistillationPreset
```

---

## Идея

Классический knowledge distillation (Hinton et al., 2015), адаптированный под табличный бинарный кейс:

```
1. Учитель (teacher_preset) обучается как обычно — на X_train/y_train/X_valid/y_valid.
2. Учитель скорит X_train — получаем "мягкие" вероятности p_teacher вместо
   жёстких 0/1 меток.
3. p_teacher смягчается температурой T: p_soft = sigmoid(logit(p_teacher) / T).
4. Студент (один CatBoost) обучается на (X_train, p_soft) через
   loss_function='CrossEntropy' — единственный нативный CatBoost-лосс,
   принимающий непрерывную метку в [0, 1], а не только 0/1.
5. predict_proba() финальной модели — это только студент. Учитель в
   инференсе не участвует и никак не сохраняется в предсказании.
```

### Зачем смягчать вероятности (temperature)

Вероятности уверенного учителя (0.001 или 0.999) несут почти тот же сигнал, что и жёсткие 0/1-метки — обучение студента на них теряет самую ценную часть информации: **относительную** уверенность учителя между похожими объектами (объект A учитель считает более вероятным позитивом, чем объект B, даже если оба получили score < 0.01).

`temperature` делит логит перед обратной сигмоидой:

```
p_soft = 1 / (1 + exp(-logit(p_teacher) / T))
```

- `T = 1.0` — без изменений, `p_soft == p_teacher`.
- `T > 1.0` — вероятности стягиваются к 0.5, относительный порядок сохраняется (сигмоида монотонна), но становится более различимым для градиента CrossEntropy — рекомендуется при сильном дисбалансе, когда учитель уверен почти во всём.
- `T < 1.0` — обратный эффект, вероятности расходятся к краям сильнее, чем у учителя. Практически не нужен.

Рекомендуемый диапазон — **1.0–4.0**; для `EasyEnsembleClassifier`/rank-averaged учителей, чьи "вероятности" и так уже смещены (см. `easy_ensemble.py` — предупреждение про `predict_proba()`), начинайте с `T=2.0` и смотрите на `student_score_` относительно `teacher_score_`.

---

## Параметры

```python
KnowledgeDistillationPreset(
    teacher_preset: BasePreset,
    student_params: dict[str, Any] | None = None,
    temperature: float = 2.0,
    n_optuna_trials: int = 0,
    param_space: Callable[[Any], dict[str, Any]] | None = None,
    optuna_timeout: int | None = None,
    optuna_verbose: bool = False,
    optuna_pruner: str | object | None = 'none',
    random_seed: int = 42,
)
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `teacher_preset` | обязателен | Любой **необученный** `BasePreset` (`EasyEnsembleClassifier`, `BoostedEnsemble`, `HeterogeneousStacking` и т.д.). `fit()` вызывается внутри `KnowledgeDistillationPreset.fit()` — передавать уже обученный объект не нужно и не имеет смысла (будет переобучен). |
| `student_params` | None (модальные дефолты — неглубокий, быстрый CatBoost) | Параметры CatBoost-студента. `loss_function`/`eval_metric` игнорируются, даже если заданы — всегда `CrossEntropy`/`AUC` (см. ниже почему). Игнорируется целиком, если `n_optuna_trials > 0`. |
| `temperature` | 2.0 | Смягчение вероятностей учителя перед обучением студента. См. раздел выше. |
| `n_optuna_trials` | 0 | Если > 0 — архитектура студента подбирается Optuna. Objective — честный `average_precision_score` на **настоящих** `y_valid`, не внутренняя метрика CatBoost (см. ниже). |
| `param_space` | None | Кастомный search space для Optuna (см. другие пресеты пакета — тот же контракт: `loss_function`/`eval_metric` в нём не участвуют, они фиксированы). |
| `optuna_timeout`, `optuna_verbose`, `optuna_pruner`, `random_seed` | — | Как во всех остальных пресетах `high_pr_auc/` с Optuna-тюнингом. |

## Атрибуты после `fit`

| Атрибут | Тип | Описание |
|---|---|---|
| `teacher_` | `BasePreset` | Обученный `teacher_preset` (со всеми его собственными атрибутами — `estimators_`, `ensemble_score_` и т.п., в зависимости от того, что это за пресет) |
| `teacher_score_` | `float` | val PR-AUC учителя на настоящих (не смягчённых) `y_valid` |
| `student_score_` | `float` | val PR-AUC студента на настоящих `y_valid` — прямое сравнение с `teacher_score_` показывает цену компрессии |
| `train_pred_` / `valid_pred_` | `np.ndarray` | Предсказания **студента** (не учителя) |
| `best_params_` | `dict` | `{'student_params': ..., 'temperature': ...}` |

---

## Примеры

### Базовый сценарий

```python
from ml_toolkit.presets.classification.high_pr_auc import (
    EasyEnsembleClassifier, KnowledgeDistillationPreset,
)

model = KnowledgeDistillationPreset(
    teacher_preset=EasyEnsembleClassifier(n_estimators=15, neg_ratio=10),
    temperature=2.0,
)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)

print(f'teacher={model.teacher_score_:.4f}  student={model.student_score_:.4f}')
proba = model.predict_proba(X_test)  # один CatBoost, не 15 моделей учителя
```

### С подбором архитектуры студента через Optuna

```python
model = KnowledgeDistillationPreset(
    teacher_preset=EasyEnsembleClassifier(n_estimators=15, neg_ratio=10, base_params={'iterations': 500}),
    temperature=3.0,
    n_optuna_trials=30,
)
model.fit(X_train, y_train, X_valid, y_valid)
print(model.best_params_['student_params'])
```

### С произвольным учителем (не обязательно ансамбль)

`teacher_preset` — любой `BasePreset`, необязательно многокомпонентный ансамбль. Дистилляция также работает как способ **перенести** знания одной дорогой конфигурации (например, с большим `n_optuna_trials`) в дешёвую фиксированную модель для частого переобучения:

```python
from ml_toolkit.presets.classification.high_pr_auc import HeterogeneousStacking

model = KnowledgeDistillationPreset(
    teacher_preset=HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'xgboost'], n_folds=5),
    temperature=1.5,
)
model.fit(X_train, y_train, X_valid, y_valid)
```

### Подбор температуры

`temperature` не тюнится Optuna автоматически (в отличие от архитектуры) — это осознанный выбор: слишком большая температура ухудшает `student_score_`, слишком маленькая теряет часть сигнала. Быстрый ручной перебор:

```python
for T in [1.0, 1.5, 2.0, 3.0, 5.0]:
    model = KnowledgeDistillationPreset(
        teacher_preset=EasyEnsembleClassifier(n_estimators=10, neg_ratio=10, base_params={'iterations': 300}),
        temperature=T,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    print(f'T={T}  student_score_={model.student_score_:.4f}  (teacher={model.teacher_score_:.4f})')
```

---

## Технические детали

### Почему `eval_metric='AUC'`, а не `'PRAUC'` (как везде в пакете)

Все остальные пресеты `high_pr_auc/` используют `eval_metric='PRAUC'` по умолчанию — это единообразие сознательно нарушено здесь. Причина — ограничение CatBoost (проверено эмпирически на 1.2.10):

```python
CatBoostClassifier(loss_function='CrossEntropy', eval_metric='PRAUC', ...).fit(
    Pool(X_train, label=soft_continuous_labels),  # метка в [0, 1], не 0/1
    eval_set=Pool(X_valid, label=hard_binary_labels),
)
# → CatBoostError: catboost/libs/metrics/metric.cpp:4763: No element of a positive class
```

`PRAUC` падает, как только train-метка непрерывна (что необходимо для дистилляции по soft labels) — независимо от того, что `eval_set` при этом использует нормальные бинарные метки. `AUC`, `Logloss` и сам `CrossEntropy` в качестве `eval_metric` с той же комбинацией работают штатно.

Это не влияет на то, что реально репортируется пользователю: `teacher_score_`/`student_score_`/Optuna-objective считаются отдельно через `sklearn.average_precision_score` на настоящих `y_valid`, а не через внутренний CatBoost-метрику — `eval_metric='AUC'` используется только как сигнал для early stopping внутри `fit()`.

### Почему `loss_function='CrossEntropy'`, а не Logloss + `sample_weight`

CatBoost поддерживает 0/1-метки с весами (`sample_weight`) для стандартного `Logloss`, но это ортогонально задаче: вес объекта — это "насколько сильно штрафовать ошибку", а не "какова целевая вероятность". Дистилляция по soft labels требует именно второго — студент должен воспроизводить конкретное число `p_teacher`, а не просто быть точнее на «важных» объектах. `CrossEntropy` — единственный нативный CatBoost-лосс, который берёт эту непрерывную метку напрямую как таргет.

---

## Когда использовать

- В проде нужен один быстрый скорер (задержка, память, операционная простота важнее качества последних процентов PR-AUC), а разработка/подбор ведётся на тяжёлом ансамбле.
- Нужно A/B-тестировать несколько дорогих стратегий офлайн, но задеплоить можно только одну лёгкую модель.
- Учитель дорог именно в **инференсе** (много базовых моделей), а не в обучении — дистилляция не ускоряет сам процесс тренировки (учитель всё равно обучается полностью).

## Когда не подходит

- Если разрыв `teacher_score_ - student_score_` велик — значит, ансамбль учителя действительно даёт что-то, что один CatBoost не может выучить даже по soft labels (нелинейные взаимодействия между разными базовыми алгоритмами семейства `HeterogeneousStacking`, например). В этом случае дистилляция — не бесплатный обед; стоит явно сравнить издержки инференса ансамбля с потерей качества.
- Если инференс учителя и так достаточно быстрый (например, `MultiSeedBlend` из 3 CatBoost) — выигрыш от компрессии в одну модель может не окупить потерю качества.
