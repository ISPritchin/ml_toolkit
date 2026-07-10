# Пресеты классификации с высоким дисбалансом (`high_pr_auc/`)

36 специализированных пресетов для бинарной классификации при доле позитивов < 5%
(типично < 1%). Все нацелены на максимизацию **PR-AUC**, а не ROC-AUC. Для
мультиклассовой классификации с длинным хвостом — см. соседний
`multiclass_imbalance/`.

---

## Общий API

Все пресеты реализуют единый интерфейс, унаследованный от `BasePreset(BaseModel)`.

```python
from ml_toolkit.presets.classification.high_pr_auc import TwoStageCascade

model = TwoStageCascade()
model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...], cat_features=[...])

proba  = model.predict_proba(X_test)        # np.ndarray, вероятности класса 1
labels = model.predict(X_test, threshold=0.3)  # np.ndarray, 0/1
```

После `fit()` доступны:

| Атрибут | Тип | Содержимое |
|---------|-----|------------|
| `selected_features_` | `list[str]` | Использованные признаки |
| `cat_features_` | `list[str]` | Категориальные признаки |
| `train_pred_` | `np.ndarray` | Вероятности на train |
| `valid_pred_` | `np.ndarray` | Вероятности на val |
| `best_params_` | `dict` | Лучшие параметры (Optuna или заданные) |

---

## Быстрый выбор

```
Что известно о задаче?
│
├─ ROC-AUC < 0.70 → нет дискриминирующего сигнала, смотрите признаки
│
├─ ROC-AUC ≥ 0.70, PR-AUC низкий
│   ├─ Undersampling помог → EasyEnsembleClassifier, HardNegativeMiner
│   ├─ Мало позитивов (< 200 в train) → SyntheticOversamplingClassifier
│   ├─ Подозрение на незамеченных позитивов → PULearningClassifier, SelfTrainingBooster
│   │   ├─ Элкан-Ното нестабилен (мало val-позитивов) → ElkanNotoHoldoutPU, BaggingPUClassifier
│   │   ├─ Нужны явные reliable negatives → SpyPUClassifier
│   │   └─ Известен и стабилен class_prior → NNPUClassifier
│   ├─ Подозрение на шумные (неверные) метки → ConfidentLearningCleaner, CoTeachingClassifier
│   ├─ Вероятности завышены (многие > 0.5) → CalibratedWrapper
│   ├─ train/inference разнесены во времени, признаки дрейфуют
│   │   ├─ Дрейфующие признаки можно выбросить → DriftRobustClassifier
│   │   └─ Дрейфующие признаки слишком ценны, чтобы терять → AdversarialValidationWeighting
│   └─ Стандартный Logloss достиг плато
│       ├─ Быстрый одномодельный бейзлайн → FocalLossClassifier, PolyLossClassifier
│       ├─ Шумные негативы / мультилейбл-подобный дисбаланс → AsymmetricLossClassifier, LambdaRankClassifier
│       ├─ Focal не давит выбросы/шумные метки → GHMLossClassifier
│       ├─ Дисбаланс внутри класса (не только между классами) → InfluenceBalancedLossClassifier
│       ├─ ASL уже используется, нужна ещё одна степень свободы → AsymmetricPolyLossClassifier
│       └─ Экстремальный дисбаланс, Focal/ASL не помогли → LDAMClassifier
│
├─ Метрика — F1/Dice, а не PR-AUC → DiceLossClassifier
│
├─ Нужен контроль recall/precision
│   ├─ Зафиксировать минимальный recall → ThresholdMovingCV(optimize='precision_at_recall')
│   ├─ Максимизировать precision@K → PrecisionAtKClassifier
│   └─ Зашить FP/FN трейдофф прямо в градиент → TverskyLossClassifier
│
├─ Ансамбль / стэкинг
│   ├─ Простой, без утечки → EasyEnsembleClassifier
│   ├─ Со стэкингом OOB, один алгоритм, разные конфиги → SubsampleStacking
│   ├─ Со стэкингом OOB, разные семейства алгоритмов → HeterogeneousStacking
│   ├─ Несколько loss-функций → BoostedEnsemble
│   ├─ Сотни коррелированных признаков → FeatureBaggingEnsemble
│   ├─ Бюджет — одна модель → SnapshotEnsembleClassifier
│   ├─ Дешёвое снижение дисперсии перед сравнением пресетов → MultiSeedBlend
│   ├─ Уже есть 10+ обученных моделей, нужно выбрать комбинацию → GreedyForwardEnsembleSelection
│   ├─ Панельные данные, свежее важнее старого, но выбрасывать жалко → WeightedBaggingByRecency
│   └─ Качество — от ансамбля, но в проде нужен один быстрый скорер → KnowledgeDistillationPreset
│
├─ Важность признаков скачет между запусками → StabilitySelectionClassifier
│
├─ Модель должна быть защитима перед бизнесом/регулятором
│   (доменное «больше X → не ниже скор») → MonotonicConstrainedClassifier
│
└─ Панельные данные клиент × период, один train/valid/test cutoff шумный
    или метка присваивается с лагом → TimeAwareValidationClassifier
```

---

## BoostedEnsemble

**Файл:** `ensemble_losses.py`

Ансамбль CatBoost-моделей, каждая обучается с другой функцией потерь (Logloss, Focal, взвешенный Logloss). Итоговый скор агрегируется через `mean`, `rank`, `weighted` или `power`-среднее.

**Когда использовать:** базовый ансамблевый пресет, с которого стоит начинать. Focal Loss фокусируется на трудных примерах и подавляет лёгкие негативы.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_models` | 4 | Число моделей в ансамбле |
| `gamma` | 2.0 | Параметр Focal Loss (γ ≥ 1) |
| `alpha` | 0.25 | Вес позитивов в Focal Loss |
| `averaging` | `'rank'` | Стратегия усреднения: `'mean'`, `'rank'`, `'weighted'`, `'power'` |
| `base_params` | None | Параметры CatBoost; None → дефолтные |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import BoostedEnsemble

model = BoostedEnsemble(gamma=2.0, averaging='rank')
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

---

## PrecisionAtKClassifier

**Файл:** `precision_at_k.py`

Optuna подбирает гиперпараметры CatBoost, оптимизируя **precision в топ-K объектов** по score (ранговая метрика). Подходит, когда аналитики обрабатывают фиксированное число лидов.

**Когда использовать:** цель — не «найти всех», а получить максимально чистый список из K кандидатов.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `k` | 50 | Размер топа для precision@K |
| `n_optuna_trials` | 50 | Число Optuna-триалов |
| `base_params` | None | Стартовые параметры CatBoost |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import PrecisionAtKClassifier

model = PrecisionAtKClassifier(k=30, n_optuna_trials=40)
model.fit(X_train, y_train, X_valid, y_valid)
print(model.best_params_)
```

---

## TwoStageCascade

**Файл:** `cascade.py`

Двухэтапный каскад: первая CatBoost-модель отсеивает явных негативов, вторая — дообучается на «сложных» кандидатах. Эффективно снижает число ложных срабатываний без потери recall.

**Когда использовать:** высокий recall на первом этапе, высокая precision — на втором.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `stage1_threshold` | 0.05 | Порог первой стадии (пропускает кандидатов дальше) |
| `stage1_params` | None | CatBoost параметры этапа 1 |
| `stage2_params` | None | CatBoost параметры этапа 2 |
| `n_optuna_trials` | 0 | Optuna-триалы для stage 2 |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import TwoStageCascade

model = TwoStageCascade(stage1_threshold=0.02)
model.fit(X_train, y_train, X_valid, y_valid)
```

---

## HardNegativeMiner

**Файл:** `hard_negative_mining.py`

Итеративный Hard Negative Mining: модель обучается, затем повышает вес негативов с высоким score (те, которые она «путает» с позитивами). Повтор N раз.

**Когда использовать:** когда большинство негативов слишком лёгкие для модели и она не видит информативного сигнала от трудных случаев.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_rounds` | 3 | Число раундов переобучения |
| `hard_quantile` | 0.9 | Квантиль score для определения «трудных» негативов |
| `weight_multiplier` | 3.0 | Множитель веса для трудных негативов |
| `base_params` | None | Параметры CatBoost |

---

## SubsampleStacking

**Файл:** `stacking.py`

N CatBoost-моделей на бутстрэп-подвыборках train; OOB-предсказания складываются в мета-признак для финальной модели второго уровня. Аналог bagging-стэкинга.

**Когда использовать:** хочется стэкинга без утечки данных через OOB-сплит.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_models` | 5 | Число базовых моделей |
| `subsample_ratio` | 0.7 | Доля train для каждой модели |
| `base_params` | None | Параметры CatBoost |
| `meta_params` | None | Параметры мета-модели |

---

## EasyEnsembleClassifier

**Файл:** `easy_ensemble.py`

N моделей (LightGBM или CatBoost), каждая обучается на **всех позитивах** + случайном срезе `neg_ratio * n_pos` негативов. Итог — среднее нормированных рангов.

**Когда использовать:** undersampling помогает, но нужно больше diversity чем в одной модели. Первый кандидат при дисбалансе > 50:1.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_estimators` | 10 | Число базовых моделей |
| `neg_ratio` | 10 | Негативов на позитив в каждой подвыборке |
| `base` | `'lightgbm'` | `'lightgbm'` или `'catboost'` |
| `base_params` | None | Параметры базовой модели |
| `random_seed` | 42 | Зерно (estimator i получает seed + i) |

**Атрибуты после fit:**
- `estimator_scores_` — val PR-AUC каждой базовой модели
- `ensemble_score_` — val PR-AUC ансамбля

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import EasyEnsembleClassifier

model = EasyEnsembleClassifier(n_estimators=15, neg_ratio=10)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(f'Ансамбль: {model.ensemble_score_:.4f}  '
      f'Среднее одиночное: {sum(model.estimator_scores_)/len(model.estimator_scores_):.4f}')
```

---

## HeterogeneousStacking

**Файл:** `heterogeneous_stacking.py`

Стек **разных семейств** алгоритмов (CatBoost + LightGBM + XGBoost +
LogisticRegression) на честном K-fold OOF — та же OOF-механика, что в
SubsampleStacking, но diversity не из разных конфигов одного алгоритма, а из
принципиально разных индуктивных смещений. xgboost — опциональная
зависимость: если не установлен, молча исключается из зоопарка (после
фильтрации нужно >= 2 доступных члена).

**Когда использовать:** одномодельный (или однородный-стековый) потолок уже достигнут.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `base_zoo` | все 4 | Подмножество `['catboost','lightgbm','xgboost','logistic']` |
| `meta` | `'logistic'` | `'logistic'`, `'weighted'` или `'catboost'` |
| `n_folds` | 5 | Число фолдов честного OOF |
| `calibrate` | True | Изотоническая калибровка финальных предсказаний |

**Атрибуты после fit:** `zoo_used_`, `base_models_` (dict), `oob_pr_aucs_` (dict), `valid_pr_auc_`.

```python
from ml_toolkit.presets.classification.high_pr_auc import HeterogeneousStacking

model = HeterogeneousStacking(base_zoo=['catboost', 'lightgbm', 'logistic'])
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats, cat_features=cats)
```

## MultiSeedBlend

**Файл:** `multi_seed_blend.py`

Один конфиг CatBoost, `n_seeds` независимых обучений, rank-avg blend
(`fit_rank_reference`/`rank_transform`, как в EasyEnsembleClassifier). Самое
дешёвое снижение дисперсии из всех пресетов пакета — никакой diversity, кроме
случайного зерна.

**Когда использовать:** перед тем как вообще сравнивать пресеты между собой —
нестабильно сравнивать A против B, если сам A шумит сильнее разницы A и B.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_seeds` | 7 | Число независимых обучений (отдача убывает быстро после ~10) |

**Атрибуты после fit:** `seed_scores_`, `blend_score_`.

## GreedyForwardEnsembleSelection

**Файл:** `greedy_ensemble_selection.py`

Caruana ensemble selection (Caruana et al., 2004) — единственный пресет
пакета, который **не обучает ничего с нуля**: `model_library` — уже
обученные модели/пресеты, задача — выбрать лучшую комбинацию. Жадный отбор с
возвратом (модель может быть выбрана несколько раз → больший вес),
`n_bags` bootstrap-повторов на val для регуляризации против переобучения под
конкретный val-сплит.

**Когда использовать:** есть зоопарк из 10+ уже обученных моделей/пресетов.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `model_library` | обязателен | Список обученных объектов с `predict_proba(X)` |
| `max_members` | 10 | Максимум слотов в ансамбле за один жадный прогон |
| `n_bags` | 20 | Число bootstrap-повторов отбора |

**Атрибуты после fit:** `weights_` (доля выборов на модель, сумма=1), `pick_counts_`. `train_pred_` не вычисляется (`None`) — библиотека обучена на внешних данных, X_train этого пресета не используется.

```python
from ml_toolkit.presets.classification.high_pr_auc import GreedyForwardEnsembleSelection

model = GreedyForwardEnsembleSelection(model_library=[m1, m2, m3, ...], max_members=5)
model.fit(X_train, y_train, X_valid, y_valid)
print(dict(zip(range(len(model.weights_)), model.weights_)))
```

---

## KnowledgeDistillationPreset

**Файл:** `knowledge_distillation.py`

Обучает тяжёлого «учителя» (любой `BasePreset` — ансамбль, стекинг и т.д.), затем маленький одиночный CatBoost-«студент» учится на смягчённых по температуре вероятностях учителя (soft labels) через `loss_function='CrossEntropy'` — единственный нативный CatBoost-лосс, принимающий непрерывную метку в [0, 1]. Финальный `predict_proba()` — только студент, учитель в инференсе не участвует.

**Когда использовать:** качество нужно от тяжёлого ансамбля, но в проде должен скорить один быстрый CatBoost (задержка/память/операционная простота важнее, чем 10-15 моделей учителя).

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `teacher_preset` | обязателен | Необученный `BasePreset` — fit() вызывается внутри |
| `student_params` | None | Параметры CatBoost-студента (без loss_function/eval_metric) |
| `temperature` | 2.0 | Смягчение вероятностей учителя (1.0 — без смягчения) |
| `n_optuna_trials` | 0 | Тюнинг архитектуры студента по честному PR-AUC на val |

**Атрибуты после fit:**
- `teacher_` — обученный `teacher_preset`
- `teacher_score_` / `student_score_` — val PR-AUC учителя и студента (на настоящих метках)

**Пример:**
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
```

---

## PULearningClassifier

**Файл:** `pu_learning.py`

CatBoost + **коррекция Элкана–Ното**. Оценивает `c = P(s=1|y=1)` по валидационным позитивам, корректирует вероятности: `P(y=1|x) = raw(x) / c`.

**Когда использовать:** есть основания считать, что часть «негативов» в train — незамеченные позитивы (сегментация менялась, time split с запаздыванием пометки, noisy labels).

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `base_params` | None | Параметры CatBoost |
| `n_optuna_trials` | 0 | Число Optuna-триалов |
| `c_lower_bound` | 0.1 | Нижняя граница c (защита от деления на очень малое) |

**Атрибуты после fit:**
- `c_` — оценённое P(s=1|y=1)
- `raw_pr_auc_` / `corrected_pr_auc_` — PR-AUC до и после коррекции

**Интерпретация `c_`:**
- `c_ ≈ 1.0` → модель хорошо отличает классы, коррекция минимальна
- `c_ ≈ 0.3` → только ~30% истинных позитивов были помечены → сильная коррекция

---

## Шумные метки и PU-обучение, продолжение (ConfidentLearningCleaner / CoTeaching / BaggingPU / SpyPU / ElkanNotoHoldoutPU / NNPU)

Шесть пресетов для двух разных, но смежных проблем: **шумные метки** (метка
неверна с обеих сторон — часть позитивов на деле негативы, и наоборот) и
**PU-структура** (специфичный односторонний шум: часть позитивов не помечена,
негативы в среднем достоверны). PULearningClassifier — Элкан-Ното — уже
решает PU пост-хок коррекцией вероятностей; здесь — более специализированные
варианты.

### ConfidentLearningCleaner

**Файл:** `confident_learning_cleaner.py`

Нативная (без зависимости от библиотеки cleanlab) реализация confident
learning (Northcutt et al., 2021): честные OOF-вероятности → self-confidence
пороги по классам → confident joint → `prune_by_noise_rate` удаляет
подозрительные метки → переобучение на очищенном train.

**Когда использовать:** есть основания не доверять разметке в обе стороны (не
только unlabeled-позитивы, как в PU-постановке).

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_folds` | 5 | Число фолдов для честных OOF-вероятностей |
| `filter` | `'prune_by_noise_rate'` | Единственный реализованный метод чистки |

**Атрибуты после fit:** `class_thresholds_`, `confident_joint_` (2x2), `removed_indices_`, `oof_pr_auc_`.

```python
from ml_toolkit.presets.classification.high_pr_auc import ConfidentLearningCleaner

model = ConfidentLearningCleaner(n_folds=5)
model.fit(X_train, y_train, X_valid, y_valid)
print(f"Удалено {len(model.removed_indices_)} подозрительных меток")
```

### CoTeachingClassifier

**Файл:** `co_teaching.py`

Две CatBoost-модели с разными seed'ами взаимно обучаются на small-loss
примерах ПАРТНЁРА (Han et al., 2018) — без перекрёстного обмена одна модель,
отбирающая свои же "лёгкие" примеры, скатывается в confirmation bias.
`forget_rate` нарастает линейно от 0 до целевого значения за `n_rounds`.
Малый loss отбирается **отдельно внутри каждого класса** (не глобально) —
иначе агрессивный forget_rate при сильном дисбалансе может целиком вымыть
миноритарный класс из "чистой" выборки.

**Когда использовать:** доля шумных меток заметная (>5%), одиночная чистка
(ConfidentLearningCleaner) нестабильна.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `forget_rate` | 0.2 | Итоговая доля train, отбрасываемая как шум |
| `n_rounds` | 5 | Число раундов co-teaching |

**Атрибуты после fit:** `round_scores_a_`/`round_scores_b_`/`ensemble_scores_`, `keep_fraction_history_`.

### BaggingPUClassifier

**Файл:** `bagging_pu.py`

PU-бэггинг (Mordelet & Vert, 2014): каждый из `n_estimators` обучается на всех
позитивах против bootstrap-выборки из U; скор точки U усредняется только по
estimator'ам, где она была **out-of-bag** — иначе оценка смещена именно там,
где U ошибочно (незамеченный позитив).

**Когда использовать:** Элкан-Ното (PULearningClassifier) нестабилен из-за
шумной оценки `c` на малом val.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_estimators` | 30 | Число базовых моделей |
| `u_sample_size` | None → n_pos | Размер bootstrap-выборки из U на estimator |

**Атрибуты после fit:** `oob_coverage_`, `train_pu_pr_auc_`.

### SpyPUClassifier

**Файл:** `spy_pu.py`

S-EM spy-техника (Liu et al., 2003): часть позитивов временно маскируется под
U ("шпионы"); порог reliable-negative выбирается так, что ровно
`spy_threshold_pct`% шпионов (заведомо позитивных) окажутся ниже него —
контролируемая, а не произвольная, цена ошибки. Финальная модель — обычный
supervised P vs RN (reliable negatives).

**Когда использовать:** нужно явное множество надёжных негативов — например,
для последующей ручной проверки.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `spy_frac` | 0.1 | Доля позитивов, маскируемых под шпионов |
| `spy_threshold_pct` | 5.0 | % шпионов, которым разрешено оказаться ниже порога RN |

**Атрибуты после fit:** `threshold_`, `n_reliable_negative_`, `n_spies_`, `stage1_pr_auc_`.

### ElkanNotoHoldoutPU

**Файл:** `elkan_noto_holdout_pu.py` (наследует `PULearningClassifier`)

Расширяет Элкана-Ното bootstrap доверительным интервалом для `c` — при малом
числе c-holdout позитивов точечная оценка `c_` сама по себе шумная; широкий
CI — сигнал "доверяйте ранжированию, а не абсолютным вероятностям".

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `c_holdout_frac` | 0.3 | То же, что `c_estimation_frac` у PULearningClassifier |
| `n_bootstrap` | 100 | Число bootstrap-ресэмплов для CI |

**Атрибуты после fit (в дополнение к PULearningClassifier):** `c_ci_`, `c_bootstrap_std_`.

### NNPUClassifier

**Файл:** `nnpu_loss.py` (лосс — `ml_toolkit/losses/_nnpu.py`)

Единственный из PU-пресетов, встраивающий PU-структуру прямо в градиент, а не
лечащий её пост-хок: **non-negative PU risk estimator** (Kiryo et al., 2017)
напрямую оптимизирует несмещённую оценку истинного риска, если `class_prior`
(pi = P(y=1), включая незамеченные) известен заранее.

**Когда использовать:** pi известен и стабилен (например, пересчитан по
полным данным прошлых периодов) — тогда не нужны эвристики (spy_frac,
u_sample_size, c-holdout) других PU-методов.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `class_prior` | обязателен | pi = P(y=1), истинная доля позитивов |
| `beta` | 0.0 | Порог срабатывания non-negative коррекции |
| `gamma` | 1.0 | Множитель "обратного" градиента при коррекции |

```python
from ml_toolkit.presets.classification.high_pr_auc import NNPUClassifier

model = NNPUClassifier(class_prior=0.05, n_optuna_trials=30)
model.fit(X_train, y_train, X_valid, y_valid)
```

---

## CalibratedWrapper

**Файл:** `calibrated.py`

Обёртка над **любым другим пресетом**: обучает его, затем калибрует вероятности на валидации (isotonic regression или Platt/LogReg).

**Когда использовать:** undersampling завышает вероятности (модель «думает», что позитивов 50%); после калибровки PR-AUC может заметно вырасти без переобучения.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `base_preset` | — | Любой необученный `BasePreset` |
| `method` | `'isotonic'` | `'isotonic'` (нелинейная, нужно ≥ 20 val-pos) или `'platt'` (логрег, устойчива при < 20 val-pos) |

**Атрибуты после fit:**
- `raw_pr_auc_` / `calibrated_pr_auc_` — PR-AUC до и после калибровки
- `calibrator_` — обученный калибратор (IsotonicRegression или LogisticRegression)

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import CalibratedWrapper, EasyEnsembleClassifier

model = CalibratedWrapper(
    EasyEnsembleClassifier(n_estimators=10, neg_ratio=10),
    method='isotonic',
)
model.fit(X_train, y_train, X_valid, y_valid)
print(f'Δ PR-AUC от калибровки: {model.calibrated_pr_auc_ - model.raw_pr_auc_:+.4f}')
```

---

## ThresholdMovingCV

**Файл:** `threshold_moving.py`

Обёртка над **любым другим пресетом**: находит оптимальный порог по валидации.
`predict()` использует найденный порог вместо дефолтного 0.5.

**Когда использовать:** всегда при < 1% позитивов — порог 0.5 почти гарантированно неоптимален.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `base_preset` | — | Любой необученный `BasePreset` |
| `optimize` | `'f2'` | `'f1'`, `'f2'`, `'f0.5'`, `'precision_at_recall'` |
| `min_recall` | None | Обязателен при `optimize='precision_at_recall'` |
| `n_thresholds` | 500 | Шагов в поиске |

**Атрибуты после fit:**
- `threshold_` — найденный оптимальный порог
- `threshold_metric_` — значение метрики на этом пороге
- `scan_df_` — DataFrame (threshold, metric) по всем точкам
- `base_` — обученный base_preset

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import ThresholdMovingCV, TwoStageCascade

model = ThresholdMovingCV(
    TwoStageCascade(),
    optimize='precision_at_recall',
    min_recall=0.70,
)
model.fit(X_train, y_train, X_valid, y_valid)
print(f'Порог: {model.threshold_:.4f}  Precision@recall≥0.70: {model.threshold_metric_:.4f}')
model.plot_threshold_scan()
```

---

## SyntheticOversamplingClassifier

**Файл:** `synthetic_oversampling.py`

SMOTE/ADASYN/BorderlineSMOTE/SMOTEENN генерируют синтетических позитивов интерполяцией в feature space. Затем CatBoost или LightGBM обучается на дополненной выборке.

**Когда использовать:** мало позитивов в train (< 200), или undersampling достиг потолка и хочется попробовать альтернативу.

**Требует:** `imbalanced-learn` (установлен в зависимостях проекта).

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `method` | `'smote'` | `'smote'`, `'adasyn'`, `'borderline'`, `'smoteenn'` |
| `sampling_strategy` | 0.1 | Целевое соотношение minority/majority ПОСЛЕ oversampling. 0.1 → 1:10. |
| `base` | `'catboost'` | `'catboost'` или `'lightgbm'` |
| `base_params` | None | Параметры базовой модели |
| `random_seed` | 42 | Зерно |

**Атрибуты после fit:**
- `n_synthetic_` — число сгенерированных примеров
- `augmented_ratio_` — реальное соотношение после oversampling

**Замечание о категориальных признаках:** SMOTE корректно работает только с числовыми. Для категориальных синтетические примеры берут значения от ближайшего исходного позитива. При обильных категориальных признаках — рассмотрите `EasyEnsembleClassifier`.

**Выбор метода:**

| Метод | Суть | Когда лучше |
|-------|------|-------------|
| `smote` | Интерполяция между k соседями | Базовый, всегда попробовать первым |
| `adasyn` | Больше генерации на границе классов | Когда позитивы «перемешаны» с негативами |
| `borderline` | Только граничные позитивы | Позитивы образуют кластер, граница чёткая |
| `smoteenn` | SMOTE + очистка шума ENN | Много шума в метках (незамеченные позитивы) |

---

## LambdaRankClassifier

**Файл:** `lambda_rank.py`

LightGBM с `objective='lambdarank'`. Градиенты вычисляются из изменений MAP при перестановке пар (positive, negative) — **прямая оптимизация рангового качества**, не surrogate logloss.

**Когда использовать:** модели с logloss достигли плато по PR-AUC, но ROC-AUC продолжает расти — это сигнал расхождения surrogate и целевой метрики.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `base_params` | None | LightGBM параметры (кроме `objective`, `metric`, `label_gain` — задаются автоматически) |
| `eval_at` | None | Позиции для MAP@k, e.g. `[10, 50, 100]`. None → MAP по всем |
| `early_stopping_rounds` | 80 | Раундов без улучшения MAP |
| `random_seed` | 42 | Зерно |

**Атрибуты после fit:**
- `map_train_` / `map_valid_` — MAP на train/val

**Замечание:** `predict_proba` возвращает нормализованные ранги [0, 1], а не вероятности. Значения не интерпретируемы как P(y=1|x), но монотонно связаны с рангом — для PR-AUC и ranking задач этого достаточно.

**Совет:** попробуйте совместно с `ThresholdMovingCV` для нахождения оптимального порога.

---

## Кастомные лоссы (FocalLoss / TverskyLoss / PolyLoss / AsymmetricLoss / LDAM / GHM / IB / Dice / AsymmetricPoly)

`FocalLossClassifier`, `TverskyLossClassifier`, `PolyLossClassifier`, `AsymmetricLossClassifier`, `LDAMClassifier`, `GHMLossClassifier`, `InfluenceBalancedLossClassifier`, `DiceLossClassifier` и `AsymmetricPolyLossClassifier` — CatBoost с одним кастомным Python-лоссом из `ml_toolkit.losses` вместо Logloss. Все девять — тонкие подклассы общего движка `_custom_loss_base.py` (`_CustomLossClassifierBase` + `_LossSpec`): fit/Optuna-тюнинг/predict_proba реализованы один раз, каждый класс лишь объявляет свой `_LossSpec` (класс лосса + границы Optuna-поиска его параметров) и именованные kwargs конструктора. Добавить новый кастомный лосс — значит добавить такой же тонкий файл, а не копировать fit/tune заново.

Общее для всех девяти:
- `base_params: dict | None` — параметры CatBoost (без `loss_function`); `None` → дефолтные.
- `n_optuna_trials: int = 0` — если > 0, Optuna ищет **и** параметры лосса, **и** гиперпараметры CatBoost совместно (архитектурный поиск идентичен во всех); первый триал — значения из `__init__`, чтобы явно заданные не терялись молча.
- `eval_metric='AUC'` зашит внутри (кастомный Python-лосс не совместим с `PRAUC`-метрикой CatBoost так же надёжно, как встроенный Logloss — используется во всех пресетах с кастомными лоссами в проекте).

`LDAMClassifier` и `InfluenceBalancedLossClassifier` — особые случаи: их лоссы (`LDAMLoss`, `InfluenceBalancedLoss`) нуждаются в статистике датасета (n_pos/n_neg), недоступной в момент `__init__`. Для этого `_CustomLossClassifierBase._make_loss(loss_params, *, tr_pool, arch_params)` принимает пул и архитектурные параметры — большинству лоссов они не нужны (дефолтная реализация их игнорирует), эти два класса — единственные, кто эту сигнатуру переопределяет.

### FocalLossClassifier

**Файл:** `focal_loss.py`

Один общий `gamma` на все примеры (в отличие от ASL, где γ+/γ- разные для позитивов/негативов). Быстрый одномодельный бейзлайн для дисбаланса — без ансамбля, как в `BoostedEnsemble`.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `gamma` | 2.0 | Фокусирующий параметр (>= 1) |
| `alpha` | 0.25 | Вес позитивного класса |

```python
from ml_toolkit.presets.classification.high_pr_auc import FocalLossClassifier

model = FocalLossClassifier(n_optuna_trials=40)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### TverskyLossClassifier

**Файл:** `tversky_loss.py`

Управляет трейдоффом FP/FN прямо в градиенте через `alpha`/`beta` batch Tversky index.

**Когда использовать:** recall существенно дороже precision (или наоборот), и это нужно зашить в обучение, а не подбирать порогом (`ThresholdMovingCV`) после.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `alpha` | 0.3 | Вес FP; меньше → выше recall |
| `beta` | 0.7 | Вес FN; больше → выше recall |

```python
from ml_toolkit.presets.classification.high_pr_auc import TverskyLossClassifier

model = TverskyLossClassifier(alpha=0.3, beta=0.7)  # recall важнее precision
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### PolyLossClassifier

**Файл:** `poly_loss.py`

Poly-1 (Leng et al. 2022): CE + `eps1`×(1−p_t) — линейное расширение CE, дешёвая альтернатива Focal Loss.

**Когда использовать:** нужен бейзлайн дешевле Focal, часто не хуже (а то и лучше) обычного CE на дисбалансе.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `eps1` | 2.0 | Коэффициент линейного члена (рекомендуется 1.0–3.0 при дисбалансе) |

```python
from ml_toolkit.presets.classification.high_pr_auc import PolyLossClassifier

model = PolyLossClassifier(n_optuna_trials=40)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### AsymmetricLossClassifier

**Файл:** `asymmetric_loss.py`

CatBoost с **Asymmetric Loss (ASL)** — эволюцией Focal Loss для экстремального дисбаланса (Ridnik et al., 2021). Два разных фокусирующих параметра + вероятностный margin для обрезки лёгких негативов.

**Отличие от BoostedEnsemble (Focal Loss):**

| | Focal Loss | ASL |
|-|------------|-----|
| Параметры | один γ для всех | γ+ для позитивов, γ- для негативов |
| Позитивы | фокусируется на трудных | слабая или нулевая фокусировка (γ+ = 0) |
| Негативы | фокусируется на трудных | **сильная** фокусировка + margin-срез |
| margin | нет | все негативы с p < m исключаются из градиента |

**Когда использовать:** Focal Loss не помог; подозрение на шумные негативы (p < 0.05 → выброшены из градиента).

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `gamma_pos` | 0.0 | Фокусировка для позитивов (рекомендуется 0–1) |
| `gamma_neg` | 4.0 | Фокусировка для негативов (рекомендуется 2–6) |
| `prob_margin` | 0.05 | Порог обрезки: p_s = max(p − m, 0) для негативов |
| `base_params` | None | Параметры CatBoost |
| `n_optuna_trials` | 0 | Optuna-поиск по γ+, γ-, m и гиперпараметрам |

**Пример с Optuna:**
```python
from ml_toolkit.presets.classification.high_pr_auc import AsymmetricLossClassifier

model = AsymmetricLossClassifier(n_optuna_trials=40)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.best_params_)  # оптимальные γ+, γ-, m
```

### LDAMClassifier

**Файл:** `ldam.py`

CatBoost с **LDAM + Deferred Re-Weighting** (Cao et al., 2019). Миноритарный класс получает больший обязательный margin от границы решения (`Δ_j = C / n_j^{1/4}`); первые `reweight_epoch_frac` от общего числа итераций обучение идёт с равными весами (только margin), затем включаются веса по effective number of samples (Cui et al., 2019) — эмпирически лучше, чем reweight с первой итерации.

**Когда использовать:** экстремальный дисбаланс, Focal/ASL не дали прироста.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `max_margin` | 0.5 | Максимальный margin C среди классов (рекомендуется 0.1–1.0) |
| `reweight_epoch_frac` | 0.8 | Доля итераций до включения DRW-перевзвешивания (рекомендуется 0.5–0.95) |
| `beta` | 0.9999 | Коэффициент effective number of samples для DRW-весов; не тюнится Optuna |
| `base_params` | None | Параметры CatBoost |
| `n_optuna_trials` | 0 | Optuna-поиск по max_margin, reweight_epoch_frac и гиперпараметрам |

**Важно:** точка переключения DRW считается как `reweight_epoch_frac * iterations` вызовов лосса (один вызов на дерево CatBoost). Если early stopping остановит обучение раньше этой точки, DRW не успеет включиться за этот запуск — ожидаемое поведение, не баг.

```python
from ml_toolkit.presets.classification.high_pr_auc import LDAMClassifier

model = LDAMClassifier(max_margin=0.5, reweight_epoch_frac=0.8)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### GHMLossClassifier

**Файл:** `ghm_loss.py` (лосс — `ml_toolkit/losses/_ghm.py`)

CatBoost с **Gradient Harmonizing Mechanism** (Li et al., 2019). В отличие от Focal Loss (давит только лёгкие примеры), GHM подавляет ОБА конца распределения градиента |p−y| — и лёгкие негативы, и настоящие выбросы/шумные метки — взвешивая каждый пример обратно пропорционально плотности градиента в его окрестности (гистограмма из `bins` бинов, сглаженная EMA с коэффициентом `momentum` между итерациями бустинга).

**Когда использовать:** Focal Loss (`FocalLossClassifier`) недостаточно давит "неудобные" выбросы.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `bins` | 30 | Число интервалов гистограммы плотности градиента |
| `momentum` | 0.75 | EMA-коэффициент сглаживания между итерациями (0 → без сглаживания) |

```python
from ml_toolkit.presets.classification.high_pr_auc import GHMLossClassifier

model = GHMLossClassifier(bins=30, momentum=0.75)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### InfluenceBalancedLossClassifier

**Файл:** `influence_balanced_loss.py` (лосс — `ml_toolkit/losses/_ib.py`)

CatBoost с **Influence-Balanced Loss** (Park et al., 2021, адаптация под интерфейс CatBoost — без доступа к вектору признаков, influence оценивается через |p−y|). По-сэмпловая (не по-классовая, как `ClassBalancedWeightClassifier`) альтернатива: примеры с большим |p−y| (доминирующие в агрегированном градиенте) подавляются множителем `1/(1+alpha*|p-y|)` поверх стандартного class-balanced веса.

**Когда использовать:** дисбаланс внутри класса — часть позитивов лёгкие дубликаты, часть редкие паттерны, и одного по-классового веса недостаточно.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `alpha` | 1000.0 | Сила подавления примеров с большим influence (рекомендуется 100–1000) |
| `beta` | 0.9999 | Коэффициент effective number of samples для class-balanced части; не тюнится Optuna |

```python
from ml_toolkit.presets.classification.high_pr_auc import InfluenceBalancedLossClassifier

model = InfluenceBalancedLossClassifier(alpha=1000.0)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### DiceLossClassifier

**Файл:** `dice_loss.py` (лосс — `ml_toolkit/losses/_dice.py`, тонкий подкласс `TverskyLoss` c `alpha=beta=0.5`)

CatBoost с **Dice/soft-F1 Loss** — частный случай Tversky index, где FP и FN штрафуются одинаково (прямая мягкая аппроксимация F1, а не произвольный precision/recall трейдофф, как в `TverskyLossClassifier`).

**Когда использовать:** бизнес-метрика — F1/Dice-подобная, а не PR-AUC, и нет причины смещать FP/FN трейдофф в одну сторону.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `smooth` | 1.0 | Коэффициент сглаживания для численной устойчивости |

```python
from ml_toolkit.presets.classification.high_pr_auc import DiceLossClassifier

model = DiceLossClassifier(smooth=1.0)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

### AsymmetricPolyLossClassifier

**Файл:** `asymmetric_poly_loss.py` (лосс — `ml_toolkit/losses/_asymmetric_poly.py`, композиция `AsymmetricLoss` + Poly-1 поправки)

CatBoost с **ASL + Poly-1** в одном лоссе — ещё одна степень свободы (`eps1`) поверх уже настроенного `AsymmetricLossClassifier`.

**Когда использовать:** ASL уже используется, нужна дополнительная настройка через линейный Poly-1 член.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `gamma_pos` | 0.0 | Фокусировка ASL для позитивов (рекомендуется 0–1) |
| `gamma_neg` | 4.0 | Фокусировка ASL для негативов (рекомендуется 2–6) |
| `prob_margin` | 0.05 | Порог обрезки вероятностей негативов ASL |
| `eps1` | 2.0 | Коэффициент линейного Poly-1 члена (рекомендуется 1.0–3.0) |

```python
from ml_toolkit.presets.classification.high_pr_auc import AsymmetricPolyLossClassifier

model = AsymmetricPolyLossClassifier(gamma_pos=0.0, gamma_neg=4.0, eps1=2.0)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
```

---

## SelfTrainingBooster

**Файл:** `self_training.py`

Итеративный pseudo-labeling: CatBoost обучается, предсказывает на «негативах», добавляет высокоуверенные из них как pseudo-positives с пониженным весом и переобучается. Повторяется N раундов.

**Когда использовать:** сегментация «Крупные» менялась со временем → часть клиентов в train ещё не была помечена, хотя по поведению уже соответствует целевому сегменту.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_rounds` | 3 | Число раундов self-training |
| `threshold` | None | Порог score для pseudo-positives. None → 5-й персентиль val-позитивов (консервативный авто-режим) |
| `pseudo_weight` | 0.3 | Вес pseudo-labeled примеров (реальные данные — 1.0) |
| `max_pseudo_ratio` | 2.0 | Макс. число pseudo-positives как кратное реальным (защита от накопления шума) |
| `base_params` | None | Параметры CatBoost |

**Атрибуты после fit:**
- `round_scores_` — val PR-AUC после каждого раунда
- `pseudo_added_` — число pseudo-positives в каждом раунде
- `threshold_used_` — реально использованный порог

**Диагностика:**
```python
model = SelfTrainingBooster(n_rounds=3)
model.fit(X_train, y_train, X_valid, y_valid)
for i, (score, added) in enumerate(zip(model.round_scores_, [0] + model.pseudo_added_)):
    print(f'Раунд {i}: PR-AUC={score:.4f}  pseudo_added={added}')
```

Если `round_scores_` не растёт — pseudo-labeling не помогает; данные не имеют PU-структуры.

---

## AnomalyBlendClassifier

**Файл:** `anomaly_blend.py`

Комбинирует два принципиально разных сигнала:
1. **Isolation Forest** обучается **только на позитивах**; высокий score = «похож на позитивов».
2. **CatBoost** supervised обучается на полной выборке.

Итог: `α × supervised + (1−α) × anomaly_if`. Оптимальный `α` ищется перебором 51 значения по val PR-AUC.

**Когда использовать:** позитивный класс образует компактный кластер в feature space; или supervised модель «переобучается» на видимых негативах и теряет recall на новых паттернах.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_if_estimators` | 200 | Деревьев в Isolation Forest |
| `supervised_params` | None | Параметры CatBoost |
| `n_alpha_steps` | 51 | Точек в поиске alpha |
| `random_seed` | 42 | Зерно |

**Атрибуты после fit:**
- `alpha_` — оптимальное alpha (вес supervised)
- `if_pr_auc_` — PR-AUC только IF-сигнала
- `sup_pr_auc_` — PR-AUC только supervised-сигнала
- `blend_pr_auc_` — PR-AUC blend

**Диагностика:**
```python
model = AnomalyBlendClassifier()
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(f'IF={model.if_pr_auc_:.4f}  supervised={model.sup_pr_auc_:.4f}  '
      f'blend={model.blend_pr_auc_:.4f}  α={model.alpha_:.2f}')
model.plot_alpha_scan()
```

`alpha_ ≈ 0.5` при `blend > max(if, sup)` → сигналы ортогональны, blend работает.
`alpha_ ≈ 1.0` → IF-сигнал не добавляет ничего к supervised.

---

## FeatureBaggingEnsemble

**Файл:** `feature_bagging.py`

N CatBoost-моделей, каждая обучается на своём случайном подмножестве признаков (`feature_frac` от общего числа), а не подвыборке строк. Итог — среднее нормированных рангов (как в `EasyEnsembleClassifier`).

**Когда использовать:** сотни коррелированных инженерных признаков (типичный результат feature generation), деревья одного бустинга «залипают» на одних и тех же ведущих фичах.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_estimators` | 15 | Число базовых моделей |
| `feature_frac` | 0.6 | Доля признаков на модель (без возврата) |
| `base_params` | None | Параметры CatBoost |
| `random_seed` | 42 | Зерно (модель i получает seed + i) |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import FeatureBaggingEnsemble

model = FeatureBaggingEnsemble(n_estimators=15, feature_frac=0.6)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.feature_subsets_[0])   # подпространство признаков первой модели
```

---

## WeightedBaggingByRecency

**Файл:** `weighted_bagging_recency.py`

N моделей (LightGBM или CatBoost), каждая обучается на независимом бутстрэп-сэмпле **всего** train (не только негативов), где вероятность строки попасть в сэмпл убывает экспоненциально с её давностью относительно самого свежего периода в train: `0.5 ** (age_periods / halflife_periods)`. Ничего не выбрасывается — в отличие от жёсткого скользящего окна, старые строки просто реже выбираются. Итог — среднее нормированных рангов (как в `EasyEnsembleClassifier`).

**Когда использовать:** панельные данные клиент × период, где свежие наблюдения важнее старых, но полный рефит на скользящем окне рискует выбросить сезонные/периодически повторяющиеся паттерны совсем.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_estimators` | 10 | Число базовых моделей |
| `halflife_periods` | 6.0 | Период полураспада веса свежести (в единицах `period_unit`) |
| `period_unit` | `'M'` | Pandas frequency alias для бинования datetime `ts_key`; игнорируется для числового `ts_key` |
| `sample_frac` | 1.0 | Доля от n_train строк в каждом бутстрэпе (с возвращением) |
| `base` | `'catboost'` | `'lightgbm'` или `'catboost'` |
| `base_params` | None | Параметры базовой модели |
| `random_seed` | 42 | Зерно (независимые генераторы через `SeedSequence.spawn`) |

**Атрибуты после fit:**
- `estimator_scores_` — val PR-AUC каждой базовой модели
- `ensemble_score_` — val PR-AUC ансамбля

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import WeightedBaggingByRecency

model = WeightedBaggingByRecency(n_estimators=10, halflife_periods=6)
model.fit(X_train, y_train, X_valid, y_valid, ts_key=X_train_report_dates, selected_features=feats)
print(f'Ансамбль: {model.ensemble_score_:.4f}')
```

---

## SnapshotEnsembleClassifier

**Файл:** `snapshot_ensemble.py`

Обучает **одну** CatBoost-модель, затем берёт «снэпшоты» предсказаний по префиксам деревьев на долях `snapshot_fracs` от итогового числа деревьев (`tree_count_`) — без переобучения, через `ntree_end` в `predict_proba`. Снэпшоты усредняются простым средним (все — префиксы одного бустинга, шкала предсказаний одна и та же, ранговая нормализация не нужна).

**Когда использовать:** бюджет на обучение — одна модель, а ансамблевое разнообразие всё же хочется получить почти бесплатно.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `snapshot_fracs` | `[0.4, 0.6, 0.8, 1.0]` | Доли от `tree_count_` для срезов |
| `base_params` | None | Параметры единственного CatBoost |
| `calibrate` | True | Изотоническая калибровка усреднённых вероятностей |
| `random_seed` | 42 | Зерно CatBoost |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import SnapshotEnsembleClassifier

model = SnapshotEnsembleClassifier(snapshot_fracs=[0.5, 0.75, 1.0])
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.tree_counts_, model.snapshot_scores_)
```

---

## MonotonicConstrainedClassifier

**Файл:** `monotonic_constrained.py`

Одна модель (LightGBM или CatBoost) с `monotone_constraints` по имени признака: `{'trans_sum__level': 1, 'inactive_streak': -1}` — скор обязан не убывать/не возрастать со значением признака, независимо от того, что выучил бы бустинг на шумных данных. Ограничения не тюнятся Optuna — это доменное знание, а не гиперпараметр качества.

**Когда использовать:** модель должна быть защитима перед бизнесом/регулятором («почему у клиента с бо́льшим оборотом скор ниже?») и устойчива к контринтуитивным сплитам на шумных/малых данных.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `monotone_constraints` | обязателен | `{имя_признака: ±1}` |
| `base` | `'lightgbm'` | `'lightgbm'` или `'catboost'` |
| `base_params` | None | Параметры базовой модели |
| `random_seed` | 42 | Зерно базовой модели |

**Атрибуты после fit:**
- `model_score_` — val PR-AUC

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import MonotonicConstrainedClassifier

model = MonotonicConstrainedClassifier(monotone_constraints={'trans_sum__level': 1})
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.model_score_)
```

---

## TimeAwareValidationClassifier

**Файл:** `time_aware_validation.py`

Расширяющееся окно (walk-forward) по `ts_key` вместо одного train/valid/test cutoff: сортированные уникальные периоды делятся на `n_windows + 1` блок, каждый следующий блок по очереди становится validation-окном, train для него — все периоды до этого блока за вычетом `embargo_periods` периодов непосредственно перед ним (purge/embargo). Архитектура тюнится Optuna один раз на последнем (самом полном) окне и переиспользуется во всех окнах. Итоговая модель — модель последнего окна.

**Когда использовать:** метка зависит от будущего (сегментация/лейбл присваивается с лагом относительно события — как `last_full_month_before_it`-подобная логика) и/или один train/valid/test cutoff даёт шумную, невоспроизводимую оценку и подбор гиперпараметров (закрывает [M2] из `plan.txt`).

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_windows` | 5 | Число последовательных validation-окон |
| `embargo_periods` | 1 | Периодов, исключаемых из train перед каждым окном |
| `period_unit` | `'M'` | Pandas frequency alias для бинования datetime `ts_key` |
| `base` | `'catboost'` | `'lightgbm'` или `'catboost'` |
| `base_params` | None | Параметры базовой модели |

**Атрибуты после fit:**
- `window_scores_` — val PR-AUC каждого окна
- `window_bounds_` — границы train/val каждого окна (для диагностики/графиков)
- `oof_score_` — PR-AUC по объединённым out-of-window предсказаниям всех окон
- `final_estimator_` — модель последнего окна (используется в `predict_proba`)

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import TimeAwareValidationClassifier

model = TimeAwareValidationClassifier(n_windows=5, embargo_periods=1)
model.fit(X, y, ts_key=X['REPORT_DATE'], selected_features=feats)
print(model.window_scores_, model.oof_score_)
```

---

## StabilitySelectionClassifier

**Файл:** `stability_selection.py`

Обучает `n_bootstrap` лёгких CatBoost на стратифицированных (по классу) бутстрэп-подвыборках train, на каждой смотрит топ-`top_k` признаков по важности. Признак попадает в стабильное ядро `stable_features_`, если он оказался в топе хотя бы в `freq_threshold` доле бутстрэпов. Финальная модель обучается один раз на полном train, но только по стабильному ядру.

**Когда использовать:** важность признаков заметно скачет между запусками (типично при сотнях коррелированных признаков), а бизнесу нужен воспроизводимый, интерпретируемый набор фич — а не максимум PR-AUC любой ценой.

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `n_bootstrap` | 50 | Число бутстрэп-повторов |
| `top_k` | 20 | Размер топа важности на каждом бутстрэпе |
| `freq_threshold` | 0.6 | Мин. доля бутстрэпов для попадания в ядро |
| `bootstrap_params` | None | Параметры быстрых CatBoost для важностей |
| `final_params` | None | Параметры финальной CatBoost-модели |
| `calibrate` | True | Изотоническая калибровка финальных вероятностей |

**Пример:**
```python
from ml_toolkit.presets.classification.high_pr_auc import StabilitySelectionClassifier

model = StabilitySelectionClassifier(n_bootstrap=50, top_k=20, freq_threshold=0.6)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.stable_features_)
print(model.selection_freq_.head(10))   # частота попадания в топ по убыванию
```

---

## Дрифт и adversarial validation (DriftRobustClassifier / AdversarialValidationWeighting)

Две стратегии реакции на смещение train/valid (train и inference разнесены
во времени, признаки дрейфуют): выбросить дрейфующие **колонки** целиком, или
оставить их и перевзвесить **строки** train так, чтобы они были больше
похожи на valid.

### DriftRobustClassifier

**Файл:** `drift_robust.py`

Связывает уже существующий `ml_toolkit.feature_selection.AdversarialDriftFilter`
(итеративно удаляет признаки, по которым train/valid легко различимы) с
обучением: удалённые признаки модель просто не видит. `compute_psi` даёт
параллельный (без обучения adversarial-модели) диагностический отчёт.

**Когда использовать:** train и inference разнесены во времени, признаки
дрейфуют, и дрейфующие признаки не критично важны — можно потерять.

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `target_auc` | 0.55 | Целевой adversarial AUC для AdversarialDriftFilter |
| `base_preset` | None | Необученный объект с интерфейсом BasePreset. None → внутренний CatBoost |

**Атрибуты после fit:** `removed_features_`, `adversarial_auc_history_`, `psi_report_`.

```python
from ml_toolkit.presets.classification.high_pr_auc import DriftRobustClassifier

model = DriftRobustClassifier(target_auc=0.55)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(model.removed_features_)
```

### AdversarialValidationWeighting

**Файл:** `adversarial_weighting.py`

Adversarial-классификатор train(0) vs valid(1) → вес train-строки
`p(valid|x)/p(train|x)` (odds), клип в `clip_weights` и нормализация к
среднему 1.0 → финальная модель обучается на train с этими весами
(`sample_weight`).

**Когда использовать:** дрейф есть, но дрейфующие признаки слишком ценны,
чтобы их выбрасывать (в отличие от DriftRobustClassifier).

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `clip_weights` | (0.2, 5.0) | Диапазон клипа весов до нормализации |

**Атрибуты после fit:** `adversarial_auc_` (honest AUC diagnostic), `weights_`, `weight_stats_`.

**Важно:** при очень сильном и равномерном смещении (все train-строки
одинаково "непохожи" на valid) веса после клипа могут схлопнуться в
одинаковые (клип-порог насыщается для всех) — это ожидаемое поведение клипа,
а не баг; в этом случае DriftRobustClassifier может быть информативнее.

```python
from ml_toolkit.presets.classification.high_pr_auc import AdversarialValidationWeighting

model = AdversarialValidationWeighting(clip_weights=(0.2, 5.0))
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
print(f"adversarial AUC={model.adversarial_auc_:.3f}")
```

---

## Комбинирование пресетов

`CalibratedWrapper` и `ThresholdMovingCV` — обёртки; внутрь можно вложить любой пресет:

```python
from ml_toolkit.presets.classification.high_pr_auc import (
    ThresholdMovingCV, CalibratedWrapper, EasyEnsembleClassifier
)

# Ансамбль → калибровка → оптимальный порог
model = ThresholdMovingCV(
    CalibratedWrapper(
        EasyEnsembleClassifier(n_estimators=12, neg_ratio=10),
        method='isotonic',
    ),
    optimize='f2',
)
model.fit(X_train, y_train, X_valid, y_valid, selected_features=feats)
# model.base_.calibrated_pr_auc_  ← PR-AUC после калибровки
# model.threshold_                ← найденный порог
```

```python
# LambdaRank + порог по recall
model = ThresholdMovingCV(
    LambdaRankClassifier(),
    optimize='precision_at_recall',
    min_recall=0.65,
)
```

---

## Диагностический чек-лист

| Симптом | Диагноз | Пресет |
|---------|---------|--------|
| PR-AUC низкий, ROC-AUC тоже | Нет сигнала в признаках | Признаки, а не пресет |
| PR-AUC низкий, ROC-AUC ≥ 0.75 | Surrogate расходится с PR-AUC | `LambdaRankClassifier` |
| Многие вероятности > 0.3 | Завышенные вероятности от undersampling | `CalibratedWrapper` |
| Recall падает при высоком precision | Порог 0.5 неоптимален | `ThresholdMovingCV` |
| Мало позитивов (< 150 в train) | Недостаточно сигнала для обучения | `SyntheticOversamplingClassifier` |
| Focal Loss не помог | Нужна асимметрия по классам | `AsymmetricLossClassifier` |
| Time split: val positives > train positives | PU-структура данных | `PULearningClassifier` |
| Сегментация менялась во времени | Незамеченные позитивы в train | `SelfTrainingBooster` |
| Все пресеты дают одинаковый результат | Ортогональные сигналы | `AnomalyBlendClassifier` |
