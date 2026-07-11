# Поддерживаемые модели

## Структура адаптеров

Каждая модель реализована в отдельном файле `ml_toolkit/models/_{name}.py` и экспортирует пару классов `XxxRegressor`/`XxxClassifier`, наследующих `BaseModel` (`ml_toolkit/models/_base.py`) — единый контракт `fit(X_train, y_train, X_valid=None, y_valid=None, selected_features=None, cat_features=None)` / `predict(X)` / `predict_proba(X)`:

```python
from ml_toolkit.models import LightGBMRegressor

model = LightGBMRegressor(params={...})            # или n_optuna_trials=N, params=None
model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...], cat_features=[...])
pred = model.predict(X_new)
# После fit(): model.best_params_, model.selected_features_, model.cat_features_,
# model.train_pred_, model.valid_pred_
```

`model._model` — обученный «сырой» объект (произвольная структура: для tree-моделей это сам estimator; для линейных — кортеж `(estimator, prep_pipeline, num_feature_names)`; для TabM — `(model, prep, y_stats)` / `(model, prep)`).

---

## CatBoost (`catboost`)

### Как работает

Gradient boosting на симметричных деревьях (Oblivious Decision Trees — ODT): каждый уровень дерева использует одно условие разбиения для всех узлов уровня. Ключевая особенность — **ordered boosting**: для каждого объекта градиент считается на случайном подмножестве данных, не включающем этот объект. Это снижает смещение оценки градиента и уменьшает target leakage. Категориальные признаки обрабатываются через ordered target statistics без ручного кодирования.

### Когда полезен

- Датасет содержит категориальные признаки с высокой кардинальностью — нативная обработка даёт прирост по сравнению с ручным OHE или ordinal encoding.
- Небольшие и средние датасеты (до ~5M строк), где ordered boosting снижает overfitting.
- Задачи, где важна воспроизводимость и устойчивость без глубокого тюнинга.
- Требуется быстрый инференс — симметричные деревья векторизуются эффективно.

### Плюсы

- Нативные категориальные признаки без preprocessing.
- Ordered boosting снижает overfitting, особенно на небольших датасетах.
- Быстрый инференс за счёт ODT-структуры.
- SHAP TreeExplainer работает без дополнительных обёрток.
- Встроенная защита от target leakage при работе с cat-признаками.

### Минусы

- На очень больших датасетах медленнее LightGBM — гистограммный подход LGB эффективнее.
- Ordered boosting иногда даёт более консервативные оценки; LGB/XGB лучше тюнируются при достаточном объёме данных.
- Нет аналитического решения — всегда итеративная оптимизация.
- Чувствителен к числу итераций: без early stopping легко переобучиться.

### Best practices

- Всегда передавать `cat_features` явно — это основное конкурентное преимущество перед LGB/XGB.
- `baseline` через `Pool(baseline=...)` позволяет модели учиться на остатках от известного прогноза, что ускоряет сходимость.
- `grow_policy='Lossguide'` + `max_leaves` вместо `depth` для более гибких деревьев при большом числе признаков.
- Оптимальный диапазон `depth`: 4–8. Глубже — overfitting, мельче — underfitting.
- `l2_leaf_reg` > 3 на зашумлённых данных; `random_strength` > 0 при малом объёме выборки.
- Early stopping с 50–100 раундами стандартен.

### Литература

1. **[CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)** — Prokhorenkova et al., NeurIPS 2018. *О чём:* Основная статья. Формализует понятие ordered boosting и ordered target statistics для категориальных признаков; доказывает смещённость классического gradient boosting и предлагает решение. *Польза:* Объясняет, почему `cat_features` в CatBoost работают лучше OHE, и как ordered boosting снижает overfitting — помогает понять, когда выбирать CatBoost вместо LGB/XGB.

2. **[CatBoost: gradient boosting with categorical features support](https://arxiv.org/abs/1810.11363)** — Dorogush et al., workshop paper 2018. *О чём:* Более короткий технический обзор с акцентом на инженерные решения: ODT-деревья, CPU/GPU-ускорение, бенчмарки против XGBoost и LightGBM. *Польза:* Даёт практическое сравнение скоростей и качества на реальных датасетах; помогает принять решение о выборе модели по соотношению скорость/качество.

3. **[Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)** — Friedman, Annals of Statistics, 2001. *(не на arxiv)* *О чём:* Основополагающая статья, которая формализует gradient boosting как покоординатный спуск в функциональном пространстве. Вводит понятие loss function, weaklearner, shrinkage. *Польза:* Читать для понимания математической основы всех GBDT-методов — без этого сложно интерпретировать гиперпараметры learning_rate, n_estimators, subsample.

4. **[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** — Lundberg & Lee, NeurIPS 2017. *О чём:* Вводит SHAP (SHapley Additive exPlanations) — унифицированную теорию объяснений предсказаний на основе теории игр Шепли. Показывает, что LIME, DeepLIFT и другие методы — частные случаи SHAP. *Польза:* Теоретическое обоснование графиков feature importance и waterfall; понимание этой статьи позволяет правильно интерпретировать SHAP-значения.

5. **[Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888)** — Lundberg et al., 2018. *О чём:* TreeSHAP — полиномиальный алгоритм точного вычисления SHAP-значений для деревьев. До него SHAP для деревьев вычислялся приближённо за экспоненциальное время. *Польза:* Объясняет, почему SHAP для CatBoost/LGB/XGB работает быстро и точно — без этого алгоритма интерпретация дерева на больших датасетах была бы неосуществима.

6. **[From Local Explanations to Global Understanding with Explainable AI for Trees](https://arxiv.org/abs/1905.04610)** — Lundberg et al., Nature Machine Intelligence, 2020. *О чём:* Расширяет SHAP до глобальных объяснений: SHAP interaction values, dependence plots, summary plots. Показывает, как агрегировать локальные объяснения в глобальные паттерны. *Польза:* Практическое руководство по построению beeswarm и dependence plots; объясняет разницу между SHAP mean |value| и SHAP interaction importance.

7. **[Benchmarking State-of-the-Art Gradient Boosting Algorithms for Classification](https://arxiv.org/abs/2305.17094)** — Grinsztajn et al., 2023. *О чём:* Детальное сравнение CatBoost, LightGBM и XGBoost на > 100 табличных датасетах. Анализирует влияние категориальных признаков, дисбаланса классов, размера датасета на относительное качество трёх бустинговых алгоритмов. *Польза:* Конкретные рекомендации, когда CatBoost выигрывает у LGB (высокая кардинальность категорий, маленький датасет), а когда LGB быстрее без потери качества.

8. **[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312)** — Popov et al., ICLR 2020. *О чём:* NODE — нейросеть, имитирующая ансамбль Oblivious Decision Trees (ODT), аналогичных тем, что использует CatBoost. Обучается backpropagation; дифференцируемые сплиты через entmax. *Польза:* Альтернативный взгляд на ODT-архитектуру CatBoost через призму нейросетей; объясняет, почему симметричные деревья векторизуются эффективно и при каких условиях NODE может превзойти CatBoost.

---

## LightGBM (`lightgbm`)

### Как работает

Gradient boosting с **гистограммным подходом**: непрерывные признаки дискретизируются в бины, что ускоряет поиск оптимального сплита с O(data) до O(bins). **Leaf-wise growth**: на каждом шаге растёт лист с максимальным приростом функции потерь — в отличие от level-wise у XGBoost, это даёт более глубокие асимметричные деревья при том же числе листьев. **GOSS** (Gradient-based One-Side Sampling) — обучение только на объектах с большим градиентом, остальные сэмплируются. **EFB** (Exclusive Feature Bundling) — объединение разреженных признаков в бандлы.

### Когда полезен

- Большие датасеты (от миллиона строк) — гистограммный метод даёт ~10x ускорение vs XGBoost на level-wise growth.
- Когда скорость обучения критична (много итераций Optuna, частый retraining).
- Много числовых признаков с высокой кардинальностью.
- Задачи, где объём данных не помещается в RAM — поддержка chunked training.

### Плюсы

- Самый быстрый из тройки GBDT на больших данных.
- Низкое потребление памяти за счёт бинаризации.
- GOSS и EFB дают дополнительное ускорение без заметной потери качества.
- Нативная MAE-оптимизация (`objective='regression_l1'`).
- `init_score` позволяет задавать baseline предсказание.

### Минусы

- Leaf-wise growth склонен к overfitting на маленьких датасетах — `num_leaves` и `min_data_in_leaf` требуют аккуратной настройки.
- Категориальные признаки обрабатываются хуже, чем в CatBoost (Fisher's optimal split вместо ordered statistics).
- В sklearn API (LGBMRegressor) `.booster_` скрыт — нужен `model.booster_` для SHAP и feature importance.

### Best practices

- `num_leaves` — ключевой параметр: 31 (default) мало для сложных задач; 63–511 — типичный range.
- `min_child_samples` (min_data_in_leaf) не менее 20 для предотвращения overfitting.
- `subsample` + `colsample_bytree` < 1.0 совместно дают сильную регуляризацию.
- `objective='regression_l1'` для прямой оптимизации MAE.
- `boosting_type` (`gbdt` / `dart` / `goss`) выбирается Optuna автоматически; `dart` — dropout деревьев, снижает overfitting, но инференс медленнее; `goss` — ускоряет обучение на больших данных.
- Уменьшить `learning_rate` + увеличить `n_estimators` + early stopping — классический рецепт повышения качества без overfitting.

### Литература

1. **[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)** — Ke et al., NeurIPS 2017. *(официального arxiv-препринта нет)* *О чём:* Основная статья. Вводит GOSS (выборка по градиенту) и EFB (бандлинг разреженных признаков). Показывает теоретические гарантии точности GOSS и ускорение на практике. *Польза:* Объясняет, почему LightGBM быстрее при меньшем потреблении памяти — помогает правильно настроить `bagging_fraction` и понять, когда GOSS ухудшает качество.

2. **[Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)** — Friedman, Annals of Statistics, 2001. *(не на arxiv)* *О чём:* Основополагающая статья gradient boosting (см. раздел CatBoost). *Польза:* Базовое понимание, общее для всех GBDT-адаптеров.

3. **[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** — Lundberg & Lee, NeurIPS 2017. *О чём:* SHAP (см. раздел CatBoost). *Польза:* Актуально для всех tree-моделей.

4. **[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)** — Grinsztajn et al., NeurIPS 2022. *О чём:* Систематическое сравнение GBDT (в том числе LightGBM) и нейросетей на 45 табличных датасетах. Анализирует, какие свойства данных определяют победителя. *Польза:* Аргументированный ответ на вопрос «когда выбирать LGB/CatBoost, а когда нейросеть»; полезно при выборе модели для нового датасета.

5. **[Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473(01)00065-2)** — Friedman, Computational Statistics & Data Analysis, 2002. *(не на arxiv)* *О чём:* Вводит subsampling строк (stochastic gradient boosting) — обучение каждого дерева на случайной подвыборке. Показывает улучшение обобщения и снижение дисперсии ансамбля. *Польза:* Теоретическое обоснование параметров `subsample` и `bagging_fraction` в LightGBM; понимание, почему значение < 1.0 даёт лучшую регуляризацию, чем полный датасет.

6. **[From Local Explanations to Global Understanding with Explainable AI for Trees](https://arxiv.org/abs/1905.04610)** — Lundberg et al., Nature Machine Intelligence, 2020. *О чём:* SHAP interaction values и глобальные summary plots для tree-ансамблей (см. CatBoost). *Польза:* Интерпретация признаков и их попарных взаимодействий в моделях LightGBM.

---

## XGBoost (`xgboost`)

### Как работает

Gradient boosting с **регуляризованной целевой функцией**: L1 и L2 штрафы встроены непосредственно в loss (не как постобработка). **Level-wise growth** по умолчанию — дерево строится уровень за уровнем. Поддерживает `tree_method='hist'` (аналог LightGBM) и точный greedy algorithm. `base_score` задаёт глобальный сдвиг предсказаний до начала boosting.

### Когда полезен

- Задачи, где нужен точный контроль регуляризации через явные L1/L2 штрафы.
- Как участник стекинга — другой inductive bias по сравнению с LGB и CatBoost.
- GPU-обучение (`tree_method='gpu_hist'`) — конкурентно с LightGBM.
- Разреженные входные матрицы (встроенная поддержка sparse format).

### Плюсы

- Встроенная L1+L2 регуляризация в функции потерь — точный контроль сложности модели.
- `tree_method='hist'` обеспечивает конкурентную скорость с LightGBM.
- Стабильная работа с разреженными матрицами.
- Отличная документация и широкое сообщество.
- `monotone_constraints` — монотонность по признакам из prior знаний.

### Минусы

- Level-wise growth медленнее leaf-wise на равном числе листьев при сопоставимой глубине.
- Нет нативной поддержки категориальных признаков (только числовое кодирование).
- На очень больших датасетах уступает LightGBM по скорости при стандартных настройках.

### Best practices

- `tree_method='hist'` обязателен для датасетов > 100k строк.
- `alpha` (L1) обнуляет малозначимые признаки в деревьях — полезно при большом числе фичей.
- `gamma` (минимальный прирост для разбиения) — мягкая регуляризация сложности дерева.
- `scale_pos_weight` для классификации при сильном дисбалансе классов.
- `eval_metric` + `early_stopping_rounds` обязательны.
- `monotone_constraints` при наличии доменных знаний о знаке зависимости.

### Литература

1. **[XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)** — Chen & Guestrin, KDD 2016. *О чём:* Основная статья. Описывает регуляризованную целевую функцию, approximate split finding через weighted quantile sketch, sparsity-aware split finding для пропущенных значений и гистограммный метод. *Польза:* Объясняет значение гиперпараметров `gamma`, `lambda`, `alpha` — без этого тюнинг превращается в перебор; также объясняет, почему XGBoost устойчив к разреженным данным.

2. **[Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473(01)00065-2)** — Friedman, Computational Statistics & Data Analysis, 2002. *(не на arxiv)* *О чём:* Вводит subsampling (stochastic gradient boosting) — обучение каждого дерева на случайной подвыборке данных. Показывает улучшение обобщения и ускорение. *Польза:* Теоретическое обоснование параметров `subsample` и `colsample_bytree`; понимание, почему значения < 1.0 дают лучшую регуляризацию.

3. **[Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888)** — Lundberg et al., 2018. *О чём:* TreeSHAP (см. раздел CatBoost). *Польза:* Актуально для всех tree-моделей.

4. **[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)** — Grinsztajn et al., NeurIPS 2022. *О чём:* Систематическое сравнение (см. раздел LightGBM). *Польза:* Актуально для всех tree-моделей.

5. **[From Local Explanations to Global Understanding with Explainable AI for Trees](https://arxiv.org/abs/1905.04610)** — Lundberg et al., Nature Machine Intelligence, 2020. *О чём:* SHAP interaction values и глобальные объяснения для tree-ансамблей (см. CatBoost). *Польза:* Анализ feature interactions в XGBoost-моделях через SHAP.

6. **[Additive Logistic Regression: A Statistical View of Boosting](https://doi.org/10.1214/aos/1016218223)** — Friedman, Hastie & Tibshirani, Annals of Statistics, 2000. *(не на arxiv)* *О чём:* Интерпретирует AdaBoost как аппроксимацию максимизации log-likelihood; вводит понятие «statistical view of boosting» — связь между boosting и GLM с аддитивной структурой. *Польза:* Глубокое понимание связи XGBoost с обобщёнными линейными моделями; объясняет, почему XGBoost с MSE-loss — частный случай gradient descent в функциональном пространстве.

7. **[Benchmarking State-of-the-Art Gradient Boosting Algorithms for Classification](https://arxiv.org/abs/2305.17094)** — Grinsztajn et al., 2023. *О чём:* Детальное сравнение CatBoost, LightGBM и XGBoost на 100+ датасетах (см. CatBoost). *Польза:* Конкретные условия, при которых XGBoost с `tree_method='hist'` конкурентен LGB.

---

## LightAutoML / LAMA (`lama`)

### Как работает

AutoML-фреймворк, автоматически строящий ML-пайплайн: подбирает типы моделей, preprocessing, feature engineering и строит ансамбль из нескольких базовых моделей (LightGBM + logistic regression + MLP по умолчанию). Оптимизирует по заданной метрике в рамках бюджета времени (timeout). Ввод — Pandas DataFrame с указанием роли каждого признака через `roles`.

### Когда полезен

- Быстрая оценка верхней границы качества без ручного тюнинга — upper bound benchmark.
- Новая задача с неизвестными паттернами: LAMA находит рабочую конфигурацию автоматически.
- Как oracle: если LAMA не превосходит ручные модели, ручной пайплайн хорошо настроен.
- Baseline для стекинга — LAMA-предсказания как признак для meta-модели.

### Плюсы

- Минимум ручной настройки — нужны только timeout и тип задачи.
- Внутренний ансамбль часто превосходит одиночные модели.
- Встроенный feature selection.
- Единый API для регрессии и классификации.

### Минусы

- Чёрный ящик: состав ансамбля и веса компонент непрозрачны.
- SHAP не поддерживается — только permutation importance.
- Нестабильность: два запуска с одинаковыми параметрами дают разные результаты (timeout-based).
- Медленный даже при небольшом timeout из-за overhead AutoML-фреймворка.
- Сложная установка зависимостей (специфичные версии torch и numpy).

### Best practices

- `timeout` 120–300 с для реального использования; 30–60 с в тестах.
- `task='reg'` + `loss='mae'`; `task='binary'` + `metric='auc'` для классификации.
- Передавать `roles` с явным указанием типов признаков — улучшает автоматический preprocessing.
- Использовать как one-shot benchmark, не в production retraining loop.
- При нестабильных результатах — усреднять несколько запусков или увеличить timeout.

### Литература

1. **[LightAutoML: AutoML Solution for a Large Financial Services Ecosystem](https://arxiv.org/abs/2109.01528)** — Vakhrushev et al., 2021. *О чём:* Описывает архитектуру LAMA: многоуровневые пайплайны, AutoML-стратегии выбора модели, блоки linear/GBDT/NN, задачи stacking и blending. Включает production-кейсы из Сбера. *Польза:* Понимание внутреннего устройства помогает правильно интерпретировать результаты и диагностировать причины нестабильности; объясняет, когда LAMA выигрывает у ручных пайплайнов.

2. **[AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1906.02287)** — He et al., 2021. *О чём:* Обзор всего поля AutoML: Neural Architecture Search, HPO, meta-learning, pipeline composition. LAMA относится к классу pipeline-composition AutoML. *Польза:* Системное понимание, чем AutoML отличается от ручного тюнинга и каковы его теоретические пределы; полезно при оценке результатов LAMA.

3. **[Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)** — Akiba et al., KDD 2019. *О чём:* Описывает Optuna — фреймворк для гиперпараметрической оптимизации на основе TPE (Tree-structured Parzen Estimator) с define-by-run API. Используется во всех адаптерах проекта. *Польза:* Понимание алгоритма TPE объясняет, почему Optuna эффективен при малом числе trials и как правильно задавать search space для лучшей сходимости.

4. **[Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning](https://arxiv.org/abs/2007.04074)** — Feurer et al., JMLR 2022. *О чём:* Вторая версия Auto-sklearn — конкурент LAMA. Использует meta-learning для warm-start: выбирает начальные конфигурации на основе схожих датасетов из истории. Portfolio-based подход. *Польза:* Сравнение архитектурных решений LAMA и Auto-sklearn; понимание, почему meta-learning ускоряет сходимость AutoML и как LAMA реализует аналогичные идеи без явного meta-learning.

5. **[FLAML: A Fast and Lightweight AutoML Library](https://arxiv.org/abs/2005.01571)** — Wang et al., MLSys 2021. *О чём:* FLAML — AutoML с cost-frugal HPO: динамически распределяет бюджет между конфигурациями в зависимости от предсказанной стоимости. Особенно эффективен при ограниченном compute. *Польза:* Альтернатива LAMA для случаев с жёсткими ограничениями по времени; сравнение подходов LAMA (timeout-based) и FLAML (cost-aware) помогает выбрать инструмент.

6. **[TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791)** — Salinas et al., 2025. *О чём:* Живой benchmark (обновляемый) GBDT, AutoML и DL методов на 60+ датасетах. Включает CatBoost, LightGBM, AutoGluon, LAMA, TabPFN. Методически строгий протокол сравнения. *Польза:* Актуальные результаты сравнения LAMA с современными альтернативами; помогает понять, где AutoML-подход выигрывает у ручного пайплайна в 2024–2025 году.

---

## TabM (`tabm`)

### Как работает

**Bag of k-headed MLPs с BatchEnsemble**: единый backbone MLP, от которого отходят k независимых «головок» через покомпонентное умножение на обучаемые векторы масштабирования (BatchEnsemble). Это имитирует ансамбль из k моделей при минимальном overhead по числу параметров — только k × d масштабирующих векторов добавляются к d параметрам backbone. Предсказание — среднее по k головкам (регрессия) или Sigmoid + среднее (классификация).

### Когда полезен

- Нелинейные и периодические зависимости, которые плохо улавливаются деревьями.
- Большие датасеты (> 100k строк) — нейросети раскрываются на объёме.
- Как независимый голос в стекинге (другой inductive bias, чем у GBDT).
- Задачи с плотными embedding-признаками или сложными взаимодействиями.

### Плюсы

- k-headed ensemble в одном forward pass — дешевле чем k отдельных моделей.
- BatchNorm + Dropout обеспечивают регуляризацию.
- GPU-ускорение — важно при больших датасетах.
- Единый API с остальными адаптерами.

### Минусы

- Требует установки `tabm` пакета и PyTorch.
- Медленнее деревьев на табличных данных малого и среднего размера.
- SHAP не поддерживается.
- Нестабильность инициализации — результаты варьируются между запусками.
- На CPU практически неприменим в production из-за скорости.

### Best practices

- `device='cuda'` при наличии GPU; на CPU ограничить `n_epochs_final` ≤ 50.
- `k=32` — баланс качество/скорость; `k=64` даёт прирост при наличии GPU.
- `patience=10–20` с early stopping по валидационной метрике.
- Нормализация таргета критична для сходимости (встроена в адаптер через y_stats).
- Минимум 10 Optuna trials для подбора архитектуры; 1 trial в тестах.
- Не использовать совместно с `fast_optuna` фикстурой — она патчит глобальный Optuna.

### Литература

1. **[TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)** — Gorishniy et al., 2024. *О чём:* Основная статья TabM. Описывает архитектуру BatchEnsemble для таблиц, сравнивает с FT-Transformer, ResNet и GBDT на обширном бенчмарке. Вводит понятие parameter-efficient ensembling для tabular DL. *Польза:* Объясняет, почему k головок лучше одной, как выбирать k и d_block; бенчмарк показывает на каких датасетах TabM опережает CatBoost, а на каких проигрывает.

2. **[BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning](https://arxiv.org/abs/2002.06715)** — Wen et al., ICLR 2020. *О чём:* Предлагает BatchEnsemble — ансамблирование через покомпонентное умножение весов на обучаемые ранг-1 матрицы. Изначально предложен для CV и NLP, адаптирован в TabM для таблиц. *Польза:* Понимание математической основы архитектуры TabM; объясняет, почему BatchEnsemble эффективнее простого дублирования параметров модели.

3. **[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)** — Gorishniy et al., NeurIPS 2021. *О чём:* Вводит FT-Transformer для таблиц и ResNet-baseline; систематически сравнивает архитектуры нейросетей с GBDT. Показывает, что простые MLP с правильным preprocessing конкурентны сложным трансформерам. *Польза:* Рекомендации по preprocessing (QuantileTransformer, OrdinalEncoder) — те же, что используются в _tabm.py; помогает понять, почему именно этот preprocessing выбран.

4. **[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)** — Grinsztajn et al., NeurIPS 2022. *О чём:* 45 датасетов, 4 типа моделей. Выявляет, что GBDT лучше на данных с нерегулярными паттернами и неинформативными признаками; нейросети лучше на данных с плавными зависимостями. *Польза:* Систематические критерии выбора между TabM и CatBoost/LGB; объясняет, почему на большинстве табличных задач GBDT побеждает без тщательного тюнинга нейросети.

5. **[On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)** — Gorishniy et al., NeurIPS 2022. *О чём:* Изучает, как кодировать числовые признаки для нейросетей: piecewise-linear encoding, periodic encoding, QuantileTransformer. Показывает значительный прирост от правильного числового embedding. *Польза:* Обоснование использования QuantileTransformer в адаптере; если TabM даёт плохие результаты — в первую очередь проверить preprocessing числовых признаков.

6. **[TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848)** — Hollmann et al., ICLR 2023. *О чём:* Трансформер, предобученный на синтетических данных (prior-data fitted networks). Работает без тюнинга: один forward pass даёт предсказание для малых датасетов (< 1000 строк). *Польза:* Ориентир для сравнения TabM: на малых датасетах TabPFN часто точнее без тюнинга; показывает, что meta-learned prior может заменить gradient descent при малом n.

7. **[SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342)** — Somepalli et al., 2021. *О чём:* SAINT применяет self-attention как по столбцам (признакам), так и по строкам (объектам) — inter-sample attention. Contrastive pre-training улучшает обобщение. *Польза:* Альтернатива TabM с row-attention механизмом; сравнение с TabM показывает условия, при которых внимание по объектам даёт прирост.

8. **[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312)** — Popov et al., ICLR 2020. *О чём:* NODE — нейросеть из дифференцируемых ODT-слоёв. Сочетает интерпретируемость деревьев с обучением backpropagation. Исторически — один из первых DL-методов, превзошедших GBDT на части датасетов. *Польза:* Контекст для оценки TabM: NODE — предшественник идеи DL на таблицах; понимание его ограничений (медленное обучение, сложная реализация) объясняет, зачем появился TabM.

9. **[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)** — Srivastava et al., JMLR, 2014. *О чём:* Базовая статья Dropout: случайное обнуление нейронов при обучении как регуляризация. Интерпретируется как implicit ансамблирование экспоненциального числа субсетей. *Польза:* Теоретическое основание Dropout в backbone-сети TabM; объясняет, как Dropout взаимодействует с BatchEnsemble — два разных вида ансамблирования в одной модели.

10. **[TabArena: A Living Benchmark for Machine Learning on Tabular Data](https://arxiv.org/abs/2506.16791)** — Salinas et al., 2025. *О чём:* Актуальный benchmark 2025 года; включает TabM, CatBoost, AutoGluon, TabPFN и другие. *Польза:* Самые свежие сравнительные результаты TabM на разнообразных датасетах.

---

## Ridge (`ridge`)

### Как работает

Линейная регрессия с L2-регуляризацией: минимизирует `||y − Xw||² + α||w||²`. Имеет аналитическое решение: `w = (XᵀX + αI)⁻¹Xᵀy`. Регуляризация сжимает все коэффициенты к нулю пропорционально `α`, но не обнуляет их. Loss — MSE.

### Когда полезен

- Быстрый интерпретируемый baseline: если Ridge близок к GBDT, задача линейно решаема.
- Высокая мультиколлинеарность признаков — Ridge стабилизирует оценки коэффициентов.
- Диагностика: `|coef_|` показывает, какие признаки несут линейный сигнал.
- Ситуации, где детерминизм и воспроизводимость критичны (нет случайности).

### Плюсы

- Чрезвычайно быстрый (аналитическое решение).
- Полностью детерминирован.
- Стабилен при мультиколлинеарности.
- Коэффициенты интерпретируемы в единицах StandardScaler («эффект одного σ»).

### Минусы

- Оптимизирует MSE — чувствителен к выбросам в таргете.
- Не обнуляет незначимые признаки: нет встроенного feature selection.
- Улавливает только линейные зависимости и не моделирует взаимодействия признаков.
- Требует ручного масштабирования и импутации NaN (встроено в адаптер).

### Best practices

- Запускать первым после построения нового набора признаков — быстро покажет наличие линейного сигнала.
- Сравнивать `|coef_|` Ridge с gain importance GBDT: совпадение топ-признаков — признак устойчивости сигнала.
- `alpha` в диапазоне [0.1, 100] при большом числе признаков; [0.01, 1] при небольшом.
- При Ridge ≈ GBDT по качеству — признаки уже кодируют всю нужную нелинейность.

### Литература

1. **[Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634)** — Hoerl & Kennard, Technometrics, 1970. *(не на arxiv)* *О чём:* Основополагающая статья, вводящая Ridge как решение проблемы мультиколлинеарности в МНК-регрессии. Доказывает существование `α > 0`, при котором MSE Ridge меньше МНК. *Польза:* Понимание, почему Ridge стабилизирует коэффициенты при мультиколлинеарности и как выбирать `alpha` на основе ridge trace — кривой коэффициентов в зависимости от `alpha`.

2. **[A Survey of Cross-Validation Procedures for Model Selection](https://arxiv.org/abs/0907.4728)** — Arlot & Celisse, Statistics Surveys, 2010. *О чём:* Обзор методов кросс-валидации для выбора гиперпараметров (в том числе `alpha` для Ridge). Рассматривает Leave-One-Out, k-fold, их bias-variance tradeoff. *Польза:* Обоснование использования валидационной выборки вместо LOO для тюнинга `alpha`; понимание, когда Optuna с hold-out набором достаточен, а когда нужна k-fold CV.

3. **[Scikit-learn: Machine Learning in Python](https://arxiv.org/abs/1201.0490)** — Pedregosa et al., JMLR 2011. *О чём:* Описывает sklearn: дизайн API, реализацию алгоритмов, включая Ridge, ElasticNet, LogisticRegression. Актуально для всех линейных адаптеров. *Польза:* Референс по реализации, поведению edge cases (singularity, max_iter, solver); полезно при отладке неожиданного поведения sklearn-моделей.

4. **[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)** — Hastie, Tibshirani & Friedman, Springer, 2009 (2nd ed.). *(free PDF)* *О чём:* Глава 3 — линейные методы регрессии: Ridge, Lasso, PCR, PLS. Геометрическая интерпретация L2-регуляризации, доказательство bias-variance tradeoff, ridge trace. *Польза:* Лучший математически строгий источник по теории Ridge; объясняет, почему Ridge устойчивее МНК при мультиколлинеарности и как выбирать `alpha` через ridge trace.

5. **[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)** — Bishop, Springer, 2006. *(free PDF от автора)* *О чём:* Глава 3 — байесовский взгляд на Ridge: L2-регуляризация как MAP-оценка при Gaussian prior на весах; связь `alpha` с дисперсией prior; предсказание с uncertainty. *Польза:* Пробрасывает мост от Ridge к BayesianRidge; объясняет, когда Ridge даёт недоопределённую задачу и как prior помогает её регуляризовать.

---

## ElasticNet (`elasticnet`)

### Как работает

Линейная регрессия с комбинацией L1 и L2 штрафов: `||y − Xw||² + α·l1_ratio·||w||₁ + α·(1−l1_ratio)/2·||w||²`. L1 часть порождает sparse решение (обнуляет коэффициенты нерелевантных признаков), L2 часть стабилизирует при мультиколлинеарности. При `l1_ratio=1` — Lasso, при `l1_ratio=0` — Ridge.

### Когда полезен

- Подозрение, что часть признаков нерелевантна — L1 автоматически обнулит их коэффициенты.
- Много коррелирующих признаков: L2 выбирает между ними, L1 удаляет лишние.
- Интерпретируемая альтернатива генетическому отбору признаков: результат — sparse набор фичей.
- Проверка: если ElasticNet использует 10–20 признаков из 90 — остальные можно дропнуть.

### Плюсы

- Встроенный feature selection через L1.
- Стабилен при мультиколлинеарности за счёт L2.
- Нулевые коэффициенты прямо указывают на нерелевантные признаки.

### Минусы

- Два гиперпараметра (`alpha`, `l1_ratio`) усложняют тюнинг.
- MSE loss — чувствителен к выбросам.
- При высоком `l1_ratio` и большом `alpha` может занулить важные признаки.
- Требует `max_iter=5000` при большом числе признаков (уже задано в адаптере).

### Best practices

- Начинать с `l1_ratio=0.5` (равный баланс L1/L2), затем тюнить.
- Если все коэффициенты нулевые — `alpha` слишком велик, уменьшить на порядок.
- Сравнивать ненулевые признаки ElasticNet с топом GBDT — пересечение означает устойчивый линейный сигнал.

### Литература

1. **[Regularization and Variable Selection via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x)** — Zou & Hastie, JRSS-B, 2005. *(не на arxiv)* *О чём:* Основная статья, вводящая ElasticNet как компромисс между Ridge и Lasso. Доказывает, что Lasso не может выбрать более n признаков при n < p, и что ElasticNet решает эту проблему. *Польза:* Объясняет, почему ElasticNet лучше чистого Lasso при p >> n и при мультиколлинеарности; обоснование параметра `l1_ratio` как инструмента настройки sparsity.

2. **[Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)** — Tibshirani, JRSS-B, 1996. *(не на arxiv)* *О чём:* Основополагающая статья Lasso (L1-регуляризация). Первое доказательство свойства sparse решений и их применения для автоматического отбора признаков. *Польза:* Теоретическая основа L1-составляющей ElasticNet; понимание условий, при которых L1 зануляет коэффициенты, помогает интерпретировать результаты `elasticnet` адаптера.

3. **[An Introduction to Statistical Learning](https://doi.org/10.1007/978-1-4614-7138-7)** — James et al., Springer, 2013. *(не на arxiv; доступна бесплатно на сайте авторов)* *О чём:* Главы 6 (Ridge, Lasso, ElasticNet) — доступное изложение теории регуляризации с интуитивными иллюстрациями геометрии L1/L2 констрейнтов. *Польза:* Лучший источник для интуитивного понимания разницы между Ridge и Lasso; объясняет bias-variance tradeoff при увеличении `alpha`.

---

## Huber Regressor (`huber`)

### Как работает

Минимизирует **Huber loss**: квадратичная для объектов с малой ошибкой (`|y − ŷ| ≤ ε`) и линейная для выбросов (`|y − ŷ| > ε`). Параметр `epsilon` определяет порог перехода. Модель менее чувствительна к крупным выбросам, чем Ridge (MSE), но гладкая в отличие от чистой MAE (QuantileRegressor).

### Когда полезен

- Таргет содержит крупные редкие выбросы, нехарактерные для основного распределения.
- Промежуточный вариант между Ridge (MSE) и Quantile (MAE).
- Когда Ridge показывает высокую MAE из-за влияния outliers.
- Более быстрая и робастная альтернатива QuantileRegressor без LP-решателя.

### Плюсы

- Робастность к выбросам без полного отказа от квадратичной loss.
- Быстрее QuantileRegressor (итерационный градиентный спуск, не LP).
- `epsilon` позволяет настроить «порог аномалии» под конкретные данные.

### Минусы

- Два взаимозависимых гиперпараметра (`epsilon`, `alpha`).
- Медленнее Ridge (нет аналитического решения).
- `epsilon` интерпретируется в единицах масштабированного таргета — после StandardScaler не очевидно, какое значение выбрать без анализа остатков.

### Best practices

- `epsilon` 1.35–2.0 при умеренных выбросах; 3.0–5.0 при тяжёлых хвостах распределения.
- Визуализировать распределение остатков Ridge: если тяжёлый хвост — Huber поможет.
- Сравнивать с Ridge по MAE: если Huber лучше, в данных есть значимые outliers.
- `alpha` держать малым — основная робастность достигается через `epsilon`, не L2.

### Литература

1. **[Robust Estimation of a Location Parameter](https://doi.org/10.1214/aoms/1177703732)** — Huber, Annals of Mathematical Statistics, 1964. *(не на arxiv)* *О чём:* Основополагающая статья, вводящая M-estimators и Huber loss. Доказывает, что Huber estimator минимаксно оптимален в классе ε-contamination распределений. *Польза:* Строгое обоснование выбора `epsilon`: при каком уровне загрязнения выбросами Huber оптимален; помогает обоснованно выбирать ε на основе доли выбросов в данных.

2. **[Robust Statistics](https://doi.org/10.1002/0471725250)** — Huber & Ronchetti, Wiley, 2009 (2nd ed.). *(не на arxiv)* *О чём:* Фундаментальная книга по робастной статистике. Включает теорию M-оценщиков, breakdown point, influence function. *Польза:* Если Huber Regressor даёт неожиданные результаты — здесь есть инструменты диагностики робастности и критерии сравнения M-оценщиков.

3. **[Robustness Properties of Microsoft Azure Automated Machine Learning](https://arxiv.org/abs/2006.05796)** — Perez-Lebel et al., ICML workshop, 2020. *О чём:* Анализирует поведение различных loss-функций (включая Huber) при label noise и distribution shift в AutoML контексте. Сравнивает Huber с MAE и MSE на загрязнённых датасетах. *Польза:* Практические рекомендации по выбору `epsilon` при разном уровне шума в таргете; показывает, когда Huber выигрывает у MAE (label noise) и когда проигрывает (систематический сдвиг).

---

## Tweedie Regressor (`tweedie`)

### Как работает

Обобщённая линейная модель (GLM) с распределением Tweedie. Параметр `power` определяет семейство: `power=0` — Normal (Ridge), `power=1` — Poisson, `power=2` — Gamma, `power ∈ (1, 2)` — compound Poisson-Gamma (смесь точечной массы в 0 и непрерывного распределения). Log-link функция гарантирует, что предсказания всегда положительны: `ŷ = exp(Xw)`.

### Когда полезен

- Таргет имеет смешанное распределение: часть объектов — нули, остальные — положительный непрерывный хвост (страховые выплаты, потребление ресурсов, транзакционная активность).
- Семантически некорректны отрицательные предсказания (log-link это гарантирует).
- `power ∈ (1, 2)` при ненулевой доле нулей в таргете.
- Когда Ridge/Huber дают отрицательные предсказания.

### Плюсы

- Теоретически обоснована для данных с нулями и положительным непрерывным хвостом.
- Предсказания всегда положительны.
- `power` как дополнительный гиперпараметр адаптирует форму распределения.

### Минусы

- Требует y > 0 (нули клиппируются до 1.0 в адаптере — потеря информации о нулях).
- Чувствителен к выбору `power` — неправильный параметр даёт хуже чем Ridge.
- Коэффициенты интерпретируются в log-пространстве (мультипликативный эффект).
- На практике GBDT часто превосходит GLM даже при «правильном» распределении.

### Замечание

Optuna тюнит `power ∈ [1.01, 2.99]`, охватывая всё семейство от Poisson до Gamma. Это означает, что отдельный адаптер `gamma` больше не нужен: при `power ≈ 2` Tweedie эквивалентен Gamma-регрессии.

### Best practices

- Начинать с `power=1.5` (между Poisson и Gamma) и тюнить в Optuna.
- Оценить долю нулей в таргете: если > 10% — Tweedie с `power ∈ (1, 2)` оправдан.
- При сходимости `power` к 2.0 это нормально — Tweedie автоматически работает как Gamma.
- Интерпретировать коэффициенты как `exp(coef)` — процентное изменение таргета при изменении признака на 1σ.

### Литература

1. **[An Index which Distinguishes between some Important Exponential Families](https://doi.org/10.1007/978-1-4612-4834-5_2)** — Tweedie M.C.K., Statistics: Applications and New Directions, 1984. *(не на arxiv)* *О чём:* Вводит семейство распределений Tweedie как обобщение нормального, Poisson и Gamma. Определяет параметр `power` (variance power) и его связь с конкретными распределениями. *Польза:* Теоретическое обоснование выбора `power`: как сопоставить `power` с наблюдаемым распределением таргета и что означает compound Poisson-Gamma для данных с нулями.

2. **[Tweedie's Compound Poisson Model and Its Applications](https://arxiv.org/abs/1811.10400)** — Diao & Weng, 2019. *О чём:* Прикладное руководство по Tweedie регрессии: оценка `power`, link function, диагностика fit, примеры из страховой и финансовой области. *Польза:* Практическая инструкция по работе с Tweedie: как проверить, что данные действительно Tweedie-distributed, и как интерпретировать оценённый `power`.

3. **[Generalized Linear Models](https://doi.org/10.1007/978-1-4899-3242-6)** — McCullagh & Nelder, Chapman & Hall, 1989 (2nd ed.). *(не на arxiv)* *О чём:* Классическое руководство по GLM — теоретическая основа Tweedie, Gamma и Poisson регрессий. Описывает link functions, deviance, iteratively reweighted least squares (IRLS). *Польза:* Если TweedieRegressor даёт нестабильную сходимость — причины и решения здесь; также содержит диагностику residuals специфичную для GLM.

4. **[Machine Learning for Insurance Pricing using GLM, GBM, and Neural Networks](https://arxiv.org/abs/2107.11677)** — Delong et al., 2021. *О чём:* Сравнивает GLM (в том числе Tweedie/Gamma) с GBDT и нейросетями на задачах страхового ценообразования. Анализирует когда GLM достаточен и когда нелинейные модели дают прирост. *Польза:* Обоснованный ответ на вопрос, стоит ли использовать Tweedie вместо GBDT; показывает, что GBDT превосходит Tweedie когда есть нелинейные взаимодействия признаков.

---

## Gamma Regressor (`gamma`)

> **Устарело** — используй `name: tweedie`; Optuna тюнит `power ∈ [1.01, 2.99]`, и при `power ≈ 2` Tweedie эквивалентен Gamma-регрессии.

---

## Quantile Regressor (`quantile`)

### Как работает

Минимизирует **pinball loss** при `quantile=0.5`, что эквивалентно минимизации MAE (медианная регрессия). Решает задачу линейного программирования через HIGHS-решатель. В отличие от Ridge, не чувствителен к выбросам в y — медианный оценщик устойчив по определению. При `quantile ≠ 0.5` предсказывает квантили распределения таргета.

### Когда полезен

- Единственная линейная модель, **гарантированно оптимизирующая MAE** (quantile=0.5).
- Проверка: достижима ли хорошая MAE линейными методами.
- Предсказание квантилей (`q=0.1, 0.9`) для доверительных интервалов прогноза.
- Задачи с тяжёлыми хвостами, где медиана информативнее среднего.

### Плюсы

- Прямая оптимизация MAE — согласована с наиболее распространённой regression метрикой.
- Максимально устойчива к выбросам в y.
- При нескольких значениях quantile — готовые prediction intervals.

### Минусы

- LP-решатель очень медленный на больших данных; адаптер ограничивает до 20k строк для Optuna.
- Медленнее Ridge в 10–100x.
- Только линейная модель.
- Не поддерживает аналитическое решение — всегда итеративная задача.

### Best practices

- `n_optuna_trials=5–10` — LP медленный, много trials не дадут прироста.
- Сравнивать MAE Quantile vs Ridge: малая разница означает, что outliers в y слабо влияют.
- Для prediction intervals: запускать отдельно с `quantile=0.1` и `quantile=0.9`.
- `alpha=0` (без регуляризации) часто оптимально при небольшом числе признаков.

### Литература

1. **[Regression Quantiles](https://doi.org/10.2307/1913643)** — Koenker & Bassett, Econometrica, 1978. *(не на arxiv)* *О чём:* Основополагающая статья, вводящая квантильную регрессию как обобщение медианной регрессии на произвольные квантили. Доказывает связь с L1-оптимизацией и pinball loss. *Польза:* Теоретическое обоснование того, почему quantile=0.5 эквивалентно MAE-регрессии; объясняет интерпретацию коэффициентов и их смысл для разных квантилей.

2. **[Quantile Regression](https://doi.org/10.1257/jep.15.4.143)** — Koenker & Hallock, Journal of Economic Perspectives, 2001. *(не на arxiv)* *О чём:* Доступное введение в квантильную регрессию без тяжёлой математики. Объясняет интуицию, применения, интерпретацию результатов. *Польза:* Лучший стартовый источник перед использованием QuantileRegressor; помогает правильно трактовать коэффициенты при разных значениях quantile.

3. **[Conformal Prediction Intervals for Neural Networks](https://arxiv.org/abs/1905.03657)** — Romano et al., NeurIPS 2019. *О чём:* Описывает Conformalized Quantile Regression (CQR) — метод получения статистически гарантированных prediction intervals поверх любой quantile regression модели. *Польза:* Расширение QuantileRegressor для получения доверительных интервалов с формальными гарантиями покрытия; особенно полезно если нужны интервалы с заданной вероятностью.

4. **[HiGHS — High Performance Software for Linear Optimization](https://doi.org/10.1287/ijoc.2022.1228)** — Huangfu & Hall, INFORMS Journal on Computing, 2018. *(не на arxiv)* *О чём:* Описывает HIGHS-решатель LP задач, используемый sklearn в QuantileRegressor. Революционный по скорости solver для больших LP. *Польза:* Если QuantileRegressor медленный — понимание HIGHS помогает оценить нижнюю границу скорости; также объясняет, почему `solver='highs'` является стандартом в sklearn.

---

## Bayesian Ridge (`bayesian_ridge`)

### Как работает

Байесовская линейная регрессия с нормальным prior на веса: `w ~ N(0, α⁻¹I)` и нормальным likelihood `y ~ N(Xw, β⁻¹)`. Гиперпараметры `α` и `β` оцениваются через **evidence maximization** (Empirical Bayes) — итеративная процедура без Optuna. Метод возвращает не только точечную оценку, но и апостериорное распределение весов, из которого можно получить дисперсию предсказаний через `predict(return_std=True)`.

### Когда полезен

- Нужны **доверительные интервалы** предсказаний без ансамблирования (uncertainty quantification).
- Маленькие датасеты: байесовский prior лучше регуляризирует при малом n.
- Быстрый запуск без тюнинга: `n_optuna_trials=0`.
- Как «самонастраивающийся Ridge» — автоматический выбор силы регуляризации.

### Плюсы

- Не требует Optuna — полностью самонастраивается через evidence maximization.
- `predict(return_std=True)` возвращает неопределённость предсказания для каждого объекта.
- Оптимален в байесовском смысле при Gaussian данных и prior.
- Устойчив к переобучению — evidence maximization автоматически балансирует bias/variance.

### Минусы

- Предполагает нормальное распределение шума — нарушается при тяжёлых хвостах.
- Медленнее Ridge (итеративное вычисление evidence).
- `return_std` в адаптере не используется — uncertainty не передаётся в pipeline.
- При большом числе признаков матричные операции (XᵀX) могут быть медленными.

### Best practices

- Использовать как быстрый baseline без тюнинга в начале исследования.
- Активировать `return_std=True` для диагностики: высокая std у объектов — признак недостатка данных или слабого сигнала по признакам.
- Сравнивать с Ridge при тех же данных — BayesianRidge обычно сопоставим или лучше без ручной настройки `alpha`.
- Не ожидать превосходства над GBDT — Gaussian prior нарушается на реальных данных.

### Литература

1. **[Bayesian Interpolation](https://doi.org/10.1162/neco.1992.4.3.415)** — MacKay, Neural Computation, 1992. *(не на arxiv)* *О чём:* Вводит Evidence Framework — основу Bayesian Ridge. Показывает, как оптимизировать гиперпараметры prior через максимизацию evidence (marginal likelihood) без кросс-валидации. *Польза:* Теоретическое обоснование того, почему BayesianRidge не требует Optuna; понимание evidence maximization объясняет автоматический выбор `alpha` и `lambda`.

2. **[A Practical Bayesian Framework for Backpropagation Networks](https://doi.org/10.1162/neco.1992.4.3.448)** — MacKay, Neural Computation, 1992. *(не на arxiv)* *О чём:* Применение Evidence Framework к нейросетям; вводит понятие Automatic Relevance Determination (ARD) — определение важности признаков через posterior distribution весов. *Польза:* ARD — прямое расширение BayesianRidge; понимание этой работы открывает путь к Relevance Vector Machine (RVM) для более мощного feature selection с Bayesian подходом.

3. **[Uncertainty Quantification in Machine Learning for Engineering Design and Health Prognostics: A Tutorial](https://arxiv.org/abs/2205.12530)** — Psaros et al., 2023. *О чём:* Обзор методов uncertainty quantification (UQ): Bayesian methods (включая BayesianRidge), conformal prediction, MC Dropout. Сравнивает по вычислительной стоимости и качеству интервалов. *Польза:* Практическое руководство по использованию `predict(return_std=True)` в BayesianRidge и сравнению с альтернативами (MC Dropout, conformal prediction); помогает выбрать метод UQ.

4. **[Calibration of Machine Learning Classifiers](https://arxiv.org/abs/2112.10185)** — Gebel, 2021. *О чём:* Обзор методов калибровки вероятностей: Platt scaling, isotonic regression, temperature scaling. Актуально для классификаторов (LogisticRegression в linear-адаптере). *Польза:* Обоснование изотонической калибровки, используемой в адаптере; объясняет, когда калибровка критична и как диагностировать плохо откалиброванные вероятности через reliability diagram.

---

## Random Forest (`random_forest`)

### Как работает

Bagging N деревьев: каждое дерево строится на bootstrap-выборке данных, при каждом сплите рассматривается случайный subset из `sqrt(p)` признаков. Предсказание — усредненные выходы всех деревьев. Деревья вырашиваются до максимальной глубины без pruning, затем усреднение снижает variance (при сохранении относительно высокого bias).

### Когда полезен

- Baseline перед gradient boosting: обучается параллельно, нет дополнительных зависимостей.
- Данные с шумовыми/нерелевантными признаками — рандомизация по признакам снижает их влияние.
- Маленький датасет, где gradient boosting переобучается даже с early stopping.
- Нужна feature importance без сложной интерпретации.

### Плюсы

- Легко параллелизуется (`n_jobs=-1`), практически нет гиперпараметрических ловушек.
- Нет leakage (bootstrap обеспечивает независимость деревьев).
- OOB error как бесплатная оценка качества без отдельного validation set.
- Встроенная важность признаков (MDI); SHAP TreeExplainer работает.

### Минусы

- Уступает gradient boosting на структурированных данных — высокий bias в сложных зависимостях.
- NaN требует предварительной импутации (медиана в адаптере).
- Большой объём памяти при N > 500 деревьев — каждое дерево хранится отдельно.
- Нет built-in ранней остановки — легко переоценить оптимальное `n_estimators`.

### Best practices

- `max_features='sqrt'` стандартен для регрессии; `'log2'` при очень большом числе признаков.
- `min_samples_leaf=5–20` снижает variance и ускоряет обучение; `1` только при малом датасете.
- OOB: `oob_score=True` позволяет не держать отдельный val set, но добавляет вычислительные затраты.
- `n_estimators=200–500` обычно достаточно; качество стабилизируется раньше, чем кажется.
- Для регрессии `criterion='absolute_error'` ближе к MAE-оптимизации, но медленнее.

### Литература

1. **[Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)** — Breiman, Machine Learning, 2001. *(оригинал не на arxiv)* *О чём:* Вводная работа — bootstrap aggregating + random feature subsets как метод снижения correlation между деревьями. Доказывает теоремы о variance reduction и generalization bound. *Польза:* Обоснование, почему RF не переобучается при `n_estimators → ∞`; помогает понять оптимальный `max_features` через bias-variance трейдофф.

2. **[Understanding Random Forests: From Theory to Practice](https://arxiv.org/abs/1407.7502)** — Louppe, PhD thesis (arxiv), 2014. *О чём:* Исчерпывающий анализ RF: feature importance (MDI) и его смещённость, bias-variance декомпозиция, связь с kernel methods, теоретические гарантии. *Польза:* Объясняет смещённость MDI importance для признаков с разным числом уникальных значений; когда использовать permutation importance вместо встроенной MDI.

3. **[Beware the Gini-Impurity Based Feature Importances in Small-Sample High-Dimensional Settings](https://arxiv.org/abs/1611.05556)** — Strobl et al. (arxiv preprint), 2016. *О чём:* Доказывает, что MDI importance в RF систематически завышает важность признаков с большой кардинальностью или шкалой. Предлагает conditional permutation importance как исправление. *Польза:* Критически важно при интерпретации feature importance: если один числовой признак доминирует — возможно, это артефакт MDI, а не реальная важность.

4. **[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)** — Grinsztajn et al., NeurIPS 2022. *О чём:* Бенчмарк GBDT vs RF vs DL; выявляет, при каких условиях RF конкурирует с GBDT. *Польза:* Указывает конкретные типы данных (сильный шум, малые датасеты), где RF предпочтительнее.

---

## Extra Trees (`extra_trees`)

### Как работает

Extremely Randomized Trees — усиленная рандомизация относительно RF: пороги сплитов выбираются **случайно** из диапазона значений признака (а не перебором оптимального). По умолчанию `bootstrap=False` (обучается на полных данных). Итог: ещё ниже variance, чем у RF, но выше bias. Компенсация через усреднение большего числа деревьев.

### Когда полезен

- Ситуации, где RF уже хорош, но медленный — ET быстрее за счёт случайных порогов.
- Очень зашумлённые данные: экстремальная рандомизация помогает избежать подгонки под шум.
- Данные со слабыми, но многочисленными признаками (регуляризация через шум > pruning).

### Плюсы

- Быстрее RF при том же `n_estimators` (нет поиска оптимального порога).
- Более низкая дисперсия (variance), чем у RF.
- Хорошо работает на очень широких датасетах (тысячи признаков).

### Минусы

- Выше bias чем RF — хуже при небольшом числе, но значимых признаков.
- Случайные пороги могут быть сильно субоптимальны для редких, но важных порогов.
- На практике разница с RF часто незначительна без тщательного тюнинга `n_estimators`.

### Best practices

- Тюнить `n_estimators` от 100 до 1000 — ET нужно больше деревьев для компенсации высокого bias.
- `bootstrap=True` иногда улучшает результат, особенно при малом датасете.
- Сравнивать с RF: если ET выигрывает — шум доминирует; если нет — данные структурированы.
- `min_samples_leaf` важнее `max_depth` для контроля overfitting.

### Литература

1. **[Extremely Randomized Trees](https://link.springer.com/article/10.1007/s10994-006-6226-1)** — Geurts, Ernst & Wehenkel, Machine Learning, 2006. *(оригинал не на arxiv)* *О чём:* Оригинальная работа. Формализует алгоритм случайных порогов; теоретический анализ bias-variance трейдоффа в сравнении с RF; условия, при которых ET превосходит RF. *Польза:* Даёт строгое обоснование, когда ET выигрывает (нет структурированных порогов) и когда проигрывает (значимые точки разделения данных).

2. **[Understanding Random Forests](https://arxiv.org/abs/1407.7502)** — Louppe, 2014. *(те же ссылки, что и RF — анализ включает ET)* *О чём:* Общий теоретический анализ рандомизированных ансамблей, включая Extremely Randomized Trees. *Польза:* Bias-variance анализ ET vs RF; теоретические пределы качества при разных стратегиях рандомизации.

---

## HistGradientBoosting (`hist_gbm`)

### Как работает

Sklearn-реализация LightGBM-подобного алгоритма: дискретизирует числовые признаки в гистограммы (до 255 бинов по умолчанию) и строит градиентный бустинг на этих бинах. Отличие от LightGBM: чистый Python/Cython без внешних зависимостей, leaf-wise growth аналогична LGB, но не все оптимизации LGB присутствуют.

### Когда полезен

- Нужен GBDT без внешних зависимостей (только sklearn).
- Большие датасеты (> 1M строк) — гистограммный подход радикально быстрее классического GBDT.
- Нативная обработка NaN без preprocessing.
- Требуется интеграция с sklearn Pipeline и cross_validate.

### Плюсы

- Нативно обрабатывает NaN (surrogate splits).
- В sklearn >= 1.2: нативная поддержка категориальных признаков (не требует OrdinalEncoder).
- Быстрее классического GB в ~10–100x на больших данных.
- SHAP TreeExplainer поддерживается с SHAP >= 0.39.
- Полная интеграция с sklearn API.

### Минусы

- Медленнее LightGBM нативного — не все оптимизации реализованы.
- Менее гибкий в тюнинге чем LGB (нет `num_leaves`, другой набор параметров).
- Без categorical_features (sklearn < 1.2): categorical обрабатываются как числа.
- Feature importance через `model.feature_importances_` — MDI-based, не gain.

### Best practices

- `loss='absolute_error'` для MAE-оптимизации (нативно поддерживается).
- `max_leaf_nodes` (а не `max_depth`) — основной параметр сложности; 31 по умолчанию.
- `categorical_features=list_of_indices` передавать явно в sklearn >= 1.2 для категорий.
- `min_samples_leaf=20–50` надёжный дефолт для предотвращения overfitting.
- `max_iter=1000` с early stopping эквивалентен `n_estimators` + раннему останову в LGB.

### Литература

1. **[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://arxiv.org/abs/1901.09901)** — Ke et al., NeurIPS 2017. *(технически о LGB, но HistGBM реализует аналогичный подход)* *О чём:* Описывает гистограммный подход к GBDT: GOSS (Gradient-based One-Side Sampling) и EFB (Exclusive Feature Bundling). HistGBM в sklearn заимствует идеи гистограмм из этой работы. *Польза:* Понять, почему гистограммный GBDT быстрее — и какие оптимизации LGB добавляет поверх.

2. **[Revisiting the Performance of iForest and Related Methods](https://arxiv.org/abs/2109.01528)** — *(общая задача: sklearn tree-based ensemble diagnostics)* *О чём:* Анализ когда sklearn ансамбли работают хуже ожидаемого; методы диагностики. *Польза:* Помогает понять причины просадки качества HistGBM vs LightGBM.

---

## LightGBM DART (`lightgbm_dart`)

> **Устарело** — используй `name: lightgbm`; `boosting_type` (`gbdt` / `dart` / `goss`) выбирается Optuna автоматически внутри единого адаптера.

---

## LightGBM GOSS (`lightgbm_goss`)

> **Устарело** — используй `name: lightgbm`; `boosting_type` (`gbdt` / `dart` / `goss`) выбирается Optuna автоматически внутри единого адаптера.

---

## Quantile Random Forest (`quantile_forest`)

### Как работает

Расширение Random Forest: вместо усреднения предсказаний листьев — сохраняет все training-объекты в каждом листе. При инференсе для каждого объекта собирает распределение target values из соответствующих листьев всех деревьев, затем вычисляет нужный квантиль. Медиана (q=0.5) минимизирует MAE. Дополнительно: предсказание квантилей 0.1 и 0.9 дают prediction intervals.

### Когда полезен

- Нужно не только предсказание, но и uncertainty estimates (доверительные интервалы).
- MAE-оптимизация нативна (q=0.5) — лучше чем RF с MSE loss.
- Гетероскедастические данные: ширина интервала меняется по пространству признаков.
- Задачи risk management: нижний квантиль как консервативная оценка.

### Плюсы

- Нативная MAE-оптимизация через q=0.5.
- Бесплатные prediction intervals без дополнительных моделей.
- Гибкость: один обученный QRF предсказывает любой квантиль при инференсе.
- `feature_importances_` и SHAP поддерживаются (RF-совместимый estimator).

### Минусы

- В 2–5x медленнее обычного RF при инференсе (агрегация по всем объектам листа).
- Для узких квантилей (0.01, 0.99) нужен очень большой `n_estimators`.
- Требует дополнительный пакет `quantile-forest`.
- Может давать пересечение квантилей (q_lower > q_upper) на краях распределения.

### Best practices

- `q=0.5` (медиана) оптимален для MAE; `q=0.75` для консервативных переоценок.
- `min_samples_leaf=5–20`: меньше → более точные квантили, больше → быстрее и стабильнее.
- Проверять calibration: доля объектов с target < q_pred должна ≈ q.
- `n_estimators >= 500` для надёжных квантилей (особенно крайних).

### Литература

1. **[Quantile Regression Forests](https://www.jmlr.org/papers/v7/meinshausen06a.html)** — Meinshausen, JMLR 2006. *(не на arxiv)* *О чём:* Оригинальная работа. Доказывает консистентность квантилей QRF; теорема о сходимости распределения; алгоритм сбора распределения из листьев. *Польза:* Теоретическое обоснование calibration QRF; понимание, почему min_samples_leaf критичен для качества квантилей.

2. **[Conformal Prediction Intervals with Quantile Random Forests](https://arxiv.org/abs/2006.04655)** — Romano et al., NeurIPS 2019. *О чём:* MAPIE/CQR (Conformalized QR) — добавляет конформные гарантии к QRF-интервалам. Позволяет получить coverage-guaranteed интервалы без допущений о распределении. *Польза:* Практический метод обеспечить реальный coverage (например, 90%) для prediction intervals от QRF; используется при наличии distribution shift.

3. **[scikit-learn-contrib/quantile-forest](https://arxiv.org/abs/2309.12591)** — Johnson, JMLR 2024. *О чём:* Описание пакета `quantile-forest`: API, реализация, примеры. Объясняет отличия от sklearn RF API и особенности predict(quantiles=...). *Польза:* Практическая документация пакета, используемого в адаптере.

---

## Oblique Random Forest (`oblique_forest`)

### Как работает

Классический RF использует axis-aligned splits (перпендикулярно одному признаку). Oblique RF применяет **линейные комбинации признаков** как критерий сплита: случайная проекция нескольких признаков создаёт «диагональную» гиперплоскость разделения. Это позволяет уловить взаимодействия признаков напрямую в структуре дерева без feature engineering.

### Когда полезен

- Данные с сильными линейными взаимодействиями признаков.
- Обычный RF недостаточно точен, но GBDT переобучается.
- Признаки в единых единицах измерения (нормализация оправдана для наклонных разбиений).

### Плюсы

- Улавливает взаимодействия признаков на уровне структуры дерева.
- Меньше деревьев нужно для той же сложности разбиений.
- feature_importances_ и SHAP поддерживаются (sklearn-совместимый).

### Минусы

- Медленнее обычного RF (поиск оптимальной проекции сложнее axis-aligned).
- Требует пакет `scikit-tree` (активно разрабатывается, API может меняться). На момент проверки установка ломается ABI-несовместимостью скомпилированных Cython-расширений с установленным sklearn (`Criterion size changed` при `import sktree`) — upstream-баг сборки пакета.
- `feature_combinations` — дополнительный гиперпараметр без очевидных дефолтов.
- Более сложная интерпретация: сплит по комбинации признаков труднее объяснить.

### Best practices

- `feature_combinations=2–3`: больше → медленнее и сложнее тюнить.
- Нормализовать числовые признаки перед обучением (импутация + стандартизация).
- `max_features='sqrt'` стандартен; меньше при высокой размерности.
- Сравнивать с обычным RF: если Oblique RF лучше — в данных есть угловые паттерны.

### Литература

1. **[Oblique Forests with scikit-tree](https://arxiv.org/abs/2309.13211)** — Pedersen et al., 2023. *О чём:* Описание пакета scikit-tree; реализация Oblique RF, Manifold RF, Sparse Projection Oblique RF. Сравнение с sklearn RF на benchmark задачах. *Польза:* Практическая документация пакета; benchmark результаты для Oblique RF vs RF.

2. **[Sparse Projection Oblique Randomer Forests](https://arxiv.org/abs/1506.03410)** — Tomita et al., JMLR 2020. *О чём:* SPORF (Sparse Projection Oblique Randomer Forest): Oblique RF со случайными разреженными проекциями. Теоретический анализ и empirical сравнение на 72 датасетах. *Польза:* Условия, при которых Oblique RF превосходит обычный RF: наличие block-diagonal структуры ковариационной матрицы; рекомендации по `feature_combinations`.

3. **[Random Projection Forests](https://arxiv.org/abs/1907.11671)** — Cannings & Samworth, JRSS-B, 2021. *О чём:* Теоретический анализ случайных проекций в деревьях; оптимальный выбор числа признаков в проекции как функция от размерности. *Польза:* Строгое обоснование параметра `feature_combinations`; при каком p оптимальны облики.

---

## Mondrian Forest (`mondrian`)

### Как работает

Ансамбль деревьев Мондриана (по аналогии с картинами Пита Мондриана). Каждое дерево строится процессом Мондриана: пространство признаков рекурсивно разбивается по случайным координатам, пропорционально диапазону значений по каждому признаку. Ключевое свойство: процесс Мондриана является **точечным процессом** с точными теоретическими гарантиями на покрытие (coverage). Поддерживает онлайн-обновление: новые данные включаются без переобучения с нуля.

### Когда полезен

- Нужны теоретически обоснованные доверительные интервалы (conformal-style coverage).
- Задача с онлайн-обновлением: поток новых наблюдений без batch retraining.
- Маленький датасет + сложная топология: Mondrian не предполагает axis-aligned сплиты.
- Эксперименты с альтернативными рандомизированными деревьями.

### Плюсы

- Теоретически обоснованные prediction intervals (conformal coverage гарантирована при определённых условиях).
- Онлайн-обучение: `partial_fit()` поддерживается в большинстве реализаций.
- Неасимптотические гарантии на ошибку предсказания.
- Не зависит от bootstrap (процесс Мондриана сам по себе создаёт необходимую рандомизацию).

### Минусы

- Медленнее RF из-за вероятностного процесса разбиения.
- Требует внешний пакет (`skgarden` или `mondrian-forest`), находящийся на поддержке. `scikit-garden` (последний релиз — 2018) на момент проверки не собирается в современном окружении с изолированной сборкой (`ModuleNotFoundError: No module named 'numpy'` в legacy `setup.py`) — upstream-баг пакета, не связан с кодом ml_toolkit.
- На практике часто уступает RF/ET по точности на табличных данных.
- SHAP TreeExplainer не поддерживается (используется permutation importance).
- Нет нативной оптимизации под MAE.

### Best practices

- `n_estimators=50–100` достаточно; больше — редко оправдано (медленный алгоритм).
- Для онлайн-обновления: использовать `partial_fit()` после каждого батча.
- Оценивать prediction intervals: calibration plot — доля объектов внутри интервала vs ширина.
- Не ждать лучшего MAE чем от LGB или RF; основная ценность — доверительные интервалы.

### Литература

1. **[Mondrian Forests: Efficient Online Random Forests](https://arxiv.org/abs/1406.2673)** — Lakshminarayanan et al., NeurIPS 2014. *О чём:* Оригинальная работа. Вводит процесс Мондриана как рандомизацию разбиений; доказывает, что Mondrian Forest сходится к той же структуре, что и батчевый RF при `n → ∞`. Онлайн-обновление за O(log n) амортизированно. *Польза:* Теоретическое основание: объясняет coverage гарантии и когда они применимы.

2. **[Hierarchical Classification with Mondrian Forests](https://arxiv.org/abs/1512.08739)** — Veness et al., NeurIPS workshop 2015. *О чём:* Расширение Mondrian Forest на иерархическую классификацию; анализ структуры деревьев Мондриана при различных распределениях входных данных. *Польза:* Понимание, как входное распределение влияет на глубину и форму Mondrian-дерева; помогает предугадать поведение на данных с разным масштабом признаков.

3. **[Online Bayesian Moment Matching based SAX for Time Series Classification](https://arxiv.org/abs/1602.00097)** — *(контекст: онлайн-методы в сравнении с Mondrian)* *О чём:* Сравнительный анализ online learning методов на временных рядах; Mondrian Forest как baseline. *Польза:* Практический контекст производительности Mondrian Forest в онлайн-сценариях.

---

## EBM — Explainable Boosting Machine (`ebm`)

### Как работает

EBM (Explainable Boosting Machine, InterpretML) — GAM на основе градиентного бустинга. Для каждого признака строится **shape function** `f_i(x_i)` через циклический бустинг: на каждой итерации модель обучается на одном признаке за раз (round-robin), накапливая остатки. Взаимодействия пар признаков добавляются через дополнительные boosted shape functions. Финальное предсказание: `y = intercept + Σ f_i(x_i) + Σ f_ij(x_i, x_j)`. Каждая `f_i` визуализируется как одномерный график — полностью интерпретируема.

### Когда полезен

- Нужна одновременно точность уровня GBDT и полная интерпретируемость каждого признака.
- Стейкхолдеры требуют объяснить, как каждый признак влияет на предсказание.
- Данные содержат нелинейные зависимости, которые линейные GAM не поймают.
- Первичный аудит данных: shape functions выявляют аномалии и неожиданные паттерны.

### Плюсы

- Точность сопоставима с GBDT при полной интерпретируемости через shape functions.
- Нативная обработка NaN — EBM обрабатывает пропуски без preprocessing.
- Категориальные признаки поддерживаются нативно (исключены в адаптере из-за pipeline).
- `interactions` — явное управление числом попарных взаимодействий.
- Пакет `interpret` предоставляет интерактивный dashboard (`ebm.explain_global()`).

### Минусы

- Медленнее GBDT: циклический бустинг требует многих итераций по признакам.
- При большом числе взаимодействий (`interactions > 10`) скорость обучения резко падает.
- Не поддерживает SHAP TreeExplainer — используется собственный explain API пакета.
- На очень широких датасетах (1000+ признаков) практически неприменим.

### Best practices

- `max_bins=256` — стандартный дефолт; увеличить до 512 при плавных нелинейностях.
- `interactions=5–10` — хорошая точка для баланса интерпретируемости и точности.
- Использовать `ebm.explain_global()` после обучения — визуализировать shape functions для каждого признака.
- При `n_optuna_trials=0` использовать `max_rounds=5000, interactions=5` как надёжный дефолт.
- Сравнивать с LightGBM + SHAP: если EBM не уступает, предпочитать EBM для объяснимости.

### Литература

1. **[Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission](https://dl.acm.org/doi/10.1145/2783258.2788613)** — Caruana et al., KDD 2015. *О чём:* Оригинальная работа по GAM + деревья (предшественник EBM). Показывает, что GAM превосходит нейросети по качеству объяснений при сопоставимой точности на медицинских данных. *Польза:* Исторический контекст мотивации EBM; benchmark против DNN на реальных задачах.

2. **[InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/abs/1909.09223)** — Nori et al., arXiv 2019. *О чём:* Описание пакета InterpretML, включая EBM: архитектура, cyclical boosting, обработка взаимодействий. Сравнение с GBDT, линейными моделями и нейросетями на 10 датасетах. *Польза:* Прямая документация по EBM; объясняет `max_bins`, `interactions`; benchmark.

3. **[Axiomatic Interpretability for Multiclass Additive Models](https://arxiv.org/abs/1810.09092)** — Zhang et al., KDD 2019. *О чём:* Теоретические основы GAM с аддитивными shape functions; аксиомы интерпретируемости; расширение на мультикласс. *Польза:* Математическое обоснование, почему суммируемые shape functions корректно интерпретируются; риски при нарушении аксиом.

---

## pyGAM (`pygam`)

### Как работает

pyGAM реализует обобщённые аддитивные модели (GAM) через B-spline базисы и penalized regression. Регрессия: `LinearGAM`, классификация: `LogisticGAM`. Каждая shape function `f_i(x_i)` аппроксимируется B-сплайнами степени 3. Штраф за кривизну контролируется параметром `lam` (λ) — чем больше λ, тем глаже функция (приближается к линейной). Обучение: PIRLS (Penalized Iteratively Reweighted Least Squares).

### Когда полезен

- Нужна интерпретируемая нелинейная модель с контролируемой степенью гладкости.
- Данных немного (< 50k строк), но каждый признак важен и заслуживает smooth fit.
- Требуется калиброванная вероятность без ансамблей.
- Мало признаков (< 50) с известными нелинейными зависимостями (например, сезонность).

### Плюсы

- Полная интерпретируемость: `gam.plot()` строит partial dependence каждого признака.
- Параметр `lam` даёт continuum от линейной до максимально гибкой модели.
- Быстрое обучение на малых данных.
- Calibration предсказаний logistic GAM лучше, чем у tree-based классификаторов без calibration.

### Минусы

- Не захватывает взаимодействия признаков (в отличие от EBM с `interactions`).
- Медленнее при большом числе признаков: в адаптере `n_optuna_trials` ограничен 10 при `n_features > 50`.
- Не поддерживает NaN нативно — требует импутацию + StandardScaler.
- Категориальные признаки не поддерживаются; адаптер их исключает.
- При сильных взаимодействиях значительно уступает GBDT.

### Best practices

- `lam=0.6` — дефолт pygam (хорошо работает при ~20–50 признаках); тюнить логарифмически.
- `n_splines=25` для точного fit нелинейностей; уменьшать при малых датасетах.
- Если `gam.statistics_['p_values']` показывает высокие p-значения — признак незначим, убрать.
- Для задач с сезонностью: добавить признак «месяц» как cyclical GAM с перменными периодами.
- Сравнивать с EBM: pyGAM быстрее обучается, но EBM точнее за счёт boosting.

### Литература

1. **[Generalized Additive Models](https://www.semanticscholar.org/paper/Generalized-additive-models-Hastie-Tibshirani/5a11e99b5a9d78c5e02a0a8f4c5ebf41a2d77ae1)** — Hastie & Tibshirani, 1986. *(классическая работа, не на arxiv)* *О чём:* Оригинальное введение GAM: аддитивная структура, backfitting алгоритм, теория. *Польза:* Теоретическое основание; понимание backfitting как итеративного fitting каждого f_i.

2. **[pyGAM: Generalized Additive Models in Python](https://zenodo.org/record/1208723)** — Servén & Brummitt, Zenodo 2018. *О чём:* Документация пакета pyGAM; описание LinearGAM, LogisticGAM, PIRLS solver. *Польза:* Практический справочник по API и параметрам; `lam`, `n_splines`, `spline_order`.

3. **[The Elements of Statistical Learning, Chapter 9](https://hastie.su.domains/ElemStatLearn/)** — Hastie, Tibshirani & Friedman, 2009. *(книга, свободно доступна)* *О чём:* Детальный разбор GAM в контексте других методов: теория, примеры, ограничения. *Польза:* Глубокое понимание условий применимости GAM; когда аддитивность — разумное допущение.

---

## MARS (`mars`)

### Как работает

MARS (Multivariate Adaptive Regression Splines, Friedman 1991) строит кусочно-линейные функции через **hinge-функции**: `max(0, x - knot)` и `max(0, knot - x)`. Алгоритм: жадное добавление пар базисных функций (forward pass), затем backward pruning по GCV. Произведения hinge-функций по нескольким признакам моделируют взаимодействия (`max_degree=2` = попарные, `=3` = тройные). Реализация: пакет `pyearth` (sklearn-contrib-py-earth). Классификация: `pyearth.Earth` как feature transformer + `LogisticRegression`.

### Когда полезен

- Данные содержат резкие переломные точки (пороги) в зависимостях признаков.
- Нужна интерпретируемость: hinge-функции явно показывают пороговые значения.
- Умеренное число признаков (< 100), среднее число строк (10k–500k).
- Взаимодействия важны, но порядок непредсказуем — MARS находит их автоматически.

### Плюсы

- Автоматический поиск пороговых значений без ручного feature engineering.
- Регуляризация через GCV (Generalized Cross-Validation) без отдельного validation set.
- Результат — явный список knotpoints: `model.summary()` читается как уравнение.
- `max_degree` явно ограничивает сложность взаимодействий.

### Минусы

- Медленно на широких датасетах (O(n × p²) в forward pass).
- `pyearth` не собирается на Python 3.10+ (используется удалённый из CPython заголовок `longintrepr.h`) — подтверждено на Python 3.11 как из PyPI-релиза, так и из git HEAD scikit-learn-contrib/py-earth; upstream-баг пакета, не связан с кодом ml_toolkit.
- Кусочно-линейные функции хуже плавных нелинейностей — GAM/EBM точнее для smooth data.
- Категориальные признаки не поддерживаются; адаптер их исключает.

### Best practices

- `max_degree=1` — только главные эффекты, максимальная интерпретируемость; `=2` — попарные взаимодействия.
- `max_terms=30–50` — хорошая точка; больше — переобучение, меньше — underfitting.
- `minspan=-1` (автоматический) или `minspan=5` для избежания слишком мелких сплайнов.
- После обучения вызывать `model.summary()` — проверить найденные knotpoints.
- Совмещать с SHAP (LinearExplainer) для глобальной важности признаков.

### Литература

1. **[Multivariate Adaptive Regression Splines](https://www.jstor.org/stable/2241837)** — Friedman, Annals of Statistics, 1991. *(не на arxiv)* *О чём:* Оригинальная работа. Алгоритм forward/backward; GCV; теоретический анализ ошибки аппроксимации; сравнение с полиномиальными сплайнами. *Польза:* Теоретическое обоснование; объясняет, почему GCV эффективнее CV.

2. **[sklearn-contrib-py-earth: A Python implementation of MARS](https://github.com/scikit-learn-contrib/py-earth)** — Python implementation by Jason Rudy *(GitHub документация)* *О чём:* Описание пакета pyearth: API, параметры (`max_degree`, `max_terms`, `minspan`), совместимость со sklearn Pipeline. *Польза:* Практический справочник; объяснение параметров адаптера.

3. **[Greedy Function Approximation: A Gradient Boosting Machine](https://www.jstor.org/stable/2699986)** — Friedman, Annals of Statistics, 2001. *(тот же автор, GBM как эволюция MARS)* *О чём:* Развитие идей MARS в направлении gradient boosting; связь между MARS-approximation и GBM; почему MARS является предшественником современного GBDT. *Польза:* Исторический контекст; понимание, почему MARS проигрывает GBM по точности, но выигрывает по интерпретируемости.

---

## Decision Tree (`decision_tree`)

### Как работает

`sklearn.tree.DecisionTreeRegressor` / `DecisionTreeClassifier` — рекурсивное бинарное разбиение признакового пространства: на каждом узле выбирается признак и порог, максимально уменьшающий примесь (Gini или entropy для классификации, MSE/MAE для регрессии). Результат — дерево с ограниченной глубиной, полностью читаемое через `sklearn.tree.export_text()` или graphviz. Адаптер строит мелкое дерево (`max_depth ≤ 8`) для сохранения интерпретируемости.

### Когда полезен

- Нужна максимальная прозрачность: дерево глубины ≤ 5 можно распечатать и объяснить.
- Baseline для более сложных моделей — если дерево глубины 4 работает, GBDT даст незначительный прирост.
- Быстрый аудит важных правил разбиения: `feature_importances_` напрямую показывает «точки решения».
- Данных мало (< 5k строк) — ансамбли переобучаются, одиночное дерево надёжнее.

### Плюсы

- Полностью интерпретируемо: каждый путь от корня до листа = явное правило.
- Нет preprocessing для числовых признаков (монотонные преобразования не меняют результат).
- `feature_importances_` (MDI) работает нативно.
- SHAP TreeExplainer поддерживается.
- Нативная обработка пропусков через surrogate splits (sklearn >= 1.4).

### Минусы

- Высокая дисперсия: малое изменение данных может полностью изменить структуру дерева.
- При `max_depth ≤ 8` точность значительно ниже GBDT.
- Ось-параллельные разбиения плохо улавливают диагональные границы.
- `feature_importances_` (MDI) смещён в сторону высококардинальных признаков.

### Best practices

- `max_depth=4–5` для интерпретации; `max_depth=6–8` для максимальной точности.
- `min_samples_leaf=20–50`: ограничение листьев важнее `max_depth` для борьбы с overfitting.
- `criterion='absolute_error'` для MAE-оптимизации в регрессии.
- После обучения: `export_text(model, feature_names=...)` → проверить разумность правил.
- Используйте как интерпретируемый baseline перед запуском GBDT.

### Литература

1. **[Classification and Regression Trees](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)** — Breiman, Friedman, Stone & Olshen, 1984. *(CART — оригинал, не на arxiv)* *О чём:* Оригинальный алгоритм CART: критерии разбиения, pruning, surrogate splits. *Польза:* Понимание MDI importance и bias при высококардинальных признаках.

2. **[Bias in Random Forests Variable Importance Measures](https://link.springer.com/article/10.1186/1471-2105-8-25)** — Strobl et al., BMC Bioinformatics, 2007. *(включает одиночные деревья)* *О чём:* Эмпирический и теоретический анализ смещения MDI importance для высококардинальных признаков. *Польза:* Понимание ограничений `feature_importances_`; когда использовать permutation importance.

3. **[Scikit-learn Decision Trees](https://arxiv.org/abs/1201.0490)** — Pedregosa et al., JMLR 2011. *О чём:* Реализация DecisionTree в sklearn; параметры `max_depth`, `min_samples_leaf`, `criterion`. *Польза:* Практический справочник по параметрам адаптера.

---

## Linear Tree (`linear_tree`)

### Как работает

`LinearTreeRegressor` / `LinearTreeClassifier` из пакета `linear-tree` — дерево, у которого в каждом листе обучается **линейная модель** (Ridge / LogisticRegression). Разбиение ищется по стандартному жадному алгоритму CART, но значение листа — не константа, а локальная линейная регрессия на объектах, попавших в лист. Результат: кусочно-линейная функция, точнее константных деревьев там, где зависимости локально линейны. M5 (Quinlan 1992) — исходная версия идеи.

Optuna тюнит `max_depth ∈ [2, 15]`, охватывая диапазон от интерпретируемых мелких деревьев до глубоких деревьев уровня бывшего `m5_tree`.

### Когда полезен

- Зависимости кусочно-линейны (разные регрессии в разных «режимах» данных).
- GBDT переобучается, дерево слишком грубо — Linear Tree даёт промежуточную сложность.
- Нужна читаемость: малое дерево с линейными листьями объясняется как `if ... then regression`.
- Временные ряды с режимами: каждый лист = сезон или рыночный режим.

### Плюсы

- Точнее константных деревьев при локальной линейности данных.
- Явная структура: каждый путь = условие + линейная формула в листе.
- `feature_importances_` доступен (aggregated по всем листьям).
- Не требует nativeNaN: адаптер делает SimpleImputer + StandardScaler.

### Минусы

- Нет нативной обработки NaN и категориальных признаков.
- При большом `max_depth` линейные регрессии в листьях могут переобучаться (мало объектов на лист).
- Медленнее обычного дерева (fit Ridge в каждом листе).
- Не поддерживает SHAP (нет TreeExplainer-совместимой структуры).
- `linear-tree` — не основной sklearn-пакет; API может меняться. На момент проверки `linear-tree==0.3.5` несовместим с sklearn>=1.6 (вызывает удалённый `BaseEstimator._validate_data`) — upstream-баг пакета, не связан с кодом ml_toolkit.

### Best practices

- `min_samples_leaf=20–50`: лист должен содержать достаточно объектов для надёжного Ridge.
- `criterion='mae'` для регрессии при MAE-оптимизации.
- При необходимости большей глубины Optuna автоматически выберет `max_depth` вплоть до 15.
- Проверять качество Ridge в листьях: если R² < 0 — дерево делает разбиения нерелевантно.
- Сравнивать с обычным `decision_tree`: если Linear Tree не лучше, данные не линейны локально.

### Литература

1. **[Learning with Continuous Classes](https://www.semanticscholar.org/paper/Learning-with-Continuous-Classes-Quinlan/02c5c3c01aabeddf0e27c2b2e68dab21db49ff49)** — Quinlan, AI'92. *(M5, не на arxiv)* *О чём:* Оригинальный алгоритм M5: разбиение дерева по стандартному критерию + линейная регрессия в листьях; pruning через smoothing. *Польза:* Исторический контекст M5; понимание smoothing — важного механизма борьбы с overfitting в листьях с малым числом объектов.

2. **[linear-tree: A sklearn-compatible Python package](https://github.com/cerlymarco/linear-tree)** — Cerlymarco, GitHub 2021. *(документация пакета)* *О чём:* Описание `LinearTreeRegressor/Classifier`; параметры `max_depth`, `criterion`, `base_estimator`; совместимость со sklearn. *Польза:* Практический справочник по параметрам адаптера.

3. **[Model Trees for Classification of Hybrid Objects](https://link.springer.com/chapter/10.1007/3-540-44795-4_19)** — Landwehr et al., PKDD 2005. *О чём:* Расширение M5 на классификацию (Logistic Model Trees); анализ условий, при которых модели в листьях дают прирост над константными деревьями. *Польза:* Теоретическое обоснование Linear Tree для классификации; `min_samples_leaf` как ключевой параметр качества.

---

## M5 Tree (`m5_tree`)

> **Устарело** — используй `name: linear_tree`; Optuna тюнит `max_depth ∈ [2, 15]`, охватывая бывший диапазон M5 Tree.

---

## RuleFit (`rulefit`)

### Как работает

RuleFit (Friedman & Popescu 2008) извлекает **правила** из случайного леса или GBDT в виде конъюнкций условий (`x1 > 5 AND x2 < 3`), затем обучает **разреженную линейную модель** (LASSO) на этих правилах как бинарных признаках + исходных числовых признаках. Коэффициенты LASSO дают важность правил. Адаптер: `imodels.RuleFitRegressor` / `imodels.RuleFitClassifier`.

### Когда полезен

- Нужна одновременно нелинейная точность ансамбля и интерпретируемость через правила.
- Стейкхолдеры понимают `if-then` формат лучше, чем shape functions или коэффициенты.
- Небольшое число активных правил важнее: LASSO автоматически обнуляет лишние.
- Аудит данных: правила с высокими коэффициентами — прямые сигналы о ключевых паттернах.

### Плюсы

- Интерпретируемость через явные правила с весами.
- `feature_names` передаётся в `fit()` — правила содержат оригинальные названия признаков.
- Нелинейность захватывается через ансамблевые правила, а не через инженерию признаков.
- `max_rules` контролирует сложность итоговой модели.

### Минусы

- Медленнее при большом `max_rules`: LASSO на тысячах бинарных признаков.
- Качество зависит от качества базового леса — плохие правила = плохая LASSO.
- Категориальные признаки не поддерживаются (адаптер исключает).
- Нет SHAP поддержки (используются коэффициенты LASSO как importance).
- При малом датасете правил мало и они могут переобучиться.

### Best practices

- `max_rules=100–200` — хорошая точка; больше 500 — читаемость теряется.
- `tree_size=3–5` — глубина базовых деревьев для генерации правил.
- После обучения: `model.get_rules()` — смотреть топ-10 правил по `|coef|`.
- Совмещать с SimpleImputer + StandardScaler (уже в адаптере).
- При `max_rules > 300` увеличить `n_optuna_trials` для надёжного подбора `tree_size`.

### Литература

1. **[Predictive Learning via Rule Ensembles](https://arxiv.org/abs/0811.1679)** — Friedman & Popescu, Annals of Applied Statistics, 2008. *О чём:* Оригинальная работа RuleFit: извлечение правил из ансамблей, LASSO как метод отбора, теоретические свойства спарсификации правил; сравнение с GBDT и RF. *Польза:* Алгоритм извлечения правил; объяснение параметров `max_rules`, `tree_size`; условия, при которых RuleFit превосходит дерево решений.

2. **[imodels: A Python Package for Fitting Interpretable Models](https://arxiv.org/abs/2210.10525)** — Singh et al., JMLR 2023. *О чём:* Описание пакета imodels; RuleFit, FIGS, BRL, RIPPER в одном API. *Польза:* Практический справочник по параметрам `RuleFitRegressor/Classifier`; benchmark на множестве датасетов; рекомендации по `max_rules`.

3. **[Interpretable Machine Learning with Rule-Based Classifiers](https://arxiv.org/abs/1703.01818)** — *(аналитический обзор rule-based методов)* *О чём:* Сравнение RuleFit, RIPPER, BRL, decision lists на задачах интерпретируемости. *Польза:* Контекст для выбора между rule-based методами; когда RuleFit точнее BRL/RIPPER.

---

## FIGS — Fast Interpretable Greedy-Tree Sums (`figs`)

### Как работает

FIGS (Singh et al. 2022) строит **сумму небольших деревьев решений**. На каждой итерации алгоритм добавляет один split к тому из деревьев-компонентов, который максимально уменьшает остаток. Финальное предсказание: `y = Σ_k tree_k(x)`. Каждое дерево-компонент отвечает за свою «часть» зависимости и содержит 3–10 листьев. Адаптер: `imodels.FIGSRegressor` / `imodels.FIGSClassifier`; параметры `max_rules` и `max_trees` управляют сложностью.

### Когда полезен

- Нужна интерпретируемость уровня decision tree, но с лучшей точностью.
- Зависимость хорошо декомпозируется: одно дерево для сезонности, другое для объёма.
- Малое число признаков (< 30) — FIGS эффективнее полных ансамблей при малой размерности.
- Для классификации используется как основной imodels-классификатор (`skope_rules`, `brl`, `ripper` — специальные случаи).

### Плюсы

- Интерпретируемее случайного леса: несколько маленьких деревьев vs тысячи деревьев RF.
- `max_rules` и `max_trees` дают явный контроль над сложностью.
- Регрессия и классификация с единым API.
- Работает как fallback для `skope_rules`, `brl`, `ripper` в регрессионных задачах (в адаптере).

### Минусы

- Точность ниже GBDT — FIGS жертвует accuracy ради интерпретируемости.
- Нет нативной обработки NaN; требует импутацию (в адаптере: SimpleImputer + StandardScaler).
- При `max_trees > 10` читаемость резко снижается.
- Нет SHAP поддержки.

### Best practices

- `max_rules=10–20` и `max_trees=3–5` — хорошо читаемая модель.
- Визуализировать каждое дерево-компонент отдельно: `model.print_tree()`.
- При `max_trees=1` FIGS = одно дерево решений (сравнивать с `decision_tree` адаптером).
- Для регрессии FIGS часто лучше одного дерева; для классификации рассмотреть `skope_rules`.

### Литература

1. **[Fast Interpretable Greedy-Tree Sums (FIGS)](https://arxiv.org/abs/2201.11931)** — Singh et al., 2022. *О чём:* Оригинальная работа. Алгоритм жадного добавления сплитов; теоретические гарантии для аддитивных моделей; benchmark на реальных датасетах против RF/GBDT. *Польза:* Полное описание алгоритма; параметры `max_rules`/`max_trees`; условия, при которых FIGS конкурентоспособен с ансамблями.

2. **[imodels: A Python Package for Fitting Interpretable Models](https://arxiv.org/abs/2210.10525)** — Singh et al., JMLR 2023. *О чём:* Описание imodels с FIGS, RuleFit, SKOPE-Rules, BRL, RIPPER в одном пакете. *Польза:* Сравнительный benchmark всех методов из imodels; рекомендации по выбору.

---

## SKOPE-Rules (`skope_rules`)

### Как работает

SKOPE-Rules (Goix et al. 2018) — алгоритм извлечения правил с фильтрацией по precision и recall. Шаги: (1) обучить случайный лес / ET, (2) извлечь все пути как правила, (3) отфильтровать правила по минимальным порогам precision/recall, (4) дедуплицировать похожие правила (ROC-distance). Результат — небольшой набор высококачественных правил. Адаптер: `imodels.SkopeRulesClassifier` (только классификация; для регрессии используется FIGSRegressor).

### Когда полезен

- Нужны точные правила с гарантиями precision/recall, а не максимальная точность.
- Аудит модели: «найди правила, которые с precision ≥ 0.8 предсказывают высокую комиссию».
- Дисбаланс классов: правила фокусируются на редком классе через precision/recall пороги.

### Плюсы

- Правила высокого качества с явными precision/recall характеристиками.
- Дедупликация: избегает избыточных похожих правил.
- Каждое правило читаемо: `income > 5000 AND transactions < 3 → high commission (P=0.87, R=0.43)`.
- `feature_names` сохраняются в правилах.

### Минусы

- Только классификация (регрессия через FIGS в адаптере).
- Число правил зависит от данных — может быть нулевым при строгих порогах.
- Медленнее FIGS при большом `n_estimators`.
- Нет SHAP поддержки.

### Best practices

- `precision_min=0.5, recall_min=0.01` — разумные дефолты (настраиваются в imodels).
- `n_estimators=20–30`, `max_depth=3–4` — достаточно для большинства задач.
- Проверять `model.rules_`: если правил мало — снизить `precision_min`.
- Комбинировать с FIGS для регрессии и SKOPE для классификации в одном pipeline.

### Литература

1. **[SKOPE-RULES: A Python package for rule mining](https://github.com/scikit-learn-contrib/skope-rules)** — Goix et al., GitHub 2018. *(документация пакета)* *О чём:* Описание алгоритма SKOPE-Rules; precision/recall фильтрация; дедупликация. *Польза:* Практический справочник по параметрам; примеры читаемых правил.

2. **[imodels: A Python Package for Fitting Interpretable Models](https://arxiv.org/abs/2210.10525)** — Singh et al., JMLR 2023. *О чём:* Сравнение SKOPE-Rules с другими rule-based методами в imodels. *Польза:* Benchmark; условия, при которых SKOPE-Rules выигрывает у RIPPER/BRL.

---

## BRL — Bayesian Rule List (`brl`)

### Как работает

BRL (Letham et al. 2015) — байесовский подход к построению **упорядоченного списка if-then правил** (decision list). Правила предварительно генерируются (FP-growth для частых паттернов), затем оптимальный список выбирается через MCMC-сэмплирование posteriori. Функция полезности балансирует точность и компактность списка. Результат: `if rule1 then c1 else if rule2 then c2 else c_default`. Адаптер: `imodels.BayesianRuleListClassifier` (только классификация).

### Когда полезен

- Нужен **аудируемый** список правил для регуляторных требований.
- Небольшой датасет (< 10k строк) — MCMC работает разумное время.
- Порядок правил важен (decision list vs decision set).
- Explainability для некомпетентной аудитории: простой if-else список.

### Плюсы

- Bayesian uncertainty: `listlengthprior` и `listwidthprior` явно контролируют сложность.
- Оптимален для compliance: каждое правило имеет статистическое обоснование.
- Читаемый вывод: `model.print_list()`.

### Минусы

- Только классификация (регрессия через FIGS в адаптере).
- MCMC медленно: на датасетах > 50k строк обучение может занять минуты.
- Точность обычно ниже FIGS/RuleFit — максимальная читаемость за счёт accuracy.
- Нет SHAP поддержки.

### Best practices

- `listlengthprior=5–7` — типичная длина списка; меньше → компактнее.
- `listwidthprior=2–3` — ширина каждого правила (число условий).
- Использовать только на задачах с < 50k строк и < 30 признаками.
- Дискретизация непрерывных признаков (`KBinsDiscretizer`, 4 квантильных бина) выполняется адаптером автоматически перед fit/predict — вручную бинировать не нужно.

### Литература

1. **[Interpretable Classifiers Using Rules and Bayesian Analysis](https://arxiv.org/abs/1511.01247)** — Letham et al., Annals of Applied Statistics, 2015. *О чём:* Оригинальная работа BRL: MCMC для decision list; prior на длину и ширину списка; сравнение с другими rule-based методами на медицинских данных. *Польза:* Теоретическое обоснование prior; понимание параметров `listlengthprior`/`listwidthprior`; условия применения BRL (compliance, audit).

2. **[imodels: A Python Package for Fitting Interpretable Models](https://arxiv.org/abs/2210.10525)** — Singh et al., JMLR 2023. *О чём:* Реализация BRL в imodels; параметры, API, benchmark. *Польза:* Практический справочник; сравнение с RIPPER и FIGS.

---

## RIPPER (`ripper`)

> **Нерабочее состояние на текущей версии пакета** — `imodels.RIPPERClassifier` отсутствует
> в `imodels==2.0.4` (класс удалён/переименован в пакете начиная с некоторой версии, при этом
> `pip install imodels` по умолчанию ставит именно её). `name='ripper'` в `IModelsClassifier`
> падает с явным `ImportError`, объясняющим проблему, вместо тихого сбоя. Альтернативы:
> `'figs'`/`'skope_rules'`/`'brl'` из того же адаптера, либо отдельный пакет `wittgenstein`
> (`pip install wittgenstein`) с независимой реализацией RIPPER (не подключён к ml_toolkit).
> Материал ниже описывает алгоритм и остаётся полезным как справка, но best practices по
> тюнингу неприменимы, пока класс недоступен.

### Как работает

RIPPER (Cohen 1995 — Repeated Incremental Pruning to Produce Error Reduction) — жадная индукция правил для классификации. Алгоритм итеративно добавляет конъюнкции условий (правила), каждое из которых покрывает максимум объектов целевого класса, затем прунит правило для уменьшения ошибки. Параметр `k` — число passes оптимизации. Результат: компактный набор правил (decision set). Адаптер: `imodels.RIPPERClassifier`.

### Когда полезен

- Нужны чёткие правила классификации без допущений о распределении.
- Дисбаланс классов: RIPPER строит правила для каждого класса последовательно.
- Быстрый baseline правил: RIPPER обучается быстрее BRL (жадный алгоритм vs MCMC).
- Задача с небольшим числом важных признаков (< 20).

### Плюсы

- Быстрее BRL — жадный алгоритм без MCMC.
- Компактный результат: обычно 5–15 правил.
- Читаемые правила с оригинальными именами признаков.
- `k > 1` — дополнительные passes оптимизации улучшают качество.

### Минусы

- Только классификация (регрессия через FIGS в адаптере).
- Жадный алгоритм не гарантирует оптимальный набор правил.
- При `k` > 3 заметно замедляется.
- Нет SHAP поддержки.

### Best practices

- `k=2` — стандартный дефолт; увеличивать только при явном недообучении.
- При дисбалансе: убедиться в правильном порядке классов (редкий класс = positive).
- Сравнивать с BRL: если RIPPER правила качественнее — данные хорошо разделимы.
- `n_optuna_trials=5–10` достаточно — пространство гиперпараметров RIPPER мало (`k=1..5`).

### Литература

1. **[Fast Effective Rule Induction](https://dl.acm.org/doi/10.5555/3091044.3091078)** — Cohen, ICML 1995. *(не на arxiv)* *О чём:* Оригинальная работа RIPPER: алгоритм жадной индукции правил; pruning через MDL; multi-pass оптимизация (параметр k). *Польза:* Полное описание алгоритма; понимание параметра `k` и его влияния на качество.

2. **[imodels: A Python Package for Fitting Interpretable Models](https://arxiv.org/abs/2210.10525)** — Singh et al., JMLR 2023. *О чём:* Реализация RIPPER в imodels; benchmark против BRL/FIGS/SKOPE. *Польза:* Условия, при которых RIPPER эффективнее других rule-based методов.

---

## NAM — Neural Additive Models (`nam`)

> **Устарело** — используй `name: gaminet`; при `n_interactions=0` GAMINET эквивалентен NAM (только главные эффекты, без взаимодействий признаков).

---

## GAMINET (`gaminet`)

### Как работает

GAMINET (Yang et al. 2021) расширяет NAM, добавляя **попарные взаимодействия**: для каждой пары признаков `(i, j)` обучается interaction network `f_ij(x_i, x_j)`. Предсказание: `y = Σ f_i(x_i) + Σ f_ij(x_i, x_j) + bias`. Параметр `n_interactions` ограничивает число учитываемых пар (top-K). Реализация: кастомный PyTorch. Классификация: LogisticRegression на QuantileTransformer-признаках (как у NAM).

При `n_interactions=0` GAMINET эквивалентен NAM: только главные эффекты `f_i(x_i)` без взаимодействий.

### Когда полезен

- Взаимодействия признаков подозреваются, но нужна интерпретируемость через shape functions.
- NAM недостаточно точен — GAMINET добавляет пары взаимодействий.
- Нужна визуализация: 2D surface plot для каждой пары `f_ij`.
- Аналог EBM с `interactions > 0`, но на нейросетевой основе.

### Плюсы

- Захватывает попарные взаимодействия при сохранении читаемости.
- `n_interactions` явно ограничивает число пар — баланс точности и объяснимости.
- 2D визуализация каждой `f_ij(x_i, x_j)` — полное понимание взаимодействий.
- PyTorch уже установлен — нет дополнительных зависимостей.

### Минусы

- Медленнее NAM: O(n_interactions × n_epochs) операций.
- При `n_interactions > 10` интерпретируемость снижается.
- Только регрессия нативно; классификация через LogisticRegression.
- Нет SHAP поддержки.
- Риск переобучения при малом датасете с большим `n_interactions`.

### Best practices

- `n_interactions=5–10` — достаточно для большинства задач; больше 15 редко оправдано.
- `hidden_dim=64`, `n_layers=2` — сбалансированные дефолты.
- Сравнивать с EBM с `interactions=10`: если EBM точнее — GAMINET не даёт прироста.
- Visualize 2D plots для топ-5 пар по важности перед интерпретацией.
- Увеличить `n_epochs=300` при сложных взаимодействиях.

### Литература

1. **[GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions](https://arxiv.org/abs/2003.07132)** — Yang et al., Pattern Recognition 2021. *О чём:* Оригинальная работа GAMINET: архитектура interaction networks; sparsity regularization для выбора взаимодействий; сравнение с NAM, EBM, GBDT. *Польза:* Полное описание архитектуры; параметр `n_interactions`; 2D visualization methodology.

2. **[Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912)** — Agarwal et al., NeurIPS 2021. *О чём:* NAM как базовый компонент GAMINET; описание feature networks; benchmark. *Польза:* Контекст NAM vs GAMINET; понимание, когда взаимодействия улучшают результат.

3. **[Interpretable Neural Networks with Frank-Wolfe: Sparse Relevance Maps and Relevance Orderings](https://arxiv.org/abs/2110.01248)** — *(теория интерпретируемых нейросетей, применима к GAMINET)* *О чём:* Теоретический анализ sparse interpretation в additive нейросетях; условия, при которых additive decomposition корректна. *Польза:* Математическое обоснование аддитивности с взаимодействиями; ограничения GAM-структуры.

---

## Soft Decision Tree (`soft_decision_tree`)

### Как работает

Soft Decision Tree — дерево с **мягкими (вероятностными) разбиениями**: вместо жёстких `if x > t` используется sigmoid `σ(w·x + b)`, задающий вероятность перейти в правый/левый дочерний узел. Листья содержат параметризованные константы. Обучение: SGD (Adam) на суммарной log-вероятности правильного пути × loss листа (L1Loss для регрессии, BCELoss для классификации). Early stopping по validation loss. Реализация: кастомный PyTorch (нет дополнительных зависимостей).

### Когда полезен

- Нужна дифференцируемая альтернатива Decision Tree — обучение end-to-end с другими нейросетями.
- Данные содержат плавные границы решений, а не резкие пороги.
- Эксперименты с гибридными архитектурами: Soft DT как компонент нейросети.
- Небольшой датасет, где стандартное дерево нестабильно из-за дискретных разбиений.

### Плюсы

- Градиентное обучение — стабильнее стандартного CART при малом датасете.
- Дифференцируемость: легко встраивать в нейросетевые pipeline.
- `depth` явно контролирует сложность (как `max_depth` в CART).
- Early stopping предотвращает переобучение.

### Минусы

- Менее интерпретируемо, чем обычное Decision Tree (вероятностные пути не читаются как правила).
- Медленнее CART: каждая итерация — forward/backward pass по всему дереву.
- При глубоком дереве (`depth > 6`) gradients затухают.
- Нет SHAP TreeExplainer (нестандартная архитектура).
- Сложнее тюнить: `lr`, `n_epochs`, `depth` взаимозависимы.

### Best practices

- `depth=3–5` — хорошая точка для интерпретируемости и скорости.
- `lr=0.01`, `n_epochs=200–300` — разумные дефолты.
- Early stopping `patience=20` — достаточно; уменьшать только при очень коротком обучении.
- Сравнивать с обычным `decision_tree`: если CART лучше — мягкие разбиения не помогают.
- Использовать StandardScaler (уже в адаптере) — sigmoid чувствительна к масштабу.

### Литература

1. **[Soft Decision Trees](https://arxiv.org/abs/1708.05256)** — Irsoy & Alpaydın, ICPR 2012 / arXiv 2017. *О чём:* Оригинальная работа: sigmoid разбиения в дереве; обучение через backpropagation; сравнение с CART и нейросетями на benchmark задачах. *Польза:* Теоретическое обоснование soft splits; условия, при которых Soft DT превосходит CART.

2. **[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312)** — Popov et al., ICLR 2020. *О чём:* NODE — ансамбль differentiable decision trees; soft splits с entmax; сравнение с GBDT. *Польза:* Контекст soft decision trees в современном ML; понимание, почему Soft DT уступает GBDT без ансамблирования.

3. **[Hierarchical Mixture of Experts and the EM Algorithm](https://arxiv.org/abs/1506.06203)** — Jordan & Jacobs. *(теоретический контекст мягких деревьев)* *О чём:* Теоретические основы мягких иерархических разбиений; связь с HME (Hierarchical Mixture of Experts) — предшественником Soft DT. *Польза:* Глубокое понимание градиентного обучения деревьев; почему мягкие разбиения дают более стабильный gradient сигнал чем CART.

---

## Locally Linear Forest (`locally_linear_forest`)

### Как работает

Locally Linear Forest (LLF) — гибрид Random Forest и линейной регрессии. Random Forest используется для оценки **близости** между объектами через `apply()`: два объекта близки, если часто попадают в один лист. Для каждого query-объекта находятся top-N ближайших training-объектов по proximity, затем обучается взвешенная Ridge-регрессия только на этих соседях. Предсказание = Ridge(x) на локальном neighbourhood. Реализация: кастомная sklearn-совместимая (`_LocallyLinearForest`).

Ограничение: `_MAX_LLF_TRAIN_ROWS = 2000` для предотвращения O(n_pred × n_train) bottleneck.

### Когда полезен

- Зависимости **локально линейны**, но глобально нелинейны.
- Нужна интерпретируемость каждого предсказания: «для этого клиента важны вот эти признаки».
- Инстанс-level объяснения: у каждого объекта своя локальная линейная модель.
- Данные < 2000 строк — LLF работает в полную силу в пределах `_MAX_LLF_TRAIN_ROWS`.

### Плюсы

- Каждое предсказание объяснимо через локальные коэффициенты Ridge.
- Proximity из RF более устойчива, чем Euclidean distance в высоких измерениях.
- Классификация: `RandomForestClassifier` (без LLF-fallback, поскольку Ridge не работает для бинарных).
- Нет дополнительных зависимостей — использует только sklearn RF.

### Минусы

- Медленно при инференсе: для каждого query-объекта — поиск N соседей + обучение Ridge.
- `_MAX_LLF_TRAIN_ROWS = 2000` ограничивает применение на больших данных.
- Нет SHAP поддержки (нестандартный estimator).
- `ridge_alpha` и `n_neighbors` сильно зависят от данных — тюнинг критичен.
- RF proximity не масштабируется на > 100k строк без approximate NN.

### Best practices

- `n_neighbors=50–100` — хорошая точка; больше → стабильнее, но менее локально.
- `ridge_alpha=1.0–10.0` разумный диапазон; увеличивать при multicollinearity.
- `max_depth=5–10` для RF proximity — неглубокие деревья дают более грубые, но более надёжные proximity.
- Использовать только если обучающая выборка < 2000 строк (иначе адаптер обрезает).
- Интерпретировать локальные коэффициенты: `forest.local_coef(x_query)` (если реализовано).

### Литература

1. **[Generalized Random Forests](https://arxiv.org/abs/1610.01271)** — Athey, Tibshirani & Wager, Annals of Statistics 2019. *О чём:* Теоретические основы обобщённых RF как non-parametric estimator с RF proximity; доказательство состоятельности; связь с Locally Linear Forests через locally weighted regression. *Польза:* Математическое обоснование LLF; понимание условий состоятельности и выбора `n_neighbors`.

2. **[Estimating and Explaining Machine Learning Predictions at the Instance Level](https://arxiv.org/abs/1802.03865)** — *(instance-level explanations через локальную линейную аппроксимацию)* *О чём:* LIME как экстремальный случай locally linear approximation; сравнение с RF proximity; условия, при которых локальная линейность — разумное допущение. *Польза:* Контекст LLF среди методов instance-level explanation; когда Ridge коэффициенты корректно интерпретируются как локальная важность признаков.

3. **[Local Linear Forests](https://arxiv.org/abs/1807.11408)** — Friedberg et al., JASA 2021. *О чём:* Оригинальная работа LLF: алгоритм, теоретические гарантии для locally linear regression с RF proximity weights; asymptotic normality предсказаний. *Польза:* Полное описание алгоритма; параметры `n_neighbors`, `ridge_alpha`; условия, при которых LLF превосходит обычный RF.

---

## Сравнительная таблица

Колонки:
- **Интерпр.** — ★★★★★ нативная (правила/shape functions/coef_) → ★★★★☆ частичная (coef_ или локальные коэф.) → ★★★☆☆ через SHAP → ★★☆☆☆ черный ящик
- **Взаим.** — захватывает ли взаимодействия признаков: ✓ явно / Ч = неявно через структуру / ✗ аддитивная модель
- **Зависим.** — `core` (основные deps) / `sklearn` (всегда в core) / `torch` (всегда в core) / `[trees]` / `[interp]`
- **Отбор фич** — встроенный отбор признаков: ✅ модель сама исключает нерелевантные (L1 / GCV / AutoML) / ⚠️ частичный (авто-регуляризация без обнуления, диагностика p-values, shape functions) / ❌ нет
- **Нужен отбор** — рекомендуется ли подавать уже отфильтрованный набор фичей: ✅ обязательно / ⚠️ желательно / ❌ нет

| Модель                   | Скорость   | Качество | Интерпр.  | SHAP | NaN нат. | Кат. нат.    | MAE          | Взаим. | Зависим.  | Отбор фич  | Нужен отбор    |
|--------------------------|------------|----------|-----------|------|----------|--------------|--------------|--------|-----------|------------|----------------|
| `catboost`               | Средняя    | ★★★★★    | ★★★☆☆     | ✓    | ✓        | ✓            | ✓ (опц.)     | ✓      | core      | ❌         | ❌             |
| `lightgbm`               | Быстрая    | ★★★★★    | ★★★☆☆     | ✓    | ✓        | Частично     | ✓            | ✓      | core      | ❌         | ❌             |
| `xgboost`                | Средняя    | ★★★★★    | ★★★☆☆     | ✓    | ✓        | ✗            | ✓            | ✓      | core      | ❌         | ⚠️ желательно  |
| `lama`                   | Медленная  | ★★★★☆    | ★★☆☆☆     | ✗    | ✓        | ✓            | ✓            | ✓      | core      | ✅ AutoML  | ❌             |
| `tabm`                   | Медленная  | ★★★★☆    | ★★☆☆☆     | ✗    | ✓ (имп.) | ✓ (ordinal)  | ✗            | ✓      | torch     | ❌         | ⚠️ желательно  |
| `random_forest`          | Средняя    | ★★★☆☆    | ★★★☆☆     | ✓    | ✗ (имп.) | ✗            | Ч (абс.)     | Ч      | sklearn   | ❌         | ❌             |
| `extra_trees`            | Быстрая    | ★★★☆☆    | ★★★☆☆     | ✓    | ✗ (имп.) | ✗            | Ч            | Ч      | sklearn   | ❌         | ❌             |
| `hist_gbm`               | Быстрая    | ★★★★☆    | ★★★☆☆     | ✓    | ✓        | ✓ (≥1.2)     | ✓            | ✓      | sklearn   | ❌         | ❌             |
| `quantile_forest`        | Средняя    | ★★★☆☆    | ★★★☆☆     | ✓    | ✗ (имп.) | ✗            | ✓ (q=0.5)    | Ч      | [trees]   | ❌         | ❌             |
| `oblique_forest`         | Медленная  | ★★★☆☆    | ★★★☆☆     | ✓    | ✗ (имп.) | ✗            | Ч            | ✓      | [trees]   | ❌         | ❌             |
| `mondrian`               | Медленная  | ★★☆☆☆    | ★★☆☆☆     | ✗    | ✗ (имп.) | ✗            | ✗            | Ч      | [trees]   | ❌         | ❌             |
| `ridge`                  | Мгновенная | ★★☆☆☆    | ★★★★☆     | ✓*   | ✗        | ✗            | ✗ (MSE)      | ✗      | sklearn   | ❌         | ⚠️ желательно  |
| `elasticnet`             | Быстрая    | ★★☆☆☆    | ★★★★★     | ✓*   | ✗        | ✗            | ✗ (MSE)      | ✗      | sklearn   | ✅ L1      | ❌             |
| `huber`                  | Быстрая    | ★★★☆☆    | ★★★★☆     | ✓*   | ✗        | ✗            | Ч            | ✗      | sklearn   | ❌         | ⚠️ желательно  |
| `tweedie`                | Быстрая    | ★★★☆☆    | ★★★★☆     | ✓*   | ✗        | ✗            | ✗            | ✗      | sklearn   | ❌         | ⚠️ желательно  |
| `quantile`               | Медленная  | ★★★☆☆    | ★★★★☆     | ✓*   | ✗        | ✗            | ✓ (точно)    | ✗      | sklearn   | ❌         | ⚠️ желательно  |
| `bayesian_ridge`         | Быстрая    | ★★☆☆☆    | ★★★★★     | ✓*   | ✗        | ✗            | ✗ (MSE)      | ✗      | sklearn   | ⚠️ авто-α  | ❌             |
| `ebm`                    | Медленная  | ★★★★☆    | ★★★★★     | ✗†   | ✓        | ✗            | ✗            | Ч      | [interp]  | ⚠️ shape   | ❌             |
| `pygam`                  | Быстрая    | ★★★☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | ✗            | ✗      | [interp]  | ⚠️ p-val   | ✅ обязательно |
| `mars`                   | Средняя    | ★★★☆☆    | ★★★★☆     | ✗    | ✗ (имп.) | ✗            | ✗            | Ч      | [interp]  | ✅ GCV     | ❌             |
| `decision_tree`          | Мгновенная | ★★☆☆☆    | ★★★★★     | ✓    | ✗ (имп.) | ✗            | ✓ (MAE crit.)| Ч      | sklearn   | ❌         | ❌             |
| `linear_tree`            | Быстрая    | ★★★☆☆    | ★★★★☆     | ✗    | ✗ (имп.) | ✗            | ✓ (MAE crit.)| Ч      | [interp]  | ❌         | ⚠️ желательно  |
| `rulefit`                | Медленная  | ★★★☆☆    | ★★★★☆     | ✗    | ✗ (имп.) | ✗            | ✗            | ✓      | [interp]  | ✅ Lasso   | ❌             |
| `figs`                   | Быстрая    | ★★☆☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | ✗            | Ч      | [interp]  | ❌         | ❌             |
| `skope_rules`            | Средняя    | ★★☆☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | FIGS (reg)   | ✓      | [interp]  | ❌         | ❌             |
| `brl`                    | Медленная  | ★★☆☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | FIGS (reg)   | ✓      | [interp]  | ❌         | ✅ < 30 фич    |
| `ripper` ‡               | Быстрая    | ★★☆☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | FIGS (reg)   | ✓      | [interp]  | ❌         | ✅ < 20 фич    |
| `gaminet`                | Медленная  | ★★★☆☆    | ★★★★★     | ✗    | ✗ (имп.) | ✗            | ✓ (L1Loss)   | ✓      | torch     | ❌         | ⚠️ желательно  |
| `soft_decision_tree`     | Медленная  | ★★☆☆☆    | ★★★☆☆     | ✗    | ✗ (имп.) | ✗            | ✓ (L1Loss)   | Ч      | torch     | ❌         | ❌             |
| `locally_linear_forest`  | Медленная  | ★★★☆☆    | ★★★★☆     | ✗    | ✗ (имп.) | ✗            | ✗            | Ч      | sklearn   | ❌         | ❌             |

\* LinearExplainer в SHAP теоретически поддерживается, но не реализован в адаптере (используется `|coef_|`). † EBM имеет собственный explain API (`ebm.explain_global()`), SHAP TreeExplainer не поддерживается. ‡ `ripper` нерабочий на текущей закреплённой версии `imodels` (класс отсутствует в пакете) — см. раздел «RIPPER» выше. Ч = частично / неявно через структуру модели. Отбор фич — ✅ Lasso: ElasticNet/RuleFit зануляют коэффициенты через L1; ✅ GCV: MARS удаляет basis-функции backward pruning; ✅ AutoML: LAMA встроенный feature selection; ⚠️ авто-α: BayesianRidge само настраивает силу регуляризации, но не обнуляет; ⚠️ shape/p-val: EBM и pyGAM диагностируют важность постфактум без автоматического удаления. Нужен отбор — ✅ обязательно: pygam деградирует при `n_features > 50`; ✅ < N фич: brl/ripper рассчитаны на небольшое число признаков; ⚠️ желательно: линейные модели (кроме elasticnet) и нейросети чувствительны к шуму и мультиколлинеарности.

---

## Общие источники

1. **[Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)** — Akiba et al., KDD 2019. *О чём:* TPE-алгоритм (Tree-structured Parzen Estimator) для Bayesian optimization; define-by-run API; pruning триалов на основе промежуточных результатов. *Польза:* Понимание TPE объясняет, почему Optuna эффективен при малом числе trials; параметр `n_optuna_trials=15–30` достаточен для большинства моделей именно благодаря TPE.

2. **[Scikit-learn: Machine Learning in Python](https://arxiv.org/abs/1201.0490)** — Pedregosa et al., JMLR 2011. *О чём:* Описание sklearn API, реализации всех линейных моделей, preprocessing pipeline, cross-validation, метрик. *Польза:* Референс по edge cases, поведению при singularity, параметрам solver; актуально для всех адаптеров на основе sklearn.

3. **[Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)** — Grinsztajn et al., NeurIPS 2022. *О чём:* Систематический бенчмарк GBDT vs DL на 45 датасетах; выявляет структурные свойства данных (irregular patterns, uninformative features), предопределяющие победителя. *Польза:* Главный источник для принятия решения «какую модель попробовать первой»; показывает, что GBDT стабильно побеждает без тщательного тюнинга нейросетей на большинстве таблиц.

4. **[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** — Lundberg & Lee, NeurIPS 2017. *О чём:* Единая теория объяснений на основе значений Шепли; доказывает, что SHAP удовлетворяет аксиомам локальной точности, постоянства и отсутствия фиктивных признаков. *Польза:* Читать для правильной интерпретации SHAP-графиков; без понимания базовых аксиом легко сделать ошибочные выводы о причинно-следственных связях.

---

## Как добавить новую модель

1. Создать `ml_toolkit/models/_{name}.py` с классами `XxxRegressor`/`XxxClassifier`, наследующими `BaseModel` (см. «Структура адаптеров» выше).
2. Добавить оба класса в `_LAZY_CLASSES` и `__all__` в `ml_toolkit/models/__init__.py`.
3. Если модель поддерживает SHAP/permutation importance нестандартным способом, добавить обработку в `ml_toolkit/model_explainer/`.

---

## Параметры model_settings

```python
# Gradient Boosting
{'reg_model': {'name': 'catboost', 'n_optuna_trials': 30},
 'cls_model': {'name': 'catboost', 'n_optuna_trials': 20, 'error_threshold': 0.25}}

# Линейные
{'reg_model': {'name': 'ridge', 'n_optuna_trials': 20},
 'cls_model': {'name': 'ridge', 'n_optuna_trials': 15, 'error_threshold': 0.25}}

# BayesianRidge — без Optuna для регрессии
{'reg_model': {'name': 'bayesian_ridge', 'n_optuna_trials': 0},
 'cls_model': {'name': 'bayesian_ridge', 'n_optuna_trials': 15, 'error_threshold': 0.25}}

# LAMA
{'reg_model': {'name': 'lama', 'n_optuna_trials': 1, 'timeout': 120},
 'cls_model': {'name': 'lama', 'n_optuna_trials': 1, 'timeout': 60, 'error_threshold': 0.25}}

# TabM (CPU)
{'reg_model': {'name': 'tabm', 'n_optuna_trials': 5,
               'n_epochs_per_trial': 10, 'n_epochs_final': 50,
               'patience': 5, 'device': 'cpu'},
 'cls_model': {'name': 'tabm', 'n_optuna_trials': 3,
               'n_epochs_per_trial': 10, 'n_epochs_final': 50,
               'patience': 5, 'device': 'cpu', 'error_threshold': 0.25}}

# EBM
{'reg_model': {'name': 'ebm', 'n_optuna_trials': 15},
 'cls_model': {'name': 'ebm', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# pyGAM
{'reg_model': {'name': 'pygam', 'n_optuna_trials': 10},
 'cls_model': {'name': 'pygam', 'n_optuna_trials': 8, 'error_threshold': 0.25}}

# MARS
{'reg_model': {'name': 'mars', 'n_optuna_trials': 15},
 'cls_model': {'name': 'mars', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# Decision Tree
{'reg_model': {'name': 'decision_tree', 'n_optuna_trials': 20},
 'cls_model': {'name': 'decision_tree', 'n_optuna_trials': 15, 'error_threshold': 0.25}}

# Linear Tree
{'reg_model': {'name': 'linear_tree', 'n_optuna_trials': 20},
 'cls_model': {'name': 'linear_tree', 'n_optuna_trials': 15, 'error_threshold': 0.25}}

# RuleFit
{'reg_model': {'name': 'rulefit', 'n_optuna_trials': 15},
 'cls_model': {'name': 'rulefit', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# FIGS
{'reg_model': {'name': 'figs', 'n_optuna_trials': 15},
 'cls_model': {'name': 'figs', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# SKOPE-Rules
{'reg_model': {'name': 'skope_rules', 'n_optuna_trials': 15},  # reg→FIGSRegressor
 'cls_model': {'name': 'skope_rules', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# BRL
{'reg_model': {'name': 'brl', 'n_optuna_trials': 10},  # reg→FIGSRegressor
 'cls_model': {'name': 'brl', 'n_optuna_trials': 8, 'error_threshold': 0.25}}

# RIPPER
{'reg_model': {'name': 'ripper', 'n_optuna_trials': 10},  # reg→FIGSRegressor
 'cls_model': {'name': 'ripper', 'n_optuna_trials': 8, 'error_threshold': 0.25}}

# GAMINET
{'reg_model': {'name': 'gaminet', 'n_optuna_trials': 15},
 'cls_model': {'name': 'gaminet', 'n_optuna_trials': 10, 'error_threshold': 0.25}}

# Soft Decision Tree
{'reg_model': {'name': 'soft_decision_tree', 'n_optuna_trials': 10},
 'cls_model': {'name': 'soft_decision_tree', 'n_optuna_trials': 8, 'error_threshold': 0.25}}

# Locally Linear Forest
{'reg_model': {'name': 'locally_linear_forest', 'n_optuna_trials': 15},
 'cls_model': {'name': 'locally_linear_forest', 'n_optuna_trials': 10, 'error_threshold': 0.25}}
```
