# ml_toolkit/model_evaluation

Пакет для оценки качества моделей классификации и регрессии. Единый API для метрик, анализа порогов, визуализаций и HTML-отчётов.

## Структура

```
model_evaluation/
    _base.py             — BaseEvaluator: хранение сплитов/метрик, metrics(), compare_splits()
    _classification.py   — ClassificationEvaluator + пресеты + фабрики
    _regression.py       — RegressionEvaluator + пресеты
    _comparison.py       — compare_models(), plot_model_comparison(), plot_model_heatmap(), plot_model_delta()
    __init__.py          — публичный экспорт
    generate_example.py  — генератор HTML-примера (запустить: uv run python ml_toolkit/model_evaluation/generate_example.py)
    example.html         — сгенерированный HTML-пример
```

## Быстрый старт

```python
from ml_toolkit.model_evaluation import ClassificationEvaluator, RegressionEvaluator
from ml_toolkit.model_evaluation import precision_at_k, recall_at_k, lift_at_k, f1_at_threshold
```

Также реэкспортируется из `ml_toolkit.models` (обратная совместимость):

```python
from ml_toolkit.models import ClassificationEvaluator, RegressionEvaluator
```

---

## ClassificationEvaluator

```python
ev = ClassificationEvaluator(task='binary')   # или task='multiclass'
```

### Регистрация сплитов

Все методы возвращают `self` — поддерживается method chaining.

```python
ev.add('train', y_true_train, y_proba_train)
ev.add('valid', y_true_valid, y_proba_valid)
ev.add('test',  y_true_test,  y_proba_test)
```

`y_proba` — 1D array вероятностей (binary) или 2D матрица `(n, K)` (multiclass).

### Регистрация метрик

```python
# Пресет по строке
ev.add_metric('roc_auc')
ev.add_metric('pr_auc')

# Набор по умолчанию (roc_auc, pr_auc, log_loss, brier, ks, gini, mcc, ece)
ev.add_default_metrics()

# Параметризованные фабрики
ev.add_metric(precision_at_k(0.10), name='precision@10%')
ev.add_metric(precision_at_k(0.20), name='precision@20%')
ev.add_metric(recall_at_k(0.05),    name='recall@5%')
ev.add_metric(lift_at_k(0.10),      name='lift@10%')
ev.add_metric(f1_at_threshold(0.3), name='f1@t=0.3')

# Любой callable (y_true, y_proba) → float
ev.add_metric(my_fn, name='custom')

# Bulk через dict {display_name: preset_str | callable}
ev.add_metrics({
    'roc_auc':      'roc_auc',
    'precision@10%': precision_at_k(0.10),
    'precision@20%': precision_at_k(0.20),
})
```

#### Доступные пресеты

| Имя | Описание | Порог |
|-----|----------|-------|
| `roc_auc` | ROC-AUC (OvR macro для multiclass) | нет |
| `pr_auc` | Average Precision (macro для multiclass) | нет |
| `log_loss` | Cross-entropy | нет |
| `brier` | Brier score (среднее OvR для multiclass) | нет |
| `ks` | Kolmogorov-Smirnov statistic | нет, только binary |
| `gini` | Gini = 2 × ROC-AUC − 1 | нет |
| `mcc` | Matthews Correlation Coefficient | 0.5 |
| `ece` | Expected Calibration Error (10 bins) | нет |
| `accuracy` | Точность | 0.5 |
| `balanced_accuracy` | Balanced accuracy | 0.5 |
| `f1` | F1-score (macro для multiclass) | 0.5 |
| `precision` | Precision (macro для multiclass) | 0.5 |
| `recall` | Recall (macro для multiclass) | 0.5 |
| `cohen_kappa` | Cohen's Kappa | 0.5 |

### Таблица метрик

```python
# Все сплиты, все метрики
df = ev.metrics()

# Фильтрация — оба параметра опциональны
df = ev.metrics(splits=['valid', 'test'])
df = ev.metrics(metrics=['roc_auc', 'pr_auc', 'precision@10%'])
df = ev.metrics(splits=['valid', 'test'], metrics=['roc_auc', 'pr_auc'])
```

Возвращает `pd.DataFrame`: строки = метрики, столбцы = сплиты.

### Сравнение сплитов

```python
# metric | valid | test | delta | ratio
df = ev.compare_splits(ref='valid', target='test')

# Детектор переобучения
df = ev.compare_splits(ref='train', target='valid')
```

### Анализ порога (только binary)

```python
# threshold | precision | recall | f1 | accuracy | specificity
df = ev.threshold_scan(split='valid', n_points=200)

# Оптимальный порог по заданной метрике
result = ev.best_threshold(metric='f1', split='valid')
# → {'threshold': 0.38, 'f1': 0.71, 'precision': 0.74, 'recall': 0.68, ...}
```

### PSI (только binary)

```python
total_psi, bin_df = ev.psi(ref='valid', target='test', n_bins=10)
# bin_df: bin | valid_pct | test_pct | psi
```

Интерпретация: PSI < 0.1 — стабильно, 0.1–0.25 — небольшой сдвиг, > 0.25 — существенный сдвиг.

---

## Визуализации (ClassificationEvaluator)

Все plot-методы:
- `splits=None` → все зарегистрированные сплиты
- `path=None` → `plt.show()` (notebook); `path='file.png'` → сохранить на диск
- `ax=` / `axes=` → нарисовать на существующем `Axes` (см. ниже)

### Компоновка на одном рисунке

Методы с одной панелью принимают `ax=`, методы с несколькими панелями — `axes=`:

```python
import matplotlib.pyplot as plt

# Два графика рядом
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ev.plot_roc(splits=['valid', 'test'], ax=ax1)
ev.plot_pr(splits=['valid', 'test'], ax=ax2)
plt.tight_layout()
plt.savefig('curves.png')

# Гистограммы скоров (axes= — по одной панели на сплит)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ev.plot_score_distribution(splits=['valid', 'test'], axes=axes)
plt.tight_layout()
plt.savefig('scores.png')

# PSI требует ровно 2 панели
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4))
ev.plot_psi(ref='valid', target='test', axes=[ax_left, ax_right])
```

Если `ax=` / `axes=` не передан — метод создаёт собственный `Figure` и сам его показывает/сохраняет.

### Кривые (ax=)

```python
ev.plot_roc(splits=['valid', 'test'])          # ROC + AUC в легенде
ev.plot_pr(splits=['valid', 'test'])           # Precision-Recall + AP в легенде
```

### Распределение скоров (axes=)

```python
ev.plot_score_distribution(splits=['valid', 'test'])  # гистограмма по классам
ev.plot_score_cdf(splits=['valid', 'test'])           # CDF по классам; макс. разрыв = KS
```

### Калибровка (ax=)

```python
ev.plot_calibration(splits=['valid', 'test'], n_bins=10)   # reliability diagram
```

### Матрица ошибок (ax=)

```python
ev.plot_confusion_matrix(split='test')                    # сырые счётчики
ev.plot_confusion_matrix(split='test', normalize='true')  # recall по классам
ev.plot_confusion_matrix(split='test', normalize='pred')  # precision по классам
ev.plot_confusion_matrix(split='test', threshold=0.35)    # нестандартный порог (binary)
```

### Бизнес-метрики (только binary, ax=)

```python
ev.plot_lift(splits=['valid', 'test'])         # lift curve
ev.plot_gains(splits=['valid', 'test'])        # cumulative gains
ev.plot_decile_bar(split='test')              # доля позитивов по децилям скора
```

### Анализ порога (только binary, ax=)

```python
ev.plot_threshold_scan(split='valid')
ev.plot_threshold_scan(split='valid', metrics=['precision', 'recall', 'f1', 'specificity'])
ev.plot_ks(split='test')                      # CDF позитивов vs негативов + KS
```

### Сдвиг распределения (только binary, axes=)

```python
ev.plot_psi(ref='valid', target='test', n_bins=10)
```

### Только multiclass

```python
ev.plot_roc_ovr(splits=['test'])              # OvR ROC-кривые (отдельный figure на сплит)
ev.plot_metrics_per_class(split='test')       # bar: precision/recall/F1 по классам (ax=)
```

---

## RegressionEvaluator

```python
ev = RegressionEvaluator()
ev.add('train', y_true_train, y_pred_train)
ev.add('valid', y_true_valid, y_pred_valid)
ev.add('test',  y_true_test,  y_pred_test)
```

`y_pred` — 1D array предсказанных значений.

### Пресеты

| Имя | Описание |
|-----|----------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `mape` | Mean Absolute Percentage Error (если `y_true=0` → знаменатель = 1) |
| `smape` | Symmetric MAPE |
| `r2` | Коэффициент детерминации R² |
| `medae` | Median Absolute Error |
| `max_error` | Максимальная абсолютная ошибка |

По умолчанию (`add_default_metrics`): `mae, rmse, r2, mape, medae`.

### Визуализации

Методы с несколькими панелями принимают `axes=`, с одной — `ax=`:

```python
ev.plot_actual_vs_predicted(splits=['valid', 'test'])    # scatter + диагональ (axes=)
ev.plot_residuals_distribution(splits=['valid', 'test']) # гистограмма остатков (axes=)
ev.plot_residuals_vs_predicted(splits=['valid', 'test']) # residuals vs predicted (axes=)
ev.plot_error_percentile(splits=['valid', 'test'])       # отсортированные |ошибки| (ax=)

# MAE по бинам реального значения — видно, где модель систематически хуже
ev.plot_prediction_error_bins(split='test', n_bins=10)   # (ax=)
```

Пример компоновки:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ev.plot_actual_vs_predicted(splits=['valid', 'test'], axes=axes)
plt.tight_layout()
plt.savefig('regression.png')
```

---

## HTML-отчёт

Генерирует самодостаточный HTML с таблицей метрик и всеми графиками.

```python
ev.report('cls_report.html')   # ClassificationEvaluator
ev.report('reg_report.html')   # RegressionEvaluator
```

Все изображения встроены как base64 — файл открывается без доступа к сети.

---

## Фабрики метрик

| Фабрика | Возвращает |
|---------|-----------|
| `precision_at_k(k)` | Precision в топ-k предсказаниях |
| `recall_at_k(k)` | Recall в топ-k предсказаниях |
| `lift_at_k(k)` | Lift = precision@k / base\_rate |
| `f1_at_threshold(t)` | F1 при фиксированном пороге t |

`k` — число объектов (`int`) или доля выборки (`float ∈ (0, 1]`).

```python
# Одновременно несколько значений k
ev.add_metric(precision_at_k(0.05), name='precision@5%')
ev.add_metric(precision_at_k(0.10), name='precision@10%')
ev.add_metric(precision_at_k(0.20), name='precision@20%')
ev.add_metric(lift_at_k(0.10),      name='lift@10%')
ev.add_metric(f1_at_threshold(0.3), name='f1@t=0.3')
ev.add_metric(f1_at_threshold(0.5), name='f1@t=0.5')
```

---

## Пользовательские метрики

Любой callable с сигнатурой `(y_true: np.ndarray, y_second: np.ndarray) → float`:

```python
# Регрессия — p90 абсолютной ошибки
ev.add_metric(
    lambda yt, yp: float(np.percentile(np.abs(yt - yp), 90)),
    name='p90_abs_error',
)

# Классификация — бизнес-метрика с матрицей выплат
def profit_metric(y_true, y_proba):
    pred = (y_proba >= 0.4).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return float(tp * 500 - fp * 100)

ev.add_metric(profit_metric, name='profit@t=0.4')
```

---

## Сравнение нескольких моделей

```python
from ml_toolkit.model_evaluation import compare_models, plot_model_comparison, plot_model_heatmap, plot_model_delta

evaluators = {
    'LightGBM':  ev_lgb,
    'CatBoost':  ev_cat,
    'XGBoost':   ev_xgb,
}

# DataFrame: строки = метрики, столбцы = модели
df = compare_models(evaluators, split='valid')

# Фасетный bar chart
plot_model_comparison(evaluators, split='valid')

# Тепловая карта
plot_model_heatmap(evaluators, split='valid')

# Дельта vs базовой модели
plot_model_delta(evaluators, ref='LightGBM', split='valid')
```

---

## Bootstrap доверительные интервалы

```python
# DataFrame: строки = метрики, столбцы = [mean, std, ci_low, ci_high]
df = ev.bootstrap_metrics(split='valid', n_iter=1000, ci=0.95, seed=42)

# Визуализация CI (горизонтальные полосы)
ev.plot_bootstrap_ci(split='valid', n_iter=1000, ci=0.95, seed=42)

# Гистограммы распределений по метрикам
ev.plot_bootstrap_distributions(split='valid', n_iter=1000, ci=0.95, seed=42)
```

---

## Импорт

```python
# Из нового пакета напрямую
from ml_toolkit.model_evaluation import (
    ClassificationEvaluator,
    RegressionEvaluator,
    precision_at_k, recall_at_k, lift_at_k, f1_at_threshold,
    CLASSIFICATION_PRESETS, REGRESSION_PRESETS,
    compare_models, plot_model_comparison, plot_model_heatmap, plot_model_delta,
)

# Или из ml_toolkit.models (реэкспортируется, обратная совместимость)
from ml_toolkit.models import ClassificationEvaluator, RegressionEvaluator
```

`ModelEvaluator` — alias для `ClassificationEvaluator`, сохранён для обратной совместимости.
