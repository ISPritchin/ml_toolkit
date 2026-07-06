# Пресеты мультиклассовой классификации с длинным хвостом (`multiclass_imbalance/`)

3 пресета для мультиклассовой классификации (>= 3 классов) с сильно
неравномерным распределением классов ("длинный хвост") — в отличие от
`high_pr_auc/`, который нацелен на бинарный дисбаланс (доля позитивов < 5%).

---

## Общий API

Как и `high_pr_auc/`, все пресеты наследуют `BasePreset(BaseModel)`, но
`predict()` возвращает argmax-класс (в исходной кодировке меток), а не
бинарную метку по порогу — порог здесь неприменим:

```python
from ml_toolkit.presets.classification.multiclass_imbalance import EqualizationLossClassifier

model = EqualizationLossClassifier()
model.fit(X_train, y_train, X_valid, y_valid, selected_features=[...], cat_features=[...])

proba = model.predict_proba(X_test)   # np.ndarray (n_samples, n_classes)
labels = model.predict(X_test)        # np.ndarray, исходные метки y (без порога)
```

После `fit()` доступны те же атрибуты, что и в `high_pr_auc/`
(`selected_features_`, `cat_features_`, `train_pred_`, `valid_pred_`,
`best_params_`), плюс `n_classes_`.

Все три лосса реализованы через `calc_ders_multi` (CatBoost вызывает его по
одному разу на объект, не на батч — построчный Python-цикл внутри CatBoost,
неизбежно медленнее, чем `calc_ders_range` бинарных лоссов). Метрика для
early stopping/Optuna — `TotalF1:average=Macro` (не accuracy: доминируется
головным классом, не отражает качество на редких).

---

## Быстрый выбор

```
Мультикласс (>= 3 классов) с длинным хвостом
│
├─ Известные и стабильные частоты классов → BalancedSoftmaxClassifier
├─ Частоты нестабильны/эволюционируют, нужна пост-хок гибкость → см. LogitAdjustmentClassifier (005)
├─ Модель переуверена именно из-за длинного хвоста → LogitNormLossClassifier
└─ Головные классы подавляют градиент от редких → EqualizationLossClassifier
```

---

## EqualizationLossClassifier

**Файл:** `equalization_loss.py` (лосс — `ml_toolkit/losses/_equalization.py`)

Seesaw Loss (Wang et al., 2021) + EQLv2-style EMA-сглаживание (Tan et al.,
2021): подавляет вклад частых негативных классов в softmax CE через
mitigation (по статической частоте) и compensation (по предсказанной
уверенности, сглаженной EMA) множители.

**Параметры:** `lambda_=0.9` (EMA-момент), `seesaw_p=0.8` (mitigation),
`seesaw_q=2.0` (compensation).

## BalancedSoftmaxClassifier

**Файл:** `balanced_softmax.py` (лосс — `ml_toolkit/losses/_balanced_softmax.py`)

Training-time аналог logit adjustment (005): сдвиг логитов на
`tau*log(class_prior)` встроен в сам CE.

**Параметры:** `tau=1.0`.

## LogitNormLossClassifier

**Файл:** `logitnorm_loss.py` (лосс — `ml_toolkit/losses/_logitnorm.py`)

Нормализация логитов их L2-нормой (масштабированной `temperature`) перед CE
против overconfidence на голове длиннохвостого мультикласса.

**Параметры:** `temperature=0.04`.
