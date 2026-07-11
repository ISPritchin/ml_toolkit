from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    from optuna.pruners import BasePruner
    import xgboost as xgb

# ── Именованные метрики ────────────────────────────────────────────────────────

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom))

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))

def _f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0))


# (fn, direction) — 'minimize' или 'maximize'
REG_METRICS: dict[str, tuple[Callable, str]] = {
    'mae':   (_mae,   'minimize'),
    'rmse':  (_rmse,  'minimize'),
    'mape':  (_mape,  'minimize'),
    'smape': (_smape, 'minimize'),
}

CLS_METRICS: dict[str, tuple[Callable, str]] = {
    'pr_auc':  (average_precision_score, 'maximize'),
    'roc_auc': (roc_auc_score,           'maximize'),
    'f1':      (_f1,                     'maximize'),
}


# ── Параметризованные метрики ─────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, *, k: float) -> float:
    """Precision@k: доля позитивных среди топ-k предсказаний по убыванию score.

    Args:
        y_true: Бинарные метки (0/1).
        y_score: Предсказанные вероятности или скоры.
        k: Число объектов (int) или доля выборки (float ∈ (0, 1]).

    Returns:
        Доля позитивных в топ-k, от 0.0 до 1.0.

    """
    n = len(y_true)
    cut = max(1, int(n * k)) if isinstance(k, float) and k <= 1.0 else min(int(k), n)
    top_idx = np.argsort(y_score)[::-1][:cut]
    return float(np.asarray(y_true)[top_idx].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, *, k: float) -> float:
    """Recall@k: доля найденных позитивов среди топ-k предсказаний.

    Args:
        y_true: Бинарные метки (0/1).
        y_score: Предсказанные вероятности или скоры.
        k: Число объектов (int) или доля выборки (float ∈ (0, 1]).

    Returns:
        Recall в топ-k; 0.0 если нет позитивных примеров.

    """
    n = len(y_true)
    cut = max(1, int(n * k)) if isinstance(k, float) and k <= 1.0 else min(int(k), n)
    top_idx = np.argsort(y_score)[::-1][:cut]
    y = np.asarray(y_true)
    total_pos = y.sum()
    return float(y[top_idx].sum() / total_pos) if total_pos > 0 else 0.0


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, *, q: float) -> float:
    """Pinball loss для квантиля q ∈ (0, 1): асимметричный MAE, оптимален при минимизации.

    При q=0.5 эквивалентен MAE (с множителем 0.5).
    При q > 0.5 штрафует недооценку сильнее; при q < 0.5 — переоценку.

    Args:
        y_true: Истинные значения.
        y_pred: Предсказанные значения.
        q: Квантиль, float ∈ (0, 1).

    Returns:
        Среднее значение Pinball loss.

    """
    if not 0.0 < q < 1.0:
        raise ValueError(f'q должен быть в (0, 1), получено {q}')
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))


# ── Фабричные функции ─────────────────────────────────────────────────────────

def make_precision_at_k(k: float) -> tuple[Callable, str]:
    """Возвращает (fn, 'maximize') для использования в model_settings['cls_metric'].

    Args:
        k: Число объектов (int) или доля выборки (float ∈ (0, 1]).

    Example::

        model_settings = {'name': 'catboost', 'cls_metric': make_precision_at_k(100)}
        model_settings = {'name': 'catboost', 'cls_metric': make_precision_at_k(0.05)}

    """
    import functools
    return (functools.partial(precision_at_k, k=k), 'maximize')


def make_recall_at_k(k: float) -> tuple[Callable, str]:
    """Возвращает (fn, 'maximize') для использования в model_settings['cls_metric'].

    Args:
        k: Число объектов (int) или доля выборки (float ∈ (0, 1]).

    """
    import functools
    return (functools.partial(recall_at_k, k=k), 'maximize')


def make_quantile_loss(q: float) -> tuple[Callable, str]:
    """Возвращает (fn, 'minimize') для использования в model_settings['reg_metric'].

    Args:
        q: Квантиль ∈ (0, 1). q=0.5 близко к MAE; q=0.75 штрафует недооценку.

    Example::

        model_settings = {'name': 'catboost', 'reg_metric': make_quantile_loss(0.75)}

    """
    import functools
    return (functools.partial(quantile_loss, q=q), 'minimize')


def build_cat_encoder(
    X_train: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    model_settings: dict[str, Any],
) -> tuple[TransformerMixin | None, list[str], list[str], list[str]]:
    """Обучает категориальный энкодер на X_train и возвращает его для последующего apply.

    Args:
        X_train: Обучающая выборка (используется только для fit энкодера).
        selected_features: Список признаков; обновляется при OHE.
        cat_features: Список категориальных признаков.
        model_settings: Словарь настроек с ключом 'cat_encoder'.

    Returns:
        Кортеж (encoder, cat_in_sel, new_col_names, sel_updated).
        ``encoder`` равен None если категориальных признаков нет.

    """
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    cat_in_sel = [c for c in cat_features if c in selected_features and c in X_train.columns]
    if not cat_in_sel:
        return None, [], list(selected_features), list(selected_features)

    spec = model_settings.get('cat_encoder')
    if spec is None or spec == 'ordinal':
        encoder: Any = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1.0,
        )
    elif spec == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    elif hasattr(spec, 'fit'):
        encoder = spec
    else:
        raise TypeError(
            f"cat_encoder: ожидается None, 'ordinal', 'onehot' или sklearn-трансформер. "
            f"Получено: {type(spec).__name__!r}"
        )

    def _to_str(df: pd.DataFrame) -> pd.DataFrame:
        return df[cat_in_sel].astype(str).fillna('__NaN__')

    encoder.fit(_to_str(X_train))

    sample_out = encoder.transform(_to_str(X_train.iloc[:1]))
    expands = sample_out.shape[1] != len(cat_in_sel)
    new_col_names: list[str] = (
        list(encoder.get_feature_names_out(cat_in_sel)) if expands else list(cat_in_sel)
    )

    if expands:
        cat_set = set(cat_in_sel)
        sel_updated = [c for c in selected_features if c not in cat_set] + new_col_names
    else:
        sel_updated = list(selected_features)

    return encoder, cat_in_sel, new_col_names, sel_updated


def apply_cat_encoder(
    X: pd.DataFrame,
    encoder: TransformerMixin | None,
    cat_in_sel: list[str],
    new_col_names: list[str],
) -> pd.DataFrame:
    """Применяет ранее обученный категориальный энкодер к DataFrame.

    Args:
        X: Входной DataFrame.
        encoder: Обученный энкодер (из ``build_cat_encoder``). None → возвращает X без изменений.
        cat_in_sel: Список категориальных столбцов, которые были закодированы.
        new_col_names: Имена выходных столбцов после кодирования.

    Returns:
        DataFrame с закодированными категориальными признаками.

    """
    if encoder is None or not cat_in_sel:
        return X

    def _to_str(df: pd.DataFrame) -> pd.DataFrame:
        return df[cat_in_sel].astype(str).fillna('__NaN__')

    enc_arr = encoder.transform(_to_str(X))
    enc_df = pd.DataFrame(enc_arr, columns=new_col_names, index=X.index)
    return pd.concat([X.drop(columns=cat_in_sel), enc_df], axis=1)


def encode_cat_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_inference: pd.DataFrame,
    selected_features: list[str],
    cat_features: list[str],
    model_settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Кодирует категориальные признаки для адаптеров без нативной поддержки.

    Кодируются только те категориальные признаки, которые входят в ``selected_features``
    и присутствуют в DataFrame. Энкодер обучается на ``X_train``, применяется ко всем трём
    выборкам.

    ``model_settings['cat_encoder']``:
        ``None`` / ``'ordinal'``    → OrdinalEncoder (по умолч.; безопасен для деревьев)
        ``'onehot'``                → OneHotEncoder; имена вида ``'col__value'``
        sklearn-трансформер        → fit на X_train, transform на остальных

    Args:
        X_train: Обучающая выборка.
        X_valid: Валидационная выборка.
        X_inference: Инференс-выборка.
        selected_features: Список признаков; обновляется при расширяющем кодировании (OHE).
        cat_features: Список категориальных признаков.
        model_settings: Словарь настроек с ключом ``'cat_encoder'``.

    Returns:
        Кортеж (X_train_enc, X_valid_enc, X_inference_enc, selected_features_updated).
        При OrdinalEncoder ``selected_features_updated`` идентичен входному (имена не меняются).

    Raises:
        TypeError: Если тип ``cat_encoder`` не поддерживается.

    Example::

        X_tr, X_va, X_in, feats = encode_cat_features(
            X_train, X_valid, X_inference, selected_features, cat_features,
            model_settings={'cat_encoder': 'onehot'},
        )

    """
    encoder, cat_in_sel, new_col_names, sel_updated = build_cat_encoder(
        X_train, selected_features, cat_features, model_settings,
    )
    if encoder is None:
        return X_train, X_valid, X_inference, selected_features
    return (
        apply_cat_encoder(X_train, encoder, cat_in_sel, new_col_names),
        apply_cat_encoder(X_valid, encoder, cat_in_sel, new_col_names),
        apply_cat_encoder(X_inference, encoder, cat_in_sel, new_col_names),
        sel_updated,
    )


def set_optuna_verbosity(model_settings: dict[str, Any]) -> int:
    """Форсирует WARNING-уровень логов Optuna для текущего тюнинга, если не запрошено иначе.

    model_settings['optuna_verbose']: bool = False — при True не трогает
    текущий уровень логирования Optuna (даёт увидеть прогресс триалов снаружи).
    Раньше verbosity форсировался безусловно на уровне модуля (при импорте
    ml_toolkit.models._*), без возможности отключить и вне привязки к
    конкретному вызову fit() — вызывайте это в начале fit(), а не полагайтесь
    на импорт модуля.

    Возвращает предыдущий уровень verbosity — `optuna.logging.set_verbosity()`
    меняет глобальное состояние процесса, а не что-то по месту вызова, поэтому
    вызывающий fit() обязан восстановить его в конце тем же значением (иначе
    приглушение "утекает" во все последующие Optuna-вызовы в этом процессе,
    даже с optuna_verbose=True).
    """
    import optuna
    prev = optuna.logging.get_verbosity()
    if not model_settings.get('optuna_verbose', False):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    return prev


def resolve_timeout(model_settings: dict[str, Any]) -> float | None:
    """Возвращает model_settings['optuna_timeout'] (секунды) для study.optimize(timeout=...).

    None (по умолчанию) — без ограничения по времени, тюнинг идёт n_optuna_trials trials.
    При заданном timeout Optuna останавливается по первому из условий: n_trials или timeout
    (текущий trial всегда доучивается до конца — обрезки посреди trial не бывает).
    """
    timeout = model_settings.get('optuna_timeout')
    return float(timeout) if timeout is not None else None


def resolve_pruner(model_settings: dict[str, Any]) -> BasePruner:
    """Резолвит optuna.pruners.BasePruner из model_settings['optuna_pruner'].

    None (по умолчанию) → MedianPruner() — тот же дефолт, который Optuna сама подставляет
    при create_study(pruner=None). str → 'median' / 'hyperband' / 'percentile' /
    'successive_halving' / 'none' (алиас NopPruner, отключает прунинг). Готовый экземпляр
    optuna.pruners.BasePruner передаётся как есть.

    Прунер реально отсекает бесперспективные trials только там, где objective вызывает
    trial.report()/should_prune() на каждой итерации обучения — в этом пакете это
    XGBoost/LightGBM/CatBoost (+ их ranker-варианты, см. make_*_pruning_callback ниже) и TabM
    (по эпохам). Для моделей без промежуточных отчётов (sklearn-адаптеры без staged-обучения:
    RandomForest, decision_tree, MARS, EBM, GAM, RuleFit и т.п.) прунер не подключается вовсе.
    """
    import optuna
    spec = model_settings.get('optuna_pruner')
    if spec is None:
        return optuna.pruners.MedianPruner()
    if isinstance(spec, optuna.pruners.BasePruner):
        return spec
    if isinstance(spec, str):
        named: dict[str, Callable[[], Any]] = {
            'median': optuna.pruners.MedianPruner,
            'hyperband': optuna.pruners.HyperbandPruner,
            'percentile': lambda: optuna.pruners.PercentilePruner(25.0),
            'successive_halving': optuna.pruners.SuccessiveHalvingPruner,
            'none': optuna.pruners.NopPruner,
        }
        if spec not in named:
            raise ValueError(
                f'Неизвестный optuna_pruner={spec!r}. Доступные: {sorted(named)}. '
                f'Для произвольного прунера передайте экземпляр optuna.pruners.BasePruner.'
            )
        return named[spec]()
    raise TypeError(
        f'optuna_pruner должен быть None, str или optuna.pruners.BasePruner, '
        f'получено {type(spec).__name__!r}'
    )


def make_xgb_pruning_callback(trial: optuna.Trial) -> xgb.callback.TrainingCallback:
    """XGBoost TrainingCallback: trial.report()/should_prune() на каждой итерации бустинга.

    Ожидает ровно один eval_set (как во всех Optuna-objective этого пакета) — метрика
    берётся по первому найденному ключу в evals_log, без завязки на конкретное имя
    eval_metric. xgboost.callback.TrainingCallback — чистый Python, поэтому optuna.TrialPruned
    доходит до study.optimize без оборачивания сторонним исключением.
    """
    import optuna
    import xgboost as xgb

    class _XGBPruningCallback(xgb.callback.TrainingCallback):
        def after_iteration(
            self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict[str, list[float]]],
        ) -> bool:
            valid_log = next(iter(evals_log.values()))
            value = next(iter(valid_log.values()))[-1]
            trial.report(value, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f'Trial pruned at iteration {epoch}.')
            return False

    return _XGBPruningCallback()


def make_lgb_pruning_callback(trial: optuna.Trial) -> Callable[[lgb.callback.CallbackEnv], None]:
    """LightGBM callback: тот же report()/should_prune(), что и make_xgb_pruning_callback.

    Берёт первую строку env.evaluation_result_list (в этом пакете objective всегда передаёт
    ровно один eval_set с одной метрикой).
    """
    import optuna

    def _callback(env: lgb.callback.CallbackEnv) -> None:
        _, _, value, _ = env.evaluation_result_list[0]
        trial.report(value, step=env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned(f'Trial pruned at iteration {env.iteration}.')

    _callback.order = 30
    return _callback


def make_catboost_pruning_callback(trial: optuna.Trial) -> Any:  # noqa: ANN401 - класс объявлен локально внутри функции
    """CatBoost callback: см. make_xgb_pruning_callback.

    CatBoost оборачивает любое исключение, поднятое внутри after_iteration, в
    catboost.CatBoostError (Cython-граница) — optuna.TrialPruned
    там не долетел бы до study.optimize нераспознанным. Вместо этого after_iteration возвращает
    False (штатная остановка обучения по колбэку) и взводит self.pruned; вызывающий objective
    обязан проверить callback.pruned сразу после m.fit(...) и поднять optuna.TrialPruned вручную.
    """
    class _CatBoostPruningCallback:
        def __init__(self) -> None:
            self.pruned = False

        def after_iteration(self, info: Any) -> bool:  # noqa: ANN401 - CatBoost не экспортирует публичный тип info
            metrics = info.metrics.get('validation', next(iter(info.metrics.values())))
            value = next(iter(metrics.values()))[-1]
            trial.report(value, step=info.iteration)
            if trial.should_prune():
                self.pruned = True
                return False
            return True

    return _CatBoostPruningCallback()


def resolve_metric_fn(
    model_settings: dict[str, Any],
    key: str,
    default_fn: Callable,
    default_direction: str,
    named: dict[str, tuple[Callable, str]],
) -> tuple[Callable, str]:
    """Возвращает (metric_fn, direction) для Optuna из model_settings[key].

    Варианты значения model_settings[key]:
    - ``None`` / отсутствует → возвращает (default_fn, default_direction).
    - ``str`` → ищет в словаре named (например, 'mae', 'rmse').
    - ``callable`` → direction берётся из model_settings[key + '_direction']
      или из default_direction.
    - ``(callable, str)`` → кортеж (fn, direction).

    Args:
        model_settings: Словарь настроек адаптера.
        key: Ключ в model_settings ('reg_metric' или 'cls_metric').
        default_fn: Метрика по умолчанию.
        default_direction: Направление оптимизации по умолчанию ('minimize'/'maximize').
        named: Словарь именованных пресетов {name: (fn, direction)}.

    Returns:
        Кортеж (fn, direction).

    Raises:
        ValueError: Если строка не найдена в named.
        TypeError: Если тип значения не поддерживается.

    Example::

        # строка
        fn, dir = resolve_metric_fn(ms, 'reg_metric', _mae, 'minimize', REG_METRICS)
        # callable
        ms = {'reg_metric': lambda yt, yp: custom(yt, yp)}
        # кортеж (fn, direction)
        ms = {'reg_metric': (lambda yt, yp: r2_score(yt, yp), 'maximize')}

    """
    spec = model_settings.get(key)
    if spec is None:
        return default_fn, default_direction
    if isinstance(spec, str):
        if spec not in named:
            raise ValueError(
                f'Неизвестный пресет {key}={spec!r}. '
                f'Доступные: {sorted(named)}. '
                f'Для произвольной метрики передайте callable.'
            )
        return named[spec]
    if isinstance(spec, tuple) and len(spec) == 2 and callable(spec[0]):
        fn, direction = spec
        if direction not in ('minimize', 'maximize'):
            raise ValueError(f'{key}: direction должен быть "minimize" или "maximize", получено {direction!r}')
        return fn, direction
    if callable(spec):
        direction = model_settings.get(f'{key}_direction', default_direction)
        return spec, direction
    raise TypeError(
        f'{key} должен быть str, callable или (callable, direction). Получено: {type(spec).__name__!r}'
    )


def prep_cat_features(X: pd.DataFrame, features: list[str], cat_features: list[str]) -> pd.DataFrame:
    """Слайсит X по features и конвертирует категориальные столбцы в dtype 'category'."""
    df = X[features].copy()
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def fit_rank_reference(scores: np.ndarray) -> np.ndarray:
    """Строит референс для rank-нормализации: отсортированные скоры train.

    Args:
        scores: Сырые скоры модели на обучающей выборке (1D).

    Returns:
        Отсортированный по возрастанию float64-массив — референсное распределение.

    """
    return np.sort(np.asarray(scores, dtype=np.float64))


def rank_transform(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Отображает скоры в [0, 1] по позиции в референсном распределении.

    В отличие от ранжирования внутри батча, результат для объекта не зависит
    от того, с какими другими объектами он попал в predict — скор определяется
    только референсом (обычно train-скорами той же модели). Используется для
    воспроизводимого predict_proba у rank-ансамблей.

    Args:
        scores: Сырые скоры (1D), любые значения.
        reference: Отсортированный референс из fit_rank_reference.

    Returns:
        Массив float64 в [0, 1]: доля референсных скоров, не превышающих данный.

    """
    reference = np.asarray(reference, dtype=np.float64)
    if len(reference) == 0:
        return np.full(len(scores), 0.5)
    pos = np.searchsorted(reference, np.asarray(scores, dtype=np.float64), side='right')
    return pos / len(reference)


def fit_calibrator(val_proba: np.ndarray, y_valid: np.ndarray) -> IsotonicRegression:
    """Обучает изотоническую регрессию на вероятностях валидационной выборки.

    Args:
        val_proba: Сырые вероятности модели на валидационной выборке (1D, бинарный класс).
        y_valid: Истинные бинарные метки валидационной выборки.

    Returns:
        Обученный калибратор с методом predict(proba) -> np.ndarray.

    """
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_proba, y_valid)
    return calibrator


def fit_multiclass_calibrators(
    val_proba: np.ndarray, y_valid: np.ndarray
) -> list[IsotonicRegression]:
    """Обучает K изотонических калибраторов по схеме OvR (один калибратор на класс).

    Предполагает, что столбец k матрицы val_proba соответствует k-му значению из
    sorted(unique(y_valid)) — стандартный порядок CatBoost/sklearn при целочисленных метках.

    Args:
        val_proba: Матрица вероятностей (n_samples, K) с валидационной выборки.
        y_valid: Истинные метки классов (целочисленные или ordinal-кодированные).

    Returns:
        Список из K обученных IsotonicRegression, по одному на класс.

    """
    classes = np.unique(y_valid)
    return [
        IsotonicRegression(out_of_bounds='clip').fit(val_proba[:, k], (y_valid == cls).astype(int))
        for k, cls in enumerate(classes)
    ]


def apply_multiclass_calibrators(
    proba: np.ndarray, calibrators: list[IsotonicRegression]
) -> np.ndarray:
    """Применяет поклассовые калибраторы и нормирует строки к сумме 1.

    Args:
        proba: Матрица сырых вероятностей (n_samples, K).
        calibrators: Список из K IsotonicRegression, обученных fit_multiclass_calibrators.

    Returns:
        Откалиброванная матрица (n_samples, K), строки нормированы к сумме 1.

    """
    calibrated = np.column_stack([cal.predict(proba[:, k]) for k, cal in enumerate(calibrators)])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.where(row_sums == 0, 1.0, row_sums)


def calibrate_proba(val_proba: np.ndarray, y_valid: np.ndarray, infer_proba: np.ndarray) -> np.ndarray:
    """Калибрует вероятности инференса изотонической регрессией, обученной на валидации.

    Args:
        val_proba: Сырые вероятности модели на валидационной выборке.
        y_valid: Истинные бинарные метки валидационной выборки.
        infer_proba: Сырые вероятности модели на инференс-выборке.

    Returns:
        Откалиброванные вероятности для инференс-выборки, зажатые в [0, 1].

    """
    return fit_calibrator(val_proba, y_valid).predict(infer_proba)
