# ml_toolkit/feature_selection/genetic.py

from collections.abc import Callable
from dataclasses import dataclass
import difflib
import logging
import random
from typing import Any
import uuid

from deap import algorithms, base, creator, tools
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Scorer: обучает модель на выбранных признаках, возвращает float для минимизации.
ScorerFn = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float]

# CLASSIFICATION_PRESETS (ml_toolkit.model_evaluation) уже умеет и binary (y_proba
# 1D — вероятность положительного класса), и multiclass (y_proba 2D (n, K)) —
# сам решает по y_proba.ndim, какую формулу применить (macro для f1/precision/
# recall/roc_auc/pr_auc, OvR-среднее для brier/ece, argmax + нативный sklearn
# для accuracy/balanced_accuracy/mcc/cohen_kappa/gini). genetic-модуль эти формулы
# не дублирует — только решает, какие из них "выше — лучше" (нужно инвертировать
# знак для минимизации) и добавляет обратную совместимость по именам.
_LOWER_IS_BETTER_CLS_METRICS = frozenset({'log_loss', 'brier', 'ece'})
_CLS_METRIC_ALIASES: dict[str, str] = {'logloss': 'log_loss'}  # старое имя в genetic API


def make_catboost_scorer(
    task: str,
    metric: str | Callable[[np.ndarray, np.ndarray], float],
    model_params: dict[str, Any],
    cat_features: list[str] | None = None,
    baseline_train: 'pd.Series | np.ndarray | None' = None,
    baseline_valid: 'pd.Series | np.ndarray | None' = None,
    postprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ScorerFn:
    """Фабрика CatBoost-скорера для select_features_genetic.

    Создаёт замыкание, которое обучает CatBoost на переданных признаках и
    возвращает значение для минимизации. Все CatBoost-специфичные параметры
    (baseline, cat_features) фиксируются здесь; GA передаёт только срезанные
    X_train / X_valid.

    ``task='classification'`` работает и для бинарной, и для multiclass задачи —
    режим определяется автоматически по числу уникальных меток в ``y_train``
    (``len(np.unique(y_train)) > 2``), тем же способом, что и адаптеры в
    ``ml_toolkit.models`` (``CatBoostClassifier``/``LightGBMClassifier``/
    ``XGBoostClassifier``). Если в ``model_params`` не задан явный
    ``loss_function``, для multiclass он проставляется в ``'MultiClass'``
    автоматически (как и в этих адаптерах) — явный ``loss_function`` в
    ``model_params`` не переопределяется.

    Args:
        task: ``'classification'`` (бинарная или multiclass — см. выше) или
            ``'regression'``.
        metric: Строка-метрика или ``callable(y_true, y_proba) -> float``
            (конвенция: значение для минимизации; для «выше — лучше» нужен минус).
            При multiclass ``y_proba`` в callable — полная матрица ``(n, n_classes)``
            от ``predict_proba``, при бинарной — как раньше, 1D-вектор вероятности
            положительного класса.

            Строки (классификация, единый реестр с
            ``ml_toolkit.model_evaluation.CLASSIFICATION_PRESETS`` — при multiclass
            каждая метрика сама переключается на macro/OvR-агрегацию, подробности
            там же): ``'pr_auc'``, ``'roc_auc'``, ``'f1'``, ``'precision'``,
            ``'recall'``, ``'balanced_accuracy'``, ``'accuracy'``, ``'mcc'``,
            ``'cohen_kappa'``, ``'gini'``, ``'log_loss'`` (алиас — старое имя
            ``'logloss'``), ``'brier'``, ``'ece'``. ``'ks'`` тоже доступна, но
            определена только для бинарной классификации и поднимет
            ``ValueError`` при multiclass ``y``.

            Строки (регрессия): ``'mae'``, ``'rmse'``, ``'median_ae'``,
            ``'mape'``, ``'smape'``, ``'r2'``.
        model_params: Параметры CatBoostClassifier / CatBoostRegressor.
        cat_features: Категориальные признаки; автоматически фильтруются по
            колонкам переданного X_train.
        baseline_train: Предвычисленный бейзлайн для обучающей выборки
            (например ``X_train['fee_nds_amount']``). CatBoost использует его
            как смещение предиктов (residual learning).
        baseline_valid: Аналогично для валидации.
        postprocess_fn: ``callable(pred: np.ndarray) -> np.ndarray``, применяется
            к предиктам регрессора до расчёта метрики. Если нужны доп. колонки —
            замкните их снаружи.

    Returns:
        ``ScorerFn``: ``(X_train, y_train, X_valid, y_valid) -> float``.

    """
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

    from ml_toolkit.model_evaluation import CLASSIFICATION_PRESETS

    _cat_set = set(cat_features or [])

    def scorer(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        cat_sel = [f for f in X_train.columns if f in _cat_set]

        if task == 'classification':
            cb_params = dict(model_params)
            if len(np.unique(np.asarray(y_train))) > 2:
                cb_params.setdefault('loss_function', 'MultiClass')
            model = CatBoostClassifier(**cb_params)
        else:
            model = CatBoostRegressor(**model_params)

        train_pool = Pool(X_train, y_train, cat_features=cat_sel, baseline=baseline_train)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_sel, baseline=baseline_valid)
        model.fit(train_pool, eval_set=valid_pool)

        if task == 'classification':
            proba = model.predict_proba(valid_pool)
            # Бинарная задача: 1D-вектор P(y=1), как и раньше (для обратной
            # совместимости и потому что CLASSIFICATION_PRESETS сам различает
            # бинарный/multiclass случай по ndim, а не по числу столбцов proba).
            # Multiclass: полная матрица (n, n_classes) — presets делают
            # argmax/OvR/macro сами (см. ml_toolkit/model_evaluation/_classification.py).
            proba_for_metric = proba[:, 1] if proba.shape[1] == 2 else proba
            if callable(metric):
                return float(metric(y_valid.to_numpy(), proba_for_metric))
            metric_name = _CLS_METRIC_ALIASES.get(metric, metric)
            if metric_name not in CLASSIFICATION_PRESETS:
                raise ValueError(
                    f'Unsupported classification metric: {metric!r}. '
                    f'Available: {sorted(CLASSIFICATION_PRESETS)}'
                )
            value = CLASSIFICATION_PRESETS[metric_name](y_valid.to_numpy(), proba_for_metric)
            return value if metric_name in _LOWER_IS_BETTER_CLS_METRICS else -value

        pred = model.predict(valid_pool)
        pred_post = postprocess_fn(pred) if postprocess_fn is not None else pred
        if callable(metric):
            return float(metric(y_valid.to_numpy(), pred_post))
        y_np = y_valid.to_numpy()
        if metric == 'r2':
            return -r2_score(y_np, pred_post)
        if metric == 'mae':
            return float(np.abs(pred_post - y_np).mean())
        if metric == 'rmse':
            return float(np.sqrt(mean_squared_error(y_np, pred_post)))
        if metric == 'median_ae':
            return float(median_absolute_error(y_np, pred_post))
        if metric == 'mape':
            return float(np.mean(np.abs((y_np - pred_post) / (np.abs(y_np) + 1e-8))))
        if metric == 'smape':
            return float(np.mean(
                2 * np.abs(pred_post - y_np) / (np.abs(pred_post) + np.abs(y_np) + 1e-8)
            ))
        raise ValueError(f'Unsupported regression metric: {metric}')

    return scorer


def _compute_fitness(individual: list[int], ctx: dict[str, Any]) -> tuple[float]:
    """Оценивает особь: нарезает признаки, вызывает scorer, добавляет штраф.

    Args:
        individual: Бинарная хромосома (0/1 для каждого кандидата-признака).
        ctx: Словарь с ключами: X_train, y_train, X_valid, y_valid,
            feature_names, scorer, gen_params, upper_bound.

    Returns:
        Кортеж из одного float (фитнес, минимизируется).

    """
    feature_names: list[str] = ctx['feature_names']
    gen_params: dict[str, Any] = ctx['gen_params']
    upper_bound: int = ctx['upper_bound']
    scorer: ScorerFn = ctx['scorer']

    selected = [feature_names[i] for i, val in enumerate(individual) if val == 1]
    n_selected = len(selected)
    if n_selected == 0 or n_selected > upper_bound:
        raise ValueError(f'Invalid number of selected features: {n_selected}')

    X_train: pd.DataFrame = ctx['X_train']
    X_valid: pd.DataFrame = ctx['X_valid']
    score = scorer(X_train[selected], ctx['y_train'], X_valid[selected], ctx['y_valid'])

    penalty: float = gen_params.get('penalty_for_extra_feature', 0.0)
    free: int = gen_params.get('n_features_without_penalty', 0)
    extra = max(0, n_selected - free)
    if extra > 0 and penalty > 0.0:
        # score >= 0: увеличиваем (хуже при минимизации)
        # score < 0: уменьшаем по модулю (тоже хуже)
        if score >= 0:
            score *= 1 + extra * penalty
        else:
            score *= 1 - extra * penalty

    return (score,)


def generate_binary_value(start_prob: float) -> int:
    """Генерирует случайное бинарное значение с заданной вероятностью единицы."""
    return 1 if random.random() < start_prob else 0


def repair_individual(
    individual: list[int],
    upper_bound: int,
    must: frozenset[int] = frozenset(),
) -> list[int]:
    """Ремонтирует особь: восстанавливает must-признаки и контролирует число единиц.

    Args:
        individual: Список из 0 и 1.
        upper_bound: Максимально допустимое число единиц в хромосоме.
        must: Индексы обязательных признаков — они всегда остаются единицами.

    Returns:
        Модифицированная особь, удовлетворяющая ограничениям.

    """
    for idx in must:
        if 0 <= idx < len(individual):
            individual[idx] = 1

    ones = [i for i, val in enumerate(individual) if val == 1]

    if len(ones) == 0:
        idx = random.randrange(len(individual))
        individual[idx] = 1
        ones.append(idx)

    if len(ones) > upper_bound:
        removable = [i for i in ones if i not in must]
        n_to_remove = min(len(ones) - upper_bound, len(removable))
        for i in random.sample(removable, n_to_remove):
            individual[i] = 0

    return individual


def cx_uniform_with_repair(
    ind1: list[int],
    ind2: list[int],
    upper_bound: int,
    must: frozenset[int] = frozenset(),
    prob: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Uniform crossover с последующей починкой обеих особей."""
    for i in range(min(len(ind1), len(ind2))):
        if random.random() < prob:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    repair_individual(ind1, upper_bound, must)
    repair_individual(ind2, upper_bound, must)
    return ind1, ind2


def mut_flip_bit_with_repair(
    individual: list[int],
    upper_bound: int,
    indpb: float,
    must: frozenset[int] = frozenset(),
) -> tuple[list[int]]:
    """Мутация flip-bit с контролем числа выбранных признаков."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
    repair_individual(individual, upper_bound, must)
    return (individual,)


# Значения по умолчанию совпадают с тем, что раньше было зашито в код: без новых ключей
# в gen_params поведение (и результат при том же seed) не меняется.
DEFAULT_MUTATION_GENE_PROBABILITY = 0.05
DEFAULT_CROSSOVER_GENE_PROBABILITY = 0.5
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_ELITE_SIZE = 1

_REQUIRED_KEYS = frozenset({
    'max_features', 'population_size', 'n_generations', 'cross_probability', 'mutation_probability',
})
_INIT_KEYS = frozenset({'start_probability_to_include_feature', 'start_n_features'})
_OPTIONAL_KEYS = frozenset({
    'n_features_without_penalty', 'penalty_for_extra_feature',
    'min_improvement', 'min_improvement_in_percents', 'n_epoch_for_min_improvement',
    'must_be_included', 'seed',
    'mutation_gene_probability', 'mutation_flips', 'crossover_gene_probability',
    'tournament_size', 'elite_size',
})
KNOWN_GEN_PARAMS = _REQUIRED_KEYS | _INIT_KEYS | _OPTIONAL_KEYS


@dataclass(frozen=True)
class _OperatorParams:
    """Разобранные настройки операторов: значения по умолчанию + то, что задано в gen_params."""

    mutation_gene_probability: float
    crossover_gene_probability: float
    tournament_size: int
    elite_size: int


def _check_known_keys(gen_params: dict[str, Any]) -> None:
    """Ловит опечатки: неизвестный ключ раньше молча игнорировался (кроме обязательных, которые падали с KeyError)."""
    unknown = sorted(set(gen_params) - KNOWN_GEN_PARAMS)
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(KNOWN_GEN_PARAMS), n=1)
        hints.append(f"'{key}'" + (f" (возможно, '{close[0]}')" if close else ''))
    raise ValueError(
        f'Неизвестные ключи gen_params: {", ".join(hints)}. Допустимые: {sorted(KNOWN_GEN_PARAMS)}'
    )


def _resolve_operator_params(gen_params: dict[str, Any], n_features: int, population_size: int) -> _OperatorParams:
    """Разбирает и валидирует настройки операторов (мутация, скрещивание, турнир, элитизм)."""
    has_gene_prob = 'mutation_gene_probability' in gen_params
    has_flips = 'mutation_flips' in gen_params
    if has_gene_prob and has_flips:
        raise ValueError('Нельзя указывать одновременно mutation_gene_probability и mutation_flips')

    if has_flips:
        flips = gen_params['mutation_flips']
        if not flips > 0:
            raise ValueError(f'mutation_flips должен быть > 0, получено {flips!r}')
        # ожидаемое число флипов на одну мутирующую особь = indpb * N
        mutation_gene_probability = min(1.0, float(flips) / n_features)
    else:
        mutation_gene_probability = float(
            gen_params.get('mutation_gene_probability', DEFAULT_MUTATION_GENE_PROBABILITY)
        )
        if not 0.0 <= mutation_gene_probability <= 1.0:
            raise ValueError(f'mutation_gene_probability должен быть в [0, 1], получено {mutation_gene_probability!r}')

    crossover_gene_probability = float(
        gen_params.get('crossover_gene_probability', DEFAULT_CROSSOVER_GENE_PROBABILITY)
    )
    if not 0.0 <= crossover_gene_probability <= 1.0:
        raise ValueError(f'crossover_gene_probability должен быть в [0, 1], получено {crossover_gene_probability!r}')

    tournament_size = gen_params.get('tournament_size', DEFAULT_TOURNAMENT_SIZE)
    if int(tournament_size) != tournament_size or tournament_size < 1:
        raise ValueError(f'tournament_size должен быть целым >= 1, получено {tournament_size!r}')

    elite_size = gen_params.get('elite_size', DEFAULT_ELITE_SIZE)
    if int(elite_size) != elite_size or not 0 <= elite_size < population_size:
        raise ValueError(
            f'elite_size должен быть целым в [0, population_size), получено {elite_size!r} '
            f'при population_size={population_size}'
        )

    return _OperatorParams(
        mutation_gene_probability=mutation_gene_probability,
        crossover_gene_probability=crossover_gene_probability,
        tournament_size=int(tournament_size),
        elite_size=int(elite_size),
    )


def select_features_genetic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_names: list[str],
    scorer: ScorerFn,
    gen_params: dict[str, Any],
    generation_callback: Callable[[int, float, float, int], None] | None = None,
) -> list[str]:
    """Отбирает признаки с помощью генетического алгоритма (DEAP).

    Кодирует подмножество признаков как бинарную хромосому и оптимизирует
    значение ``scorer`` на валидационной выборке (один trial — один вызов
    ``scorer``). Поддерживает элитизм, кэширование фитнеса и early stopping.

    ``scorer`` — произвольный callable, принимающий нарезанные по выбранным
    признакам ``X_train`` и ``X_valid``. Используйте ``make_catboost_scorer``
    для стандартного CatBoost-варианта или любой другой callable (sklearn,
    LightGBM, XGBoost и т.д.):

    .. code-block:: python

        def lightgbm_scorer(X_tr, y_tr, X_va, y_va):
            model = LGBMClassifier().fit(X_tr, y_tr)
            return -roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

        select_features_genetic(..., scorer=lightgbm_scorer, ...)

    Args:
        X_train: Обучающая выборка (Pandas DataFrame).
        y_train: Целевая переменная обучающей выборки.
        X_valid: Валидационная выборка.
        y_valid: Целевая переменная валидационной выборки.
        feature_names: Полный список кандидатов в признаки.
        scorer: ``(X_train_sel, y_train, X_valid_sel, y_valid) -> float``.
            Получает DataFrame только с выбранными признаками. Должен
            возвращать значение для минимизации; для «выше — лучше» метрик
            нужно вернуть отрицательное значение.
        generation_callback: Вызывается в конце каждого поколения:
            ``callback(gen, best_score, mean_score, n_features_best)``.
            Удобно для сбора статистики и построения графиков эволюции.
            ``best_score`` и ``mean_score`` — сырые значения фитнеса
            (для метрик «выше — лучше» они отрицательны). ``None`` — отключено.
        gen_params: Словарь параметров алгоритма. Неизвестные ключи вызывают
            ``ValueError`` (с подсказкой при опечатке).

            Обязательные ключи: ``max_features``, ``population_size``, ``n_generations``,
            ``cross_probability``, ``mutation_probability``.

            Инициализация начальной популяции — ровно один из двух ключей:

            * ``start_probability_to_include_feature`` — вероятность включить признак;
            * ``start_n_features`` — фиксированное число случайно выбранных признаков.

            Штраф и early stopping:

            * ``n_features_without_penalty`` — признаки без штрафа;
            * ``penalty_for_extra_feature`` — штраф за каждый лишний признак;
            * ``min_improvement``, ``min_improvement_in_percents``,
              ``n_epoch_for_min_improvement`` — early stopping.

            Ограничения и воспроизводимость:

            * ``must_be_included`` (``list[int | str]``) — обязательные признаки: целые числа
              интерпретируются как индексы, строки — как имена;
            * ``seed`` (``int``) — фиксация random state.

            Операторы (значения по умолчанию совпадают с прежним зашитым поведением):

            * ``mutation_gene_probability`` (``float``, по умолчанию 0.05) — вероятность флипа
              каждого гена внутри мутирующей особи;
            * ``mutation_flips`` (``float > 0``) — то же через ожидаемое число флипов на мутацию:
              ``mutation_gene_probability = mutation_flips / N``. Взаимоисключающ с
              ``mutation_gene_probability``;
            * ``crossover_gene_probability`` (``float``, по умолчанию 0.5) — вероятность обмена
              гена между родителями при uniform crossover;
            * ``tournament_size`` (``int >= 1``, по умолчанию 3) — размер турнира;
            * ``elite_size`` (``int`` в ``[0, population_size)``, по умолчанию 1) — сколько лучших
              особей за всю историю переходит в следующее поколение (0 — без элитизма; лучшая
              особь всё равно возвращается в результате).

    Returns:
        Список имён признаков из глобально лучшей особи.

    Raises:
        ValueError: Если ``feature_names`` пуст, параметры некорректны или в
            ``gen_params`` есть неизвестные ключи.

    """
    if not feature_names:
        raise ValueError('feature_names must not be empty')
    _check_known_keys(gen_params)
    if gen_params['max_features'] < 1:
        raise ValueError('max_features must be at least 1')
    if gen_params['population_size'] < 2:
        raise ValueError('population_size must be at least 2')
    if gen_params['n_generations'] < 1:
        raise ValueError('ngen must be at least 1')
    ops = _resolve_operator_params(gen_params, n_features=len(feature_names), population_size=gen_params['population_size'])

    has_prob = 'start_probability_to_include_feature' in gen_params
    has_n = 'start_n_features' in gen_params
    if has_prob and has_n:
        raise ValueError(
            'Нельзя указывать одновременно start_probability_to_include_feature и start_n_features'
        )
    if not has_prob and not has_n:
        raise ValueError(
            'Необходимо указать start_probability_to_include_feature или start_n_features'
        )

    if 'seed' in gen_params:
        random.seed(gen_params['seed'])
        np.random.seed(gen_params['seed'])

    upper_bound = gen_params['max_features']
    n_free = gen_params.get('n_features_without_penalty', 0)
    if n_free > upper_bound:
        raise ValueError(
            f'n_features_without_penalty ({n_free}) не может превышать max_features ({upper_bound})'
        )

    _uid = uuid.uuid4().hex
    fitness_cls_name = f'FitnessMin_{_uid}'
    individual_cls_name = f'Individual_{_uid}'
    creator.create(fitness_cls_name, base.Fitness, weights=(-1.0,))
    creator.create(individual_cls_name, list, fitness=getattr(creator, fitness_cls_name))
    IndividualClass: type = getattr(creator, individual_cls_name)

    _raw_must = gen_params.get('must_be_included', [])
    must_indices: list[int] = []
    for v in _raw_must:
        if isinstance(v, str):
            if v not in feature_names:
                logger.warning("must_be_included: признак '%s' не найден в feature_names, пропуск", v)
                continue
            must_indices.append(feature_names.index(v))
        else:
            idx = int(v)
            if not (0 <= idx < len(feature_names)):
                raise ValueError(
                    f'must_be_included: индекс {idx} вне диапазона [0, {len(feature_names) - 1}]'
                )
            must_indices.append(idx)
    must: frozenset[int] = frozenset(must_indices)
    if len(must) > upper_bound:
        raise ValueError(
            f'must_be_included содержит {len(must)} признаков, но max_features={upper_bound}'
        )

    def create_valid_individual() -> list[int]:
        if has_n:
            n_start = min(int(gen_params['start_n_features']), len(feature_names))
            chosen = set(random.sample(range(len(feature_names)), n_start))
            ind = [1 if i in chosen else 0 for i in range(len(feature_names))]
        else:
            start_prob = gen_params['start_probability_to_include_feature']
            ind = [generate_binary_value(start_prob) for _ in range(len(feature_names))]
        for idx in must:
            if 0 <= idx < len(ind):
                ind[idx] = 1
        return repair_individual(ind, upper_bound, must)

    ctx: dict[str, Any] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'feature_names': feature_names,
        'scorer': scorer,
        'gen_params': gen_params,
        'upper_bound': upper_bound,
    }

    _cache: dict[tuple[int, ...], tuple[float]] = {}

    def _evaluate_cached(individual: list[int]) -> tuple[float]:
        key = tuple(individual)
        if key not in _cache:
            _cache[key] = _compute_fitness(individual, ctx)
        return _cache[key]

    hof = tools.HallOfFame(max(1, ops.elite_size))

    try:
        toolbox = base.Toolbox()
        toolbox.register('individual', tools.initIterate, IndividualClass, create_valid_individual)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('evaluate', _evaluate_cached)
        toolbox.register(
            'mate', cx_uniform_with_repair, upper_bound=upper_bound, must=must, prob=ops.crossover_gene_probability,
        )
        toolbox.register(
            'mutate', mut_flip_bit_with_repair, upper_bound=upper_bound, indpb=ops.mutation_gene_probability, must=must,
        )
        toolbox.register('select', tools.selTournament, tournsize=ops.tournament_size)

        pop = toolbox.population(n=gen_params['population_size'])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('min', np.min)
        stats.register('max', np.max)

        min_improvement: float = gen_params.get('min_improvement', 0)
        min_improvement_in_percents: float = gen_params.get('min_improvement_in_percents', 0)
        n_epoch_for_min_improvement: int = gen_params.get('n_epoch_for_min_improvement', 1)
        best_fitness_history: list[float] = []

        gen_bar = tqdm(
            range(1, gen_params['n_generations'] + 1),
            desc='Генетический отбор',
            unit='gen',
        )
        for gen in gen_bar:
            offspring = algorithms.varAnd(
                pop,
                toolbox,
                cxpb=gen_params['cross_probability'],
                mutpb=gen_params['mutation_probability'],
            )

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(tqdm(
                map(toolbox.evaluate, invalid_ind),
                total=len(invalid_ind),
                desc=f'  Оценка особей (gen {gen})',
                unit='ind',
                leave=False,
            ))
            for fit, ind in zip(fits, invalid_ind, strict=True):
                ind.fitness.values = fit

            hof.update(offspring)

            elites = []
            for best_ind in list(hof)[:ops.elite_size]:
                elite = IndividualClass(best_ind[:])
                elite.fitness.values = best_ind.fitness.values
                elites.append(elite)
            pop = elites + toolbox.select(offspring, k=gen_params['population_size'] - len(elites))

            best_fitness = hof[0].fitness.values[0]
            best_fitness_history.append(best_fitness)
            compiled = stats.compile(pop)
            gen_bar.set_postfix({'best': f'{best_fitness:.4f}', 'feats': sum(hof[0])})
            logger.debug('Generation %d: %s', gen, compiled)
            if generation_callback is not None:
                generation_callback(gen, best_fitness, float(compiled['avg']), int(sum(hof[0])))

            if (
                (min_improvement or min_improvement_in_percents)
                and len(best_fitness_history) >= 1 + n_epoch_for_min_improvement
            ):
                improvement = (
                    best_fitness_history[-(1 + n_epoch_for_min_improvement)]
                    - best_fitness_history[-1]
                )
                expected_improvement = (
                    min_improvement or abs(best_fitness_history[-(1 + n_epoch_for_min_improvement)])
                    / 100
                    * min_improvement_in_percents
                )
                if improvement >= expected_improvement:
                    logger.info(
                        'Genetic gen %d: улучшение %.4f >= %.4f за %d ep, продолжаем',
                        gen, improvement, expected_improvement, n_epoch_for_min_improvement,
                    )
                else:
                    logger.info(
                        'Genetic gen %d: улучшение %.4f < %.4f за %d ep, остановка',
                        gen, improvement, expected_improvement, n_epoch_for_min_improvement,
                    )
                    break

    finally:
        for cls_name in (fitness_cls_name, individual_cls_name):
            if hasattr(creator, cls_name):
                delattr(creator, cls_name)

    logger.info(
        '[Genetic] Best score: %.4f, selected features: %d',
        hof[0].fitness.values[0], sum(hof[0]),
    )
    return [feature_names[i] for i, val in enumerate(hof[0]) if val == 1]
