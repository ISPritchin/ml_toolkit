# ml_toolkit/feature_selection/genetic.py
# ruff: noqa: N806

from collections.abc import Callable
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

    Args:
        task: ``'classification'`` или ``'regression'``.
        metric: Строка-метрика или ``callable(y_true, y_score) -> float``
            (конвенция: значение для минимизации; для «выше — лучше» нужен минус).
            Строки (классификация): ``'pr_auc'``, ``'roc_auc'``, ``'f1'``,
            ``'balanced_accuracy'``, ``'logloss'``, ``'brier'``, ``'mcc'``,
            ``'accuracy'``. Строки (регрессия): ``'mae'``, ``'rmse'``,
            ``'median_ae'``, ``'mape'``, ``'smape'``, ``'r2'``.
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
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        matthews_corrcoef,
        mean_squared_error,
        median_absolute_error,
        r2_score,
        roc_auc_score,
    )

    _cat_set = set(cat_features or [])

    def scorer(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        cat_sel = [f for f in X_train.columns if f in _cat_set]
        ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor
        model = ModelClass(**model_params)
        train_pool = Pool(X_train, y_train, cat_features=cat_sel, baseline=baseline_train)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_sel, baseline=baseline_valid)
        model.fit(train_pool, eval_set=valid_pool)

        if task == 'classification':
            proba = model.predict_proba(valid_pool)[:, 1]
            pred = (proba >= 0.5).astype(int)
            if callable(metric):
                return float(metric(y_valid.to_numpy(), proba))
            if metric == 'pr_auc':
                return -average_precision_score(y_valid, proba)
            if metric == 'roc_auc':
                return -roc_auc_score(y_valid, proba)
            if metric == 'f1':
                return -f1_score(y_valid, pred)
            if metric == 'balanced_accuracy':
                return -balanced_accuracy_score(y_valid, pred)
            if metric == 'logloss':
                return log_loss(y_valid, proba)
            if metric == 'brier':
                return brier_score_loss(y_valid, proba)
            if metric == 'mcc':
                return -matthews_corrcoef(y_valid, pred)
            if metric == 'accuracy':
                return -accuracy_score(y_valid, pred)
            raise ValueError(f'Unsupported classification metric: {metric}')

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
        gen_params: Словарь параметров алгоритма. Обязательные ключи:
            ``max_features``, ``population_size``, ``n_generations``,
            ``cross_probability``, ``mutation_probability``.
            Ровно один из двух ключей инициализации начальной популяции:
                ``start_probability_to_include_feature`` — вероятность включить
                признак; ``start_n_features`` — фиксированное число случайно
                выбранных признаков.
            Опциональные:
                ``n_features_without_penalty`` — признаки без штрафа;
                ``penalty_for_extra_feature`` — штраф за каждый лишний признак;
                ``min_improvement``, ``min_improvement_in_percents``,
                ``n_epoch_for_min_improvement`` — early stopping;
                ``must_be_included`` (list[int | str]) — обязательные признаки:
                    целые числа интерпретируются как индексы, строки — как имена;
                ``seed`` (int) — фиксация random state.

    Returns:
        Список имён признаков из глобально лучшей особи.

    Raises:
        ValueError: Если ``feature_names`` пуст или параметры некорректны.

    """
    if not feature_names:
        raise ValueError('feature_names must not be empty')
    if gen_params['max_features'] < 1:
        raise ValueError('max_features must be at least 1')
    if gen_params['population_size'] < 2:
        raise ValueError('population_size must be at least 2')
    if gen_params['n_generations'] < 1:
        raise ValueError('ngen must be at least 1')

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
        return repair_individual(ind, upper_bound)

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

    hof = tools.HallOfFame(1)

    try:
        toolbox = base.Toolbox()
        toolbox.register('individual', tools.initIterate, IndividualClass, create_valid_individual)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('evaluate', _evaluate_cached)
        toolbox.register('mate', cx_uniform_with_repair, upper_bound=upper_bound, must=must, prob=0.5)
        toolbox.register('mutate', mut_flip_bit_with_repair, upper_bound=upper_bound, indpb=0.05, must=must)
        toolbox.register('select', tools.selTournament, tournsize=3)

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

            elite = IndividualClass(hof[0][:])
            elite.fitness.values = hof[0].fitness.values
            pop = [elite] + toolbox.select(offspring, k=len(pop) - 1)

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
