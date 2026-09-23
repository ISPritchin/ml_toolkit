"""Тесты `select_features_genetic` (ядро GA): эталон поведения, новые настройки операторов, проверка ключей.

Скорер здесь — детерминированная функция от набора колонок (без обучения моделей), поэтому тесты
быстрые, а результат при фиксированном `seed` воспроизводим. Колонки данных фиктивные: важны только имена.
"""

import numpy as np
import pandas as pd
import pytest

import ml_toolkit.feature_selection.genetic._core as core
from ml_toolkit.feature_selection import select_features_genetic

N = 40
NAMES = [f'f{i}' for i in range(N)]

_rng = np.random.RandomState(123)
TARGET = {NAMES[i]: float(w) for i, w in zip(_rng.choice(N, 6, replace=False), _rng.uniform(0.5, 1.5, 6))}


def scorer(X_tr, y_tr, X_va, y_va):
    """Минимизируется: чем больше «полезных» признаков, тем лучше; небольшой штраф за размер."""
    return 1.0 - sum(TARGET.get(c, 0.0) for c in X_tr.columns) + 0.02 * len(X_tr.columns)


@pytest.fixture
def data():
    X = pd.DataFrame(np.zeros((5, N)), columns=NAMES)
    return X, pd.Series(np.zeros(5))


def run(data, gen_params, scorer_fn=scorer, callback=None):
    X, y = data
    return select_features_genetic(X, y, X, y, NAMES, scorer_fn, gen_params, generation_callback=callback)


def base_params(**overrides):
    params = dict(
        max_features=8, population_size=10, n_generations=4, cross_probability=0.6,
        mutation_probability=0.4, start_probability_to_include_feature=0.1, seed=5,
    )
    params.update(overrides)
    return params


# ── Эталон: поведение с настройками по умолчанию не должно меняться ────────────

GOLDEN = {
    'start_prob': (
        dict(max_features=8, population_size=12, n_generations=6, cross_probability=0.6,
             mutation_probability=0.3, start_probability_to_include_feature=0.2, seed=7),
        ['f5', 'f8', 'f10', 'f11', 'f18', 'f24', 'f38'],
        [(1, -2.356518021, -1.295698771, 8), (2, -2.478801152, -1.953597434, 8),
         (3, -2.478801152, -2.354801545, 8), (4, -3.744797102, -2.79530014, 8),
         (5, -3.764797102, -2.80530014, 7), (6, -3.764797102, -3.130132461, 7)],
    ),
    'start_n_penalty': (
        dict(max_features=10, population_size=10, n_generations=5, cross_probability=0.5,
             mutation_probability=0.2, start_n_features=5, n_features_without_penalty=3,
             penalty_for_extra_feature=0.01, seed=11),
        ['f2', 'f5', 'f25', 'f34', 'f38'],
        [(1, -1.620911412, -1.282411751, 5), (2, -1.620911412, -1.484469715, 5),
         (3, -1.620911412, -1.620911412, 5), (4, -1.620911412, -1.620911412, 5),
         (5, -1.620911412, -1.620911412, 5)],
    ),
    'early_stop': (
        dict(max_features=8, population_size=10, n_generations=30, cross_probability=0.7,
             mutation_probability=0.4, start_probability_to_include_feature=0.15,
             min_improvement=0.05, n_epoch_for_min_improvement=2, seed=3),
        ['f5', 'f6', 'f21', 'f25', 'f37', 'f38'],
        [(1, -2.788712544, -2.075217815, 6), (2, -2.788712544, -2.707364303, 6),
         (3, -2.788712544, -2.784712544, 6)],
    ),
}


@pytest.mark.parametrize('case', GOLDEN)
def test_default_behaviour_matches_recorded_golden(data, case):
    """Снято с реализации до вынесения зашитых параметров: выбор и вся история поколений совпадают."""
    params, expected_selected, expected_history = GOLDEN[case]
    history = []
    selected = run(data, params, callback=lambda g, b, m, k: history.append((g, round(b, 9), round(m, 9), k)))
    assert selected == expected_selected
    assert len(history) == len(expected_history)
    for got, want in zip(history, expected_history, strict=True):
        assert got[0] == want[0]
        assert got[3] == want[3]
        assert got[1:3] == pytest.approx(want[1:3], abs=1e-8)


def test_explicit_defaults_equal_omitted_keys(data):
    params = base_params()
    explicit = base_params(
        mutation_gene_probability=core.DEFAULT_MUTATION_GENE_PROBABILITY,
        crossover_gene_probability=core.DEFAULT_CROSSOVER_GENE_PROBABILITY,
        tournament_size=core.DEFAULT_TOURNAMENT_SIZE,
        elite_size=core.DEFAULT_ELITE_SIZE,
    )
    assert run(data, params) == run(data, explicit)


# ── Проверка ключей gen_params ────────────────────────────────────────────────

def test_unknown_key_raises_with_suggestion(data):
    with pytest.raises(ValueError, match=r"penalty_for_extra_featur'.*возможно, 'penalty_for_extra_feature'"):
        run(data, base_params(penalty_for_extra_featur=0.01))


def test_unknown_key_without_close_match_lists_allowed_keys(data):
    with pytest.raises(ValueError, match=r"Неизвестные ключи gen_params: 'zzz'.*Допустимые"):
        run(data, base_params(zzz=1))


def test_all_documented_keys_are_accepted(data):
    params = base_params(
        n_features_without_penalty=2, penalty_for_extra_feature=0.01, min_improvement=1e-9,
        n_epoch_for_min_improvement=2, must_be_included=['f0'], mutation_flips=2,
        crossover_gene_probability=0.3, tournament_size=2, elite_size=2,
    )
    assert isinstance(run(data, params), list)
    assert core.KNOWN_GEN_PARAMS >= {
        'max_features', 'population_size', 'n_generations', 'cross_probability', 'mutation_probability',
        'start_probability_to_include_feature', 'start_n_features', 'must_be_included', 'seed',
        'mutation_gene_probability', 'mutation_flips', 'crossover_gene_probability', 'tournament_size', 'elite_size',
    }


# ── must_be_included при инициализации ────────────────────────────────────────

def test_must_features_survive_initial_repair(data):
    """Без кроссовера и мутации особи оцениваются такими, какими их создала инициализация.

    До исправления ремонт при инициализации не знал про `must` и мог случайно снять обязательный признак,
    если N * start_probability > max_features.
    """
    seen = []

    def spy(X_tr, y_tr, X_va, y_va):
        seen.append(list(X_tr.columns))
        return scorer(X_tr, y_tr, X_va, y_va)

    for seed in range(15):
        run(data, dict(
            max_features=3, population_size=8, n_generations=1, cross_probability=0.0, mutation_probability=0.0,
            start_probability_to_include_feature=0.6, must_be_included=['f0', 'f1'], seed=seed,
        ), scorer_fn=spy)
    assert seen
    assert all({'f0', 'f1'} <= set(cols) for cols in seen)
    assert all(len(cols) <= 3 for cols in seen)


def test_must_features_survive_initial_repair_with_start_n_features(data):
    seen = []

    def spy(X_tr, y_tr, X_va, y_va):
        seen.append(list(X_tr.columns))
        return scorer(X_tr, y_tr, X_va, y_va)

    for seed in range(15):
        run(data, dict(
            max_features=4, population_size=8, n_generations=1, cross_probability=0.0, mutation_probability=0.0,
            start_n_features=30, must_be_included=['f7'], seed=seed,
        ), scorer_fn=spy)
    assert all('f7' in cols and len(cols) <= 4 for cols in seen)


# ── Настройки операторов: значения доходят до DEAP ────────────────────────────

class _Spy:
    """Подмена оператора: запоминает kwargs регистрации и вызывает оригинал."""

    def __init__(self, original):
        self.original = original
        self.kwargs = []

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        return self.original(*args, **kwargs)


def test_mutation_gene_probability_reaches_mutation_operator(data, monkeypatch):
    spy = _Spy(core.mut_flip_bit_with_repair)
    monkeypatch.setattr(core, 'mut_flip_bit_with_repair', spy)
    run(data, base_params(mutation_probability=1.0, mutation_gene_probability=0.2))
    assert spy.kwargs
    assert {kw['indpb'] for kw in spy.kwargs} == {0.2}


def test_mutation_flips_translates_to_gene_probability_over_n(data, monkeypatch):
    spy = _Spy(core.mut_flip_bit_with_repair)
    monkeypatch.setattr(core, 'mut_flip_bit_with_repair', spy)
    run(data, base_params(mutation_probability=1.0, mutation_flips=2))
    assert spy.kwargs
    assert all(kw['indpb'] == pytest.approx(2 / N) for kw in spy.kwargs)


def test_mutation_flips_is_capped_at_probability_one(data, monkeypatch):
    spy = _Spy(core.mut_flip_bit_with_repair)
    monkeypatch.setattr(core, 'mut_flip_bit_with_repair', spy)
    run(data, base_params(mutation_probability=1.0, mutation_flips=10 * N))
    assert {kw['indpb'] for kw in spy.kwargs} == {1.0}


def test_default_mutation_gene_probability_is_unchanged(data, monkeypatch):
    spy = _Spy(core.mut_flip_bit_with_repair)
    monkeypatch.setattr(core, 'mut_flip_bit_with_repair', spy)
    run(data, base_params(mutation_probability=1.0))
    assert {kw['indpb'] for kw in spy.kwargs} == {0.05}


def test_crossover_gene_probability_reaches_crossover_operator(data, monkeypatch):
    spy = _Spy(core.cx_uniform_with_repair)
    monkeypatch.setattr(core, 'cx_uniform_with_repair', spy)
    run(data, base_params(cross_probability=1.0, crossover_gene_probability=0.25))
    assert spy.kwargs
    assert {kw['prob'] for kw in spy.kwargs} == {0.25}


def test_tournament_size_reaches_selection(data, monkeypatch):
    spy = _Spy(core.tools.selTournament)
    monkeypatch.setattr(core.tools, 'selTournament', spy)
    run(data, base_params(tournament_size=5))
    assert spy.kwargs
    assert {kw['tournsize'] for kw in spy.kwargs} == {5}


@pytest.mark.parametrize('elite_size', [0, 1, 3])
def test_elite_size_controls_number_of_tournament_slots(data, monkeypatch, elite_size):
    """В следующее поколение идут elite_size элитных особей + (population_size - elite_size) из турнира."""
    ks = []
    original = core.tools.selTournament

    def spy(individuals, k, **kwargs):
        ks.append(k)
        return original(individuals, k, **kwargs)

    monkeypatch.setattr(core.tools, 'selTournament', spy)
    population_size = 10
    run(data, base_params(population_size=population_size, elite_size=elite_size))
    # на первых поколениях в зале славы может быть меньше elite_size различных особей — тогда слотов больше
    assert all(k >= population_size - elite_size for k in ks)
    assert ks[-1] == population_size - elite_size


def test_zero_elite_still_returns_best_ever(data):
    history = []
    selected = run(
        data, base_params(elite_size=0, n_generations=6),
        callback=lambda g, best, mean, k: history.append(best),
    )
    assert selected
    assert history == sorted(history, reverse=True)  # лучший за всю историю не ухудшается


# ── Валидация новых ключей ─────────────────────────────────────────────────────

def test_mutation_gene_probability_and_flips_are_mutually_exclusive(data):
    with pytest.raises(ValueError, match='mutation_gene_probability и mutation_flips'):
        run(data, base_params(mutation_gene_probability=0.1, mutation_flips=2))


@pytest.mark.parametrize(('key', 'value'), [
    ('mutation_gene_probability', -0.1),
    ('mutation_gene_probability', 1.5),
    ('mutation_flips', 0),
    ('mutation_flips', -3),
    ('crossover_gene_probability', 1.1),
    ('crossover_gene_probability', -0.5),
    ('tournament_size', 0),
    ('tournament_size', 2.5),
    ('elite_size', -1),
    ('elite_size', 10),  # == population_size
    ('elite_size', 1.5),
])
def test_invalid_operator_values_raise(data, key, value):
    with pytest.raises(ValueError, match=key):
        run(data, base_params(**{key: value}))


def test_results_stay_within_max_features_for_all_operator_settings(data):
    selected = run(data, base_params(
        mutation_flips=3, crossover_gene_probability=0.7, tournament_size=2, elite_size=2, n_generations=8,
    ))
    assert 1 <= len(selected) <= 8
    assert set(selected) <= set(NAMES)
