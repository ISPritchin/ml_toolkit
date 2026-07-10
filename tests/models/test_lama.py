"""Тесты для ml_toolkit/models/_lama.py (LAMARegressor/LAMAClassifier).

Пакет lightautoml не входит в обязательные зависимости проекта — весь модуль пропускается
через importorskip, если он не установлен.

ВАЖНО: здесь сознательно НЕТ end-to-end fit()-тестов (только чистые unit-тесты на
_build_roles() и guard params). LightAutoML использует внутренний multiprocessing
(reader CV), который на некоторых macOS-конфигурациях (включая песочницу, где писались эти
тесты) падает с "OMP: Error #179: pthread_mutex_init failed" — fork-safety баг во
взаимодействии OpenMP-рантайма и fork() на macOS. Он воспроизводится даже в чистом скрипте
без ml_toolkit (и без сокращения числа воркеров через model_settings['cpu_limit']=1), не
связан с кодом этого адаптера, и приводит не к обычному failed-тесту, а к SIGSEGV всего
процесса pytest (exit 139) — что убивает весь прогон suite, а не только один тест. Живой
fit()/fit_predict() для LAMARegressor/LAMAClassifier был проверен вручную (см. историю
изменений/commit message), но не включён сюда как permanent regression test.
"""

from __future__ import annotations

import pytest

pytest.importorskip('lightautoml')

from ml_toolkit.models._lama import LAMAClassifier, LAMARegressor, _build_roles  # noqa: E402

FAST_SETTINGS = {'timeout': 20, 'cpu_limit': 1}


class TestBuildRoles:
    """LightAutoML ожидает roles в формате {роль: [колонки]} (роль — ключ), а не {колонка: роль}."""

    def test_target_only(self):
        assert _build_roles([], ['f0', 'f1']) == {'target': '__lama_target__'}

    def test_category_role_groups_columns_under_one_key(self):
        roles = _build_roles(['cat_a', 'cat_b'], ['f0', 'cat_a', 'cat_b'])
        assert roles == {'target': '__lama_target__', 'category': ['cat_a', 'cat_b']}

    def test_cat_feature_not_in_selected_features_is_excluded(self):
        roles = _build_roles(['cat_a', 'cat_c'], ['f0', 'cat_a'])
        assert roles == {'target': '__lama_target__', 'category': ['cat_a']}


class TestLAMAParamsGuard:
    """LAMA управляет тюнингом сама — явные params должны явно отклоняться, а не молча игнорироваться."""

    def test_regressor_rejects_explicit_params(self, regression_data):
        X_train, y_train, X_valid, y_valid = regression_data
        model = LAMARegressor(params={'foo': 1}, model_settings=FAST_SETTINGS)
        with pytest.raises(ValueError, match='не поддерживает явные params'):
            model.fit(X_train, y_train, X_valid, y_valid)

    def test_classifier_rejects_explicit_params(self, classification_data):
        X_train, y_train, X_valid, y_valid = classification_data
        model = LAMAClassifier(params={'foo': 1}, model_settings=FAST_SETTINGS)
        with pytest.raises(ValueError, match='не поддерживает явные params'):
            model.fit(X_train, y_train, X_valid, y_valid)
