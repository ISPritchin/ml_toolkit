"""Пакет переиспользуемых примитивов.

`feature_generation.py`, `transformers/`, `models/`, `feature_selection/`,
`model_evaluation/`, ...

Конкретные бизнес-задачи, которые собирают эти примитивы в решение одной
проблемы (сборка датасетов, фиксированный набор входных/выходных данных),
живут в отдельных sibling-проектах (например, `auto_kkp_classification`),
подключающих `ml_toolkit` как обычную (editable path) зависимость — а не
здесь и не в подпакете этого репозитория.

Сами модули используют `logging.getLogger(__name__)` без хендлеров (стандартная
гигиена для библиотечного кода - конфигурировать вывод должен потребитель, а
не сам пакет при импорте). Чтобы увидеть логи в Jupyter/скрипте, вызовите
`configure_logging()` один раз после импорта. Обратите внимание: это покрывает
только логи `ml_toolkit.*` - у каждого бизнес-проекта своя одноимённая
`configure_logging()` для его собственного логгера (см., например,
`cltv_dataset_builder.configure_logging()` в `auto_kkp_classification`).
"""

import logging

_PACKAGE_LOGGER_NAME = __name__  # "ml_toolkit" - общий предок для ml_toolkit.feature_generation и т.п.


def configure_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """Включает вывод логов пакета (в stderr) для интерактивной работы.

    Вешает `StreamHandler` на логгер `"ml_toolkit"` - это общий предок всех модулей
    пакета (`ml_toolkit.feature_generation`, `ml_toolkit.correlation_filter`, `ml_toolkit.models`,
    ...), поэтому одного вызова достаточно, чтобы увидеть логи из любого из
    них. Безопасно вызывать повторно (например, при повторном запуске ячейки
    в notebook) - дублирующий хендлер не добавится.

    Args:
        level: Минимальный уровень логирования (`logging.INFO` по умолчанию;
            `logging.DEBUG` даёт более подробные сообщения по каждому
            product-колонке/группе трансформеров и по каждому отброшенному
            кандидату фичи).
        fmt: Формат сообщения для `logging.Formatter`. Если `None` - формат
            по умолчанию: `"%(asctime)s %(levelname)s %(name)s: %(message)s"`.

    Example:
        ```python
        from ml_toolkit import configure_logging
        from cltv_dataset_builder import configure_logging as configure_task_logging
        configure_logging()          # логи ml_toolkit.* (генерический движок и т.п.)
        configure_task_logging()     # логи cltv_dataset_builder.* (бизнес-пайплайн)

        from cltv_dataset_builder import build_feature_datasets
        build_feature_datasets(...)  # теперь логи видны
        ```

    """
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)

    has_stream_handler = any(
        isinstance(existing_handler, logging.StreamHandler)
        for existing_handler in package_logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt or '%(asctime)s %(levelname)s %(name)s: %(message)s')
        )
        package_logger.addHandler(handler)

    # не пускаем записи дальше к root - иначе при повторном вызове или при
    # уже настроенном logging.basicConfig() в окружении сообщения продублируются
    package_logger.propagate = False
