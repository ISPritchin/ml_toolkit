"""Табличные supervised-модели (X_train/y_train, BaseModel-контракт).

Подпакеты по семействам: _boosting, _forests, _interpretable, _automl.
Принципиально другой класс задач (иной контракт fit/predict, не наследующий
BaseModel — например, адаптеры sequence/time-series методов) получает свой
сиблинг-раздел на уровне ml_toolkit/models/, а не встраивается сюда.
"""
