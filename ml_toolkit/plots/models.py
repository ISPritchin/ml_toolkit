from collections.abc import Iterable
import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml_toolkit.data_description import CommissionTableColumns
from ml_toolkit.plots.client_viz import (
    draw_client_prediction,
)

try:
    from .utils import (
        modify_ticks,
        modify_ticks_percent,
        modify_xticks_for_date_axis,
        number_to_number_with_suffix,
    )
except ModuleNotFoundError:
    # utils зависит от опциональных пакетов; заглушки для корректного импорта модуля
    def modify_ticks(ax: plt.Axes, **kw) -> None: pass  # type: ignore[misc]
    def modify_ticks_percent(ax: plt.Axes, **kw) -> None: pass  # type: ignore[misc]
    def modify_xticks_for_date_axis(ax: plt.Axes, **kw) -> None: pass  # type: ignore[misc]
    def number_to_number_with_suffix(*a, **kw): return str(a[0]) if a else ''  # type: ignore[misc]


def show_mask_selection_plot(
    monthly_df: pl.DataFrame,
    *,
    ts_column_name: str,
    target_column_name: str,
    mask_column_name: str,
    show: bool = True,
    save_path: Path | None = None
) -> None:
    """Строит график, отражающий, сколько данных было отобрано на основании маски mask_column_name.

    График позволяет получить понимание, сколько подходящих клиентов находится в каждом из месяцев.

    Args:
        monthly_df (pl.DataFrame): датафрейм
        ts_column_name (str): имя колонки с датой
        target_column_name (str): имя колонки с целевой переменной
        mask_column_name (str): имя колонки с маской
        show (bool, optional): 'истина', если требуется отображать график в Jupyter Notebook. Defaults to True.
        save_path (Path | None, optional): путь для сохранения графика. Defaults to None.

    """
    stat_activity = monthly_df.group_by(ts_column_name).agg(
        (pl.col(target_column_name) != 0).sum().alias('n_any_activity'),
        (pl.col(mask_column_name) != 0).sum().alias('n_good_activity'),
    ).sort(ts_column_name).with_columns(
        (pl.col('n_good_activity') / pl.col('n_any_activity') * 100).alias('coverage')
    )

    ax = plt.gca()

    ax.plot(
        stat_activity[ts_column_name],
        stat_activity['n_any_activity'],
        'o--',
        label='Количество клиентов с ненулевой комиссией'
    )
    ax.plot(
        stat_activity[ts_column_name],
        stat_activity['n_good_activity'],
        'o--',
        label='Количество клиентов для прогнозирования'
    )

    last_row = stat_activity[-1]
    coverage = last_row['coverage'][0]
    n_selected_clients = last_row['n_good_activity'][0]
    ax.text(
        x=stat_activity[ts_column_name].max(),
        y=n_selected_clients,
        s=f'{number_to_number_with_suffix(n_selected_clients)}\n({round(coverage, 2)}%)'
    )

    modify_ticks(ax, axis='y')
    modify_xticks_for_date_axis(ax)

    ax.set_title('Результаты работы маски')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Количество клиентов')
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close()


def get_error_distribution_plot(
    predicts: dict[str, pl.DataFrame],
    *,
    min_error: float = 0,
    max_error: float = 10,
    min_confidence: float = 0.5,
    max_confidence: float = 1,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Выводит график рапределения ошибки.

    Args:
        predicts (dict[str, pl.DataFrame]): словарь с прогнозами моделей
        min_error (float, optional): минимальная ошибка при отборе в данных. Defaults to 0.
        max_error (float, optional): максимальная ошибка при отборе данных. Defaults to 10.
        min_confidence (float, optional): минимальная уверенность в прогнозе при отборе данных. Defaults to 0.5.
        max_confidence (float, optional): максимальная уверенность в прогнозе при отборе данных. Defaults to 1.
        save_path (Path | None, optional): путь для сохранения графика. Defaults to None.
        show (bool, optional): 'истина', если требуется отображать график в Jupyter Notebook. Defaults to True.

    """
    n_models = len(predicts)
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(8, n_models * 2), sharex=True, sharey=True)

    axes = np.atleast_1d(axes).flatten()
    y_max = 0
    for ax, model_name in zip(axes, predicts.keys(), strict=False):
        subset = predicts[model_name].filter(
            pl.col('error').is_between(min_error, max_error),
            pl.col('confidence').is_between(min_confidence, max_confidence)
        )
        import seaborn as sns
        sns.histplot(
            subset,
            x='error',
            ax=ax,
            hue=(subset[model_name] < subset['y_true']),
            multiple='stack',
            legend=False
        )
        ax.set_title(model_name)
        ax.set_ylabel('Количество клиентов')
        y_max = max(y_max, ax.get_ylim()[1])

    for ax in axes:
        ax.set_ylim(0, y_max)

    ax.set_xlabel('Ошибка в процентах')
    modify_ticks_percent(ax, 'x')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def get_different_quantiles(
    predicts: dict[str, pl.DataFrame],
    *,
    quantiles: list[int] = (0.05, 0.25, 0.5, 0.75, 0.95),
    min_error: float = 0,
    max_error: float = 10,
    min_confidence: float = 0.5,
    max_confidence: float = 1,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Выводит график, который отражает ошибки для разных квантилей отобранных данных.

    Args:
        predicts (dict[str, pl.DataFrame]): словарь с прогнозами моделей
        quantiles (list[int], optional): квантили ошибки для отображения. Defaults to (0.05, 0.25, 0.5, 0.75, 0.95).
        min_error (float, optional): минимальная ошибка при отборе в данных. Defaults to 0.
        max_error (float, optional): максимальная ошибка при отборе данных. Defaults to 10.
        min_confidence (float, optional): минимальная уверенность в прогнозе при отборе данных. Defaults to 0.5.
        max_confidence (float, optional): максимальная уверенность в прогнозе при отборе данных. Defaults to 1.
        save_path (Path | None, optional): путь для сохранения графика. Defaults to None.
        show (bool, optional): 'истина', если требуется отображать график в Jupyter Notebook. Defaults to True.

    """
    n_models = len(predicts)

    drop_first = 10
    fig, axes = plt.subplots(ncols=n_models, nrows=1, figsize=(n_models * 4, 4), sharey=True)
    for model_index, (ax, model_name) in enumerate(zip(axes, predicts.keys(), strict=False), start=1):
        pred = predicts[model_name].filter(
            pl.col('error').is_between(min_error, max_error),
            pl.col('confidence').is_between(min_confidence, max_confidence)
        )
        # ruff: noqa: ISC002
        ax.set_title(
            f'{model_name}\nSMedianAPE={pred["error"].median() * 100:.1f}%\n' \
            f'SMAPE={pred["error"].mean() * 100:.1f}%'
        )

        lines = {quantile: [] for quantile in quantiles}
        for i in range(1, len(pred['error']) + 1):
            for quantile in quantiles:
                lines[quantile].append(pred['error'][:i].quantile(quantile))

        lines['confidence'] = pred['confidence']

        for line_name, values in lines.items():
            ax.plot(
                range(drop_first + 1, len(pred) + 1),
                values[drop_first:],
                label=line_name,
            )

        if model_index == 1:
            modify_ticks_percent(ax, 'y')

        if model_index == n_models:
            ax.legend()

        modify_ticks(ax, axis='x', func=partial(number_to_number_with_suffix, add_new_line_character=True))

    plt.suptitle('Сравнение моделей вцелом')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def get_sample_plot(
    predicts: dict[str, pl.DataFrame],
    *,
    model_name: str,
    id_column_name: str,
    ts_column_name: str,
    field_to_predict: str,
    commissions_df: pl.DataFrame,
    daily_df: pl.DataFrame,
    last_date: datetime.date,
    n_examples: int = 21,
    n_plots_in_a_row: int = 3,
    min_confidence: float = 0,
    max_confidence: float = 1,
    min_error: float = 0,
    max_error: float = 5000,
    save_path: Path | None = None,
    show: bool = True
):
    """Строит сетку графиков с историей комиссий и предиктами для случайных клиентов.

    Фильтрует клиентов по уверенности и ошибке, затем отображает для каждого
    ежемесячный объём комиссий, предсказанное значение и понедельные данные.

    Args:
        predicts: Словарь с прогнозами моделей; ключ — имя модели.
        model_name: Имя модели, предикты которой отображать.
        id_column_name: Имя столбца с идентификатором клиента.
        ts_column_name: Имя столбца с временной меткой.
        field_to_predict: Имя поля с объёмом комиссии.
        commissions_df: Исходный датафрейм транзакций (не используется напрямую).
        daily_df: Датафрейм с дневной активностью клиентов.
        last_date: Последняя дата для отображения.
        n_examples: Максимальное число клиентов для отображения. По умолчанию 21.
        n_plots_in_a_row: Число графиков в строке. По умолчанию 3.
        min_confidence: Минимальная уверенность модели для фильтрации. По умолчанию 0.
        max_confidence: Максимальная уверенность модели для фильтрации. По умолчанию 1.
        min_error: Минимальная ошибка для фильтрации. По умолчанию 0.
        max_error: Максимальная ошибка для фильтрации. По умолчанию 5000.
        save_path: Путь для сохранения графика. По умолчанию None.
        show: Если ``True`` — отображает в Jupyter Notebook. По умолчанию True.

    Returns:
        Датафрейм `client_raw_data` последнего обработанного клиента.

    """
    t = predicts[model_name].filter(
        pl.col('confidence').is_between(min_confidence, max_confidence),
        pl.col('error').is_between(min_error, max_error),
    )
    n_total = len(t)
    n_examples = min(n_examples, len(t))
    t = t.sample(n_examples).sort('confidence',  descending=True)

    n_rows = (n_examples // n_plots_in_a_row) + int((n_examples % n_plots_in_a_row) != 0)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_plots_in_a_row,
        figsize=(16, n_rows * 3.5),
    )
    axes = np.array(axes).flatten()

    for i in range(len(t)):
        client_id = t[i][CommissionTableColumns.ID_COLUMN_NAME][0]
        confidence = round(t[i]['confidence'][0], 2)
        y_true = round(t[i]['y_true'][0], 1)
        y_pred = round(t[i]['y_pred'][0], 1)
        error = round(t[i]['error'][0] * 100, 1)
        date = t[i][ts_column_name][0]

        client_raw_data = daily_df.filter(
            pl.col(CommissionTableColumns.ID_COLUMN_NAME) == client_id
        )
        n_weeks = int(
            (client_raw_data[CommissionTableColumns.TS_COLUMN_NAME].max() - client_raw_data[CommissionTableColumns.TS_COLUMN_NAME].min()).total_seconds() \
            // 60 / 60 / 24 / 7
        )

        from ml_toolkit.feature_extraction.weekly import (
            add_records_for_absent_data,
            select_data_for_last_weeks,
        )
        weeks_data = select_data_for_last_weeks(
            client_raw_data,
            ts_column_name=CommissionTableColumns.TS_COLUMN_NAME,
            last_date=last_date,
            n_weeks=n_weeks
        )

        week_data = add_records_for_absent_data(
            weeks_data,
            id_column_name=CommissionTableColumns.ID_COLUMN_NAME,
            last_date=last_date,
            n_weeks=n_weeks
        ).group_by(
            'end_week_date'
        ).agg(
            pl.col('fee_nds_amount').sum(),
        ).sort('end_week_date')

        draw_client_prediction(axes[i], daily_df, client_id, date, y_pred, confidence, y_true)
        # Override title with legacy format expected by this function
        axes[i].set_title(
            f'id={client_id}, confidence={confidence}\ny_true={y_true}, y_pred={y_pred}, error={error}%',
            fontsize=7,
        )
        # Weekly overlay (дополнительный слой поверх базового графика)
        axes[i].plot(week_data['end_week_date'], week_data['fee_nds_amount'], 'o-', alpha=0.6)

    for i in range(len(t), n_plots_in_a_row * n_rows):
        axes[i].set_visible(False)

    plt.suptitle(f'Запросу с {min_confidence} <= confidence <= {max_confidence} and {min_error} <= error <= {max_error}\nудовлетворяет {n_total} случаев из {len(predicts[model_name])}. Отображены случайные {n_examples} клиентов', y=1)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return client_raw_data


def get_cls_proba_calibration(
    predicts: dict[str, pl.DataFrame],
    *,
    error_threshold: float = 0.2,
    min_confidence: float = 0.5,
    max_confidence: float = 1,
    n_points: int = 10,
    save_path: Path | None = None,
    show: bool=  True
) -> None:
    """Выводит график калибровки классификатора.

    Args:
        predicts (dict[str, pl.DataFrame]): словарь с прогнозами моделей
        error_threshold (float, optional): порог, по которому считаем прогноз хорошим. Defaults to 0.2.
        min_confidence (float, optional): минимальная уверенность в прогнозе при отборе данных. Defaults to 0.5.
        max_confidence (float, optional): максимальная уверенность в прогнозе при отборе данных. Defaults to 1.
        n_points (int, optional): количество токек, по которым оцениваем модель. Defaults to 10.
        save_path (Path | None, optional): путь. Defaults to None.
        save_path (Path | None, optional): путь для сохранения графика. Defaults to None.
        show (bool, optional): 'истина', если требуется отображать график в Jupyter Notebook. Defaults to True.

    """
    fig, axes = plt.subplots(ncols=len(predicts), figsize=(4*len(predicts), 4), sharey=True)
    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax, model in zip(axes, predicts, strict=False):
        if ax == axes[0]:
            ax.set_ylabel('Вероятность')

        res = predicts[model].filter(
            pl.col('confidence').is_between(min_confidence, max_confidence)
        )
        res = res.with_columns((pl.col('error') < error_threshold).alias('is_good_predict'))
        s = len(res) / n_points
        x = res.with_columns(pl.Series(range(len(res))).alias('index') // s * s).group_by('index').agg(
            pl.col('is_good_predict').mean(),
            pl.col('confidence').mean()
        ).sort('index')
        ax.plot(x['index'], x['is_good_predict'], 'o-', label='true')
        ax.plot(x['index'], x['confidence'], 'o-', label='pred')
        ax.set_title(model)
        ax.set_xlabel('Количество отобранных клиентов')
        modify_ticks(ax, axis='x')

    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def get_error_distribution_for_different_threshold(
    predicts: dict[str, pl.DataFrame],
    *,
    min_confidence: float = 0.5,
    error_thresholds: list[float] = (0.05, 0.25, 0.5, 1),
    n_drop_first: int = 10,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Построение графиков распределения ошибок для разных порогов на основе прогнозов моделей.

    Функция фильтрует данные по уверенности модели и строит кривые, показывающие долю
    правильных предсказаний при различных порогах ошибки. Графики отображаются по
    количеству клиентов, отобранных моделью.

    Args:
        predicts (dict[str, pl.DataFrame]): Словарь с именами моделей в качестве ключей
            и DataFrame с прогнозами в качестве значений. Каждый DataFrame должен содержать
            столбцы 'error', 'confidence', 'y_true'.
        min_confidence (float, optional): Минимальная уверенность модели в предсказании,
            используется для фильтрации данных. По умолчанию 0.5.
        error_thresholds (List[float], optional): Список пороговых значений ошибки для анализа.
            По умолчанию [0.05, 0.25, 0.5, 1].
        n_drop_first (int, optional): Количество первых наблюдений, которые игнорируются
            при построении графика. По умолчанию 10.
        save_path (Path | None, optional): Путь для сохранения графика. Если не указан,
            график не сохраняется. По умолчанию None.
        show (bool, optional): Флаг, определяющий, нужно ли отображать график. По умолчанию True.

    Returns:
        None: Результатом является отображение или сохранение графиков, возвращаемого значение нет.

    """
    n_models = len(predicts)

    _, axes = plt.subplots(ncols=n_models, nrows=1, figsize=(n_models * 4, 4), sharey=True)
    x_max = 0
    for model_index, (ax, model_name) in enumerate(zip(axes, predicts.keys(), strict=False), start=1):
        pred = predicts[model_name].filter(
            pl.col('error') < 5,
            pl.col('confidence') > min_confidence
        )
        ax.set_title(model_name)

        labels = [f'< {round(threshold * 100)}%' for threshold in error_thresholds]
        for error_threshold, label in zip(error_thresholds, labels, strict=False):
            v = ((pred['error'] < error_threshold).cum_sum() / pl.Series(range(1, len(pred) + 1)))[n_drop_first:]
            ax.plot(
                range(n_drop_first + 1, len(pred) + 1),
                v,
                '--',
                label=label
            )

        if model_index == n_models:
            ax.legend()

        if model_index == 1:
            ax.set_ylabel('Процент')
            modify_ticks_percent(ax, 'y')

        ax.set_ylim(0, 1)
        ax.set_xlabel('Количество отобранных клиентов')

        ax.plot(
            range(n_drop_first + 1, len(pred) + 1),
            pred['confidence'][n_drop_first:],
            label='confidence',
            alpha=0.2
        )

        ax.plot(
            range(n_drop_first + 1, len(pred) + 1),
            (1 - (pred['y_true'] == 0).cum_sum() / pl.Series(range(1, len(pred) + 1)))[n_drop_first:],
            c='red'
        )

        modify_ticks(ax, axis='x', func=partial(number_to_number_with_suffix, add_new_line_character=True))
        x_max = max(x_max, len(pred) + 1)

    for ax in axes:
        ax.set_xlim(0, x_max)

    plt.suptitle('Распределение ошибки на разном количестве отобранных клиентов')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=150, transparent=True)

    if show:
        plt.show()
    else:
        plt.close()
