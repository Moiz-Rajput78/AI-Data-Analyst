"""Chart generation tools for the AI Data Analyst."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from tools.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    load_dataset,
)


DEFAULT_CHART_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "charts"
)


SUPPORTED_CHART_TYPES = {
    "line",
    "bar",
    "pie",
    "scatter",
}


SUPPORTED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
}


class ChartError(RuntimeError):
    """Raised when a chart cannot be generated."""


def _safe_filename(
    text: str,
) -> str:
    """
    Convert chart title into a filesystem-safe filename.
    """

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        text.strip().lower(),
    )

    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = "chart"

    return cleaned[:80]


def _get_datetime_column(
    dataframe: pd.DataFrame,
) -> str:
    """
    Return the first datetime column in the dataset.
    """

    datetime_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        )
    ]

    if not datetime_columns:
        raise ChartError(
            "The requested time-based chart requires "
            "a datetime column."
        )

    return datetime_columns[0]


def _prepare_x_column(
    dataframe: pd.DataFrame,
    x: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare normal or special x-axis fields.

    Special values:

    - month
    - year
    - quarter
    """

    dataframe = dataframe.copy()

    normalized = x.strip().lower()

    if normalized not in {
        "month",
        "year",
        "quarter",
    }:
        if x not in dataframe.columns:
            raise ChartError(
                f"Column '{x}' does not exist."
            )

        return dataframe, x

    date_column = _get_datetime_column(
        dataframe
    )

    temporary_column = (
        f"__chart_{normalized}"
    )

    if normalized == "month":
        dataframe[temporary_column] = (
            dataframe[date_column]
            .dt.to_period("M")
            .astype(str)
        )

    elif normalized == "year":
        dataframe[temporary_column] = (
            dataframe[date_column]
            .dt.year
            .astype(str)
        )

    elif normalized == "quarter":
        dataframe[temporary_column] = (
            dataframe[date_column]
            .dt.to_period("Q")
            .astype(str)
        )

    return dataframe, temporary_column


def _apply_period_filter(
    dataframe: pd.DataFrame,
    period: str | None,
) -> pd.DataFrame:
    """
    Optionally filter the dataset to a time period.

    Supported examples:

    - 2026
    - 2026-09
    - September
    - September 2026
    """

    if period is None:
        return dataframe

    date_column = _get_datetime_column(
        dataframe
    )

    series = dataframe[
        date_column
    ]

    normalized = period.strip()

    if not normalized:
        return dataframe

    if re.fullmatch(
        r"\d{4}",
        normalized,
    ):
        year = int(
            normalized
        )

        return dataframe[
            series.dt.year == year
        ].copy()

    if re.fullmatch(
        r"\d{4}-\d{2}",
        normalized,
    ):
        year, month = [
            int(part)
            for part
            in normalized.split("-")
        ]

        return dataframe[
            (series.dt.year == year)
            & (series.dt.month == month)
        ].copy()

    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    lower_period = normalized.lower()

    if lower_period in month_names:
        month = month_names[
            lower_period
        ]

        return dataframe[
            series.dt.month == month
        ].copy()

    parsed = pd.to_datetime(
        normalized,
        errors="coerce",
    )

    if not pd.isna(
        parsed
    ):
        return dataframe[
            (series.dt.year == parsed.year)
            & (series.dt.month == parsed.month)
        ].copy()

    raise ChartError(
        f"Unsupported period format: '{period}'."
    )


def _aggregate_data(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """
    Aggregate the y-axis metric by the x-axis category.
    """

    grouped = dataframe.groupby(
        x_column,
        dropna=False,
    )[y_column]

    if aggregation == "sum":
        values = grouped.sum()

    elif aggregation == "mean":
        values = grouped.mean()

    elif aggregation == "median":
        values = grouped.median()

    elif aggregation == "min":
        values = grouped.min()

    elif aggregation == "max":
        values = grouped.max()

    elif aggregation == "count":
        values = grouped.count()

    else:
        raise ChartError(
            f"Unsupported aggregation '{aggregation}'."
        )

    result = (
        values
        .reset_index()
    )

    result.columns = [
        x_column,
        y_column,
    ]

    return result


def _serialize_chart_data(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Return compact chart data for the agent.
    """

    records: list[
        dict[str, Any]
    ] = []

    for _, row in dataframe.head(
        limit
    ).iterrows():

        x_value = row[
            x_column
        ]

        y_value = row[
            y_column
        ]

        if isinstance(
            x_value,
            pd.Timestamp,
        ):
            x_value = (
                x_value.isoformat()
            )

        if hasattr(
            y_value,
            "item",
        ):
            try:
                y_value = (
                    y_value.item()
                )

            except (
                AttributeError,
                ValueError,
            ):
                pass

        records.append(
            {
                "x": str(
                    x_value
                ),
                "y": y_value,
            }
        )

    return records


def create_chart(
    chart_type: str,
    x: str,
    y: str,
    title: str,
    aggregation: str = "sum",
    period: str | None = None,
    output_name: str | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_directory: str | Path = DEFAULT_CHART_DIRECTORY,
) -> dict[str, Any]:
    """
    Generate and save a chart from the dataset.

    Supported chart types:

    - line
    - bar
    - pie
    - scatter

    Parameters
    ----------
    chart_type:
        Type of visualization.

    x:
        X-axis column.

        Special values:
        - month
        - year
        - quarter

    y:
        Numeric metric column.

    title:
        Human-readable chart title.

    aggregation:
        Aggregation used for line, bar, and pie charts.

        Supported:
        - sum
        - mean
        - median
        - min
        - max
        - count

    period:
        Optional date filter.

    output_name:
        Optional PNG filename.

    dataset_path:
        Source CSV.

    output_directory:
        Directory where generated PNG files are stored.
    """

    normalized_chart_type = (
        chart_type
        .strip()
        .lower()
    )

    normalized_aggregation = (
        aggregation
        .strip()
        .lower()
    )

    if normalized_chart_type not in (
        SUPPORTED_CHART_TYPES
    ):
        raise ChartError(
            f"Unsupported chart type '{chart_type}'. "
            "Supported chart types are: "
            + ", ".join(
                sorted(
                    SUPPORTED_CHART_TYPES
                )
            )
        )

    if normalized_aggregation not in (
        SUPPORTED_AGGREGATIONS
    ):
        raise ChartError(
            f"Unsupported aggregation '{aggregation}'. "
            "Supported aggregations are: "
            + ", ".join(
                sorted(
                    SUPPORTED_AGGREGATIONS
                )
            )
        )

    if not title.strip():
        raise ChartError(
            "Chart title cannot be empty."
        )

    try:
        dataframe = load_dataset(
            dataset_path
        )

    except DatasetError as exc:
        raise ChartError(
            str(exc)
        ) from exc

    dataframe = _apply_period_filter(
        dataframe,
        period,
    )

    if dataframe.empty:
        raise ChartError(
            "No dataset rows remain after filtering."
        )

    dataframe, x_column = (
        _prepare_x_column(
            dataframe,
            x,
        )
    )

    if y not in dataframe.columns:
        raise ChartError(
            f"Column '{y}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[y]
    ):
        raise ChartError(
            f"Column '{y}' must be numeric."
        )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_name:
        filename = (
            Path(output_name).stem
            + ".png"
        )

        filename = (
            _safe_filename(
                Path(filename).stem
            )
            + ".png"
        )

    else:
        filename = (
            _safe_filename(
                title
            )
            + ".png"
        )

    output_path = (
        output_directory
        / filename
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    try:
        if normalized_chart_type in {
            "line",
            "bar",
            "pie",
        }:
            chart_data = (
                _aggregate_data(
                    dataframe,
                    x_column,
                    y,
                    normalized_aggregation,
                )
            )

            if (
                x.strip().lower()
                in {
                    "month",
                    "year",
                    "quarter",
                }
            ):
                chart_data = (
                    chart_data
                    .sort_values(
                        by=x_column
                    )
                )

            if normalized_chart_type == (
                "line"
            ):
                axis.plot(
                    chart_data[
                        x_column
                    ].astype(str),
                    chart_data[
                        y
                    ],
                    marker="o",
                )

                axis.set_xlabel(
                    x.replace(
                        "_",
                        " ",
                    ).title()
                )

                axis.set_ylabel(
                    y.replace(
                        "_",
                        " ",
                    ).title()
                )

                axis.tick_params(
                    axis="x",
                    rotation=45,
                )

                axis.grid(
                    alpha=0.3
                )

            elif normalized_chart_type == (
                "bar"
            ):
                axis.bar(
                    chart_data[
                        x_column
                    ].astype(str),
                    chart_data[
                        y
                    ],
                )

                axis.set_xlabel(
                    x.replace(
                        "_",
                        " ",
                    ).title()
                )

                axis.set_ylabel(
                    y.replace(
                        "_",
                        " ",
                    ).title()
                )

                axis.tick_params(
                    axis="x",
                    rotation=45,
                )

            elif normalized_chart_type == (
                "pie"
            ):
                axis.pie(
                    chart_data[y],
                    labels=(
                        chart_data[
                            x_column
                        ]
                        .astype(str)
                    ),
                    autopct="%1.1f%%",
                    startangle=90,
                )

                axis.axis(
                    "equal"
                )

        else:
            if not pd.api.types.is_numeric_dtype(
                dataframe[x_column]
            ):
                raise ChartError(
                    "Scatter charts require a numeric x-axis column."
                )

            chart_data = (
                dataframe[
                    [
                        x_column,
                        y,
                    ]
                ]
                .dropna()
                .copy()
            )

            if chart_data.empty:
                raise ChartError(
                    "No valid numeric rows are available "
                    "for the scatter chart."
                )

            axis.scatter(
                chart_data[
                    x_column
                ],
                chart_data[
                    y
                ],
                alpha=0.7,
            )

            axis.set_xlabel(
                x.replace(
                    "_",
                    " ",
                ).title()
            )

            axis.set_ylabel(
                y.replace(
                    "_",
                    " ",
                ).title()
            )

            axis.grid(
                alpha=0.3
            )

        axis.set_title(
            title
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    except ChartError:
        raise

    except Exception as exc:
        raise ChartError(
            f"Unable to generate chart: {exc}"
        ) from exc

    finally:
        plt.close(
            figure
        )

    if not output_path.exists():
        raise ChartError(
            "Chart generation completed but "
            "the output file was not created."
        )

    result_data = (
        _serialize_chart_data(
            chart_data,
            x_column,
            y,
        )
    )

    return {
        "status": "success",
        "chart_type": (
            normalized_chart_type
        ),
        "title": title,
        "x": x,
        "y": y,
        "aggregation": (
            None
            if normalized_chart_type
            == "scatter"
            else normalized_aggregation
        ),
        "period": period,
        "output_path": str(
            output_path
        ),
        "filename": (
            output_path.name
        ),
        "points": int(
            len(chart_data)
        ),
        "data": result_data,
    }