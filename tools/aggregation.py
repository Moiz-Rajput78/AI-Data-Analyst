"""Grouping and aggregation tools for the AI Data Analyst."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    load_dataset,
)


SUPPORTED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
}


class AggregationError(RuntimeError):
    """Raised when grouping or aggregation cannot be completed."""


def _safe_value(value: Any) -> Any:
    """Convert pandas/numpy values into normal Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass

    return value


def _apply_period_filter(
    dataframe: pd.DataFrame,
    period: str | None,
) -> pd.DataFrame:
    """
    Filter rows by a requested time period.

    Supported formats include:

    - 2026-09
    - September 2026
    - September
    - 2026
    """

    if not period:
        return dataframe

    datetime_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        )
    ]

    if not datetime_columns:
        raise AggregationError(
            "A period filter was requested, but no datetime column exists."
        )

    date_column = datetime_columns[0]

    normalized = period.strip()

    series = dataframe[date_column]

    parsed_month = pd.to_datetime(
        normalized,
        errors="coerce",
    )

    if not pd.isna(parsed_month):
        if len(normalized) <= 4:
            return dataframe[
                series.dt.year
                == parsed_month.year
            ].copy()

        return dataframe[
            (series.dt.year == parsed_month.year)
            & (
                series.dt.month
                == parsed_month.month
            )
        ].copy()

    month_lookup = {
        month.lower(): index
        for index, month in enumerate(
            [
                "",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        )
        if month
    }

    lower_period = normalized.lower()

    if lower_period in month_lookup:
        month_number = month_lookup[
            lower_period
        ]

        return dataframe[
            series.dt.month
            == month_number
        ].copy()

    raise AggregationError(
        f"Unsupported period format: '{period}'."
    )


def group_and_aggregate(
    group_by: str,
    metric: str,
    aggregation: str = "sum",
    period: str | None = None,
    sort_descending: bool = True,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """
    Group the dataset by a column and aggregate a metric.

    Examples
    --------
    group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
    )

    group_and_aggregate(
        group_by="region",
        metric="revenue",
        aggregation="sum",
        period="2026-09",
    )
    """

    normalized_aggregation = (
        aggregation.strip().lower()
    )

    if normalized_aggregation not in (
        SUPPORTED_AGGREGATIONS
    ):
        raise AggregationError(
            f"Unsupported aggregation '{aggregation}'. "
            "Supported values are: "
            + ", ".join(
                sorted(
                    SUPPORTED_AGGREGATIONS
                )
            )
        )

    try:
        dataframe = load_dataset(
            dataset_path
        )
    except DatasetError as exc:
        raise AggregationError(
            str(exc)
        ) from exc

    if group_by not in dataframe.columns:
        raise AggregationError(
            f"Grouping column '{group_by}' does not exist."
        )

    if metric not in dataframe.columns:
        raise AggregationError(
            f"Metric column '{metric}' does not exist."
        )

    if (
        normalized_aggregation != "count"
        and not pd.api.types.is_numeric_dtype(
            dataframe[metric]
        )
    ):
        raise AggregationError(
            f"Metric '{metric}' must be numeric for "
            f"aggregation '{aggregation}'."
        )

    dataframe = _apply_period_filter(
        dataframe,
        period,
    )

    if dataframe.empty:
        raise AggregationError(
            "No rows remain after applying the requested period filter."
        )

    grouped = dataframe.groupby(
        group_by,
        dropna=False,
    )[metric]

    if normalized_aggregation == "sum":
        aggregated = grouped.sum()

    elif normalized_aggregation == "mean":
        aggregated = grouped.mean()

    elif normalized_aggregation == "median":
        aggregated = grouped.median()

    elif normalized_aggregation == "min":
        aggregated = grouped.min()

    elif normalized_aggregation == "max":
        aggregated = grouped.max()

    elif normalized_aggregation == "count":
        aggregated = grouped.count()

    else:
        raise AggregationError(
            f"Unhandled aggregation: {aggregation}"
        )

    aggregated = aggregated.sort_values(
        ascending=not sort_descending
    )

    results = [
        {
            "group": str(group),
            "value": _safe_value(
                value
            ),
        }
        for group, value
        in aggregated.items()
    ]

    return {
        "group_by": group_by,
        "metric": metric,
        "aggregation": normalized_aggregation,
        "period": period,
        "rows_analyzed": int(
            len(dataframe)
        ),
        "result": results,
    }