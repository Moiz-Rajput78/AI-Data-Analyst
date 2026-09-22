"""Statistical analysis tools for the AI Data Analyst."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    load_dataset,
)


SUPPORTED_OPERATIONS = {
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "std",
    "count",
    "percentage_change",
}


class StatisticsError(RuntimeError):
    """Raised when a statistical calculation cannot be completed."""


def _validate_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Ensure the requested column exists."""

    if column not in dataframe.columns:
        raise StatisticsError(
            f"Column '{column}' does not exist in the dataset."
        )


def _ensure_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Ensure the requested column is numeric."""

    _validate_column(
        dataframe,
        column,
    )

    if not pd.api.types.is_numeric_dtype(
        dataframe[column]
    ):
        raise StatisticsError(
            f"Column '{column}' must be numeric for this operation."
        )


def _safe_python_value(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values to normal Python values.

    This is useful because tool results will later be serialized
    and sent back to the Qwen model.
    """

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()

        except (
            AttributeError,
            ValueError,
        ):
            pass

    return value


def _prepare_group_column(
    dataframe: pd.DataFrame,
    group_by: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prepare supported grouping shortcuts.

    Supported special values:

    - month
    - year
    - quarter

    These use the first datetime column found in the dataset.
    """

    dataframe = dataframe.copy()

    normalized = group_by.strip().lower()

    if normalized not in {
        "month",
        "year",
        "quarter",
    }:
        _validate_column(
            dataframe,
            group_by,
        )

        return dataframe, group_by

    datetime_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        )
    ]

    if not datetime_columns:
        raise StatisticsError(
            f"Cannot group by '{group_by}' because "
            "the dataset has no datetime column."
        )

    date_column = datetime_columns[0]

    temporary_column = (
        f"__group_{normalized}"
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


def _aggregate_series(
    series: pd.Series,
    operation: str,
) -> Any:
    """Perform one supported statistical aggregation."""

    if operation == "mean":
        return series.mean()

    if operation == "median":
        return series.median()

    if operation == "sum":
        return series.sum()

    if operation == "min":
        return series.min()

    if operation == "max":
        return series.max()

    if operation == "std":
        return series.std()

    if operation == "count":
        return series.count()

    raise StatisticsError(
        f"Unsupported operation: {operation}"
    )


def calculate_statistics(
    operation: str,
    column: str,
    group_by: str | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """
    Perform a statistical calculation on the dataset.

    Parameters
    ----------
    operation:
        Supported values:

        - mean
        - median
        - sum
        - min
        - max
        - std
        - count
        - percentage_change

    column:
        Column on which to perform the calculation.

    group_by:
        Optional grouping column.

        Special shortcuts:

        - month
        - year
        - quarter

    dataset_path:
        Path to the CSV dataset.

    Examples
    --------
    calculate_statistics(
        operation="sum",
        column="revenue",
    )

    calculate_statistics(
        operation="mean",
        column="profit",
        group_by="region",
    )

    calculate_statistics(
        operation="percentage_change",
        column="revenue",
        group_by="month",
    )
    """

    normalized_operation = (
        operation
        .strip()
        .lower()
    )

    if normalized_operation not in (
        SUPPORTED_OPERATIONS
    ):
        raise StatisticsError(
            "Unsupported operation "
            f"'{operation}'. "
            "Supported operations are: "
            + ", ".join(
                sorted(
                    SUPPORTED_OPERATIONS
                )
            )
        )

    try:
        dataframe = load_dataset(
            dataset_path
        )

    except DatasetError as exc:
        raise StatisticsError(
            str(exc)
        ) from exc

    _validate_column(
        dataframe,
        column,
    )

    if normalized_operation != "count":
        _ensure_numeric_column(
            dataframe,
            column,
        )

    if group_by is None:
        if normalized_operation == (
            "percentage_change"
        ):
            raise StatisticsError(
                "percentage_change requires group_by."
            )

        value = _aggregate_series(
            dataframe[column],
            normalized_operation,
        )

        return {
            "operation": (
                normalized_operation
            ),
            "column": column,
            "group_by": None,
            "result": (
                _safe_python_value(
                    value
                )
            ),
        }

    working_dataframe, (
        grouping_column
    ) = _prepare_group_column(
        dataframe,
        group_by,
    )

    grouped = (
        working_dataframe
        .groupby(
            grouping_column,
            dropna=False,
        )[column]
    )

    if normalized_operation == (
        "percentage_change"
    ):
        grouped_values = (
            grouped.sum()
        )

        percentage_changes = (
            grouped_values
            .pct_change()
            * 100
        )

        results: list[
            dict[str, Any]
        ] = []

        for group, value in (
            grouped_values.items()
        ):
            change_value = (
                percentage_changes.loc[
                    group
                ]
            )

            results.append(
                {
                    "group": str(group),
                    "value": (
                        _safe_python_value(
                            value
                        )
                    ),
                    "percentage_change": (
                        _safe_python_value(
                            change_value
                        )
                    ),
                }
            )

        return {
            "operation": (
                normalized_operation
            ),
            "column": column,
            "group_by": group_by,
            "aggregation_used": "sum",
            "result": results,
        }

    aggregated = grouped.apply(
        lambda series: (
            _aggregate_series(
                series,
                normalized_operation,
            )
        )
    )

    results = [
        {
            "group": str(group),
            "value": (
                _safe_python_value(
                    value
                )
            ),
        }
        for group, value
        in aggregated.items()
    ]

    return {
        "operation": (
            normalized_operation
        ),
        "column": column,
        "group_by": group_by,
        "result": results,
    }