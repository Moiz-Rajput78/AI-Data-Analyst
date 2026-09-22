"""Dataset filtering tools for the AI Data Analyst."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    load_dataset,
)


SUPPORTED_OPERATORS = {
    "equals",
    "not_equals",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "contains",
}


class FilterError(RuntimeError):
    """Raised when a dataset filter cannot be applied."""


def _safe_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""

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


def filter_dataset(
    column: str,
    operator: str,
    value: Any,
    limit: int = 20,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """
    Filter the dataset using one condition.

    Supported operators:

    - equals
    - not_equals
    - greater_than
    - greater_than_or_equal
    - less_than
    - less_than_or_equal
    - contains
    """

    normalized_operator = (
        operator.strip().lower()
    )

    if normalized_operator not in (
        SUPPORTED_OPERATORS
    ):
        raise FilterError(
            f"Unsupported operator '{operator}'. "
            "Supported operators are: "
            + ", ".join(
                sorted(
                    SUPPORTED_OPERATORS
                )
            )
        )

    if limit < 1 or limit > 100:
        raise FilterError(
            "limit must be between 1 and 100."
        )

    try:
        dataframe = load_dataset(
            dataset_path
        )
    except DatasetError as exc:
        raise FilterError(
            str(exc)
        ) from exc

    if column not in dataframe.columns:
        raise FilterError(
            f"Column '{column}' does not exist."
        )

    series = dataframe[column]

    if normalized_operator == "equals":
        mask = series == value

    elif normalized_operator == "not_equals":
        mask = series != value

    elif normalized_operator == "greater_than":
        mask = series > value

    elif normalized_operator == (
        "greater_than_or_equal"
    ):
        mask = series >= value

    elif normalized_operator == "less_than":
        mask = series < value

    elif normalized_operator == (
        "less_than_or_equal"
    ):
        mask = series <= value

    elif normalized_operator == "contains":
        mask = (
            series.astype(str)
            .str.contains(
                str(value),
                case=False,
                na=False,
            )
        )

    else:
        raise FilterError(
            f"Unhandled operator: {operator}"
        )

    filtered = dataframe[
        mask
    ].copy()

    preview = filtered.head(
        limit
    )

    records: list[
        dict[str, Any]
    ] = []

    for _, row in preview.iterrows():
        records.append(
            {
                column_name: _safe_value(
                    row[column_name]
                )
                for column_name
                in dataframe.columns
            }
        )

    return {
        "column": column,
        "operator": normalized_operator,
        "value": value,
        "matching_rows": int(
            len(filtered)
        ),
        "returned_rows": int(
            len(preview)
        ),
        "rows": records,
    }