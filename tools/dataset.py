"""Dataset loading and inspection tools for the AI Data Analyst."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sales.csv"
)


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded or inspected."""


def load_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    """
    Load a CSV dataset safely.

    Date-like columns are automatically converted to datetime when
    every non-null value in the column can be parsed successfully.
    """

    path = Path(dataset_path).expanduser().resolve()

    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise DatasetError(
            f"Dataset path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise DatasetError(
            "Only CSV files are supported."
        )

    try:
        dataframe = pd.read_csv(path)

    except Exception as exc:
        raise DatasetError(
            f"Unable to read dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise DatasetError(
            "The dataset is empty."
        )

    if len(dataframe.columns) == 0:
        raise DatasetError(
            "The dataset contains no columns."
        )

    for column in dataframe.columns:
        normalized_name = column.lower()

        is_possible_date = (
            "date" in normalized_name
            or normalized_name.endswith("_time")
            or normalized_name == "timestamp"
        )

        if not is_possible_date:
            continue

        converted = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

        original_non_null = int(
            dataframe[column].notna().sum()
        )

        converted_non_null = int(
            converted.notna().sum()
        )

        if (
            original_non_null > 0
            and converted_non_null == original_non_null
        ):
            dataframe[column] = converted

    return dataframe


def _safe_value(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values to normal Python values.

    This makes results easier to serialize later when tool results
    are sent back to the Qwen model.
    """

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()

        except (
            ValueError,
            AttributeError,
        ):
            pass

    return value


def _get_date_ranges(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, str | None]]:
    """Return minimum and maximum values for datetime columns."""

    date_ranges: dict[
        str,
        dict[str, str | None],
    ] = {}

    for column in dataframe.columns:
        if not pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        ):
            continue

        series = dataframe[column].dropna()

        if series.empty:
            date_ranges[column] = {
                "min": None,
                "max": None,
            }

            continue

        date_ranges[column] = {
            "min": (
                series.min().isoformat()
            ),
            "max": (
                series.max().isoformat()
            ),
        }

    return date_ranges


def _get_numeric_summary(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Return a compact statistical summary of numeric columns.

    This is useful for dataset understanding only. More detailed
    calculations will later be performed by calculate_statistics().
    """

    summary: dict[
        str,
        dict[str, Any],
    ] = {}

    numeric_columns = (
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    for column in numeric_columns:
        series = dataframe[column].dropna()

        if series.empty:
            summary[column] = {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
            }

            continue

        summary[column] = {
            "count": int(
                series.count()
            ),
            "min": _safe_value(
                series.min()
            ),
            "max": _safe_value(
                series.max()
            ),
            "mean": float(
                series.mean()
            ),
            "median": float(
                series.median()
            ),
        }

    return summary


def _get_sample_rows(
    dataframe: pd.DataFrame,
    sample_rows: int,
) -> list[dict[str, Any]]:
    """Return JSON-safe sample records."""

    sample = dataframe.head(
        sample_rows
    )

    records: list[
        dict[str, Any]
    ] = []

    for _, row in sample.iterrows():
        record: dict[
            str,
            Any,
        ] = {}

        for column in dataframe.columns:
            record[column] = _safe_value(
                row[column]
            )

        records.append(
            record
        )

    return records


def inspect_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sample_rows: int = 5,
) -> dict[str, Any]:
    """
    Inspect a dataset and return structured metadata.

    This tool helps the AI agent understand:

    - dataset size
    - available columns
    - data types
    - missing values
    - duplicate rows
    - date range
    - basic numeric characteristics
    - example records

    It intentionally does not answer business questions. The Qwen
    agent should use this information to decide which analytical
    tool should be called next.
    """

    if sample_rows < 0:
        raise ValueError(
            "sample_rows cannot be negative."
        )

    if sample_rows > 20:
        raise ValueError(
            "sample_rows cannot be greater than 20."
        )

    dataframe = load_dataset(
        dataset_path
    )

    path = (
        Path(dataset_path)
        .expanduser()
        .resolve()
    )

    missing_values = {
        column: int(count)
        for column, count
        in dataframe.isna().sum().items()
    }

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    duplicate_order_ids: int | None = None

    if "order_id" in dataframe.columns:
        duplicate_order_ids = int(
            dataframe[
                "order_id"
            ]
            .duplicated()
            .sum()
        )

    result = {
        "dataset": path.name,

        "dataset_path": str(
            path
        ),

        "rows": int(
            len(dataframe)
        ),

        "columns_count": int(
            len(dataframe.columns)
        ),

        "columns": list(
            dataframe.columns
        ),

        "data_types": {
            column: str(dtype)
            for column, dtype
            in dataframe.dtypes.items()
        },

        "missing_values": (
            missing_values
        ),

        "duplicate_rows": (
            duplicate_rows
        ),

        "duplicate_order_ids": (
            duplicate_order_ids
        ),

        "date_ranges": (
            _get_date_ranges(
                dataframe
            )
        ),

        "numeric_summary": (
            _get_numeric_summary(
                dataframe
            )
        ),

        "sample_rows": (
            _get_sample_rows(
                dataframe,
                sample_rows,
            )
        ),
    }

    return result