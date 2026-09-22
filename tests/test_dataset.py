"""Tests for dataset inspection tools."""

from pathlib import Path

import pandas as pd

from tools.dataset import (
    DatasetError,
    inspect_dataset,
    load_dataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_load_dataset() -> None:
    dataframe = load_dataset(
        DATASET_PATH
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert not dataframe.empty

    assert len(
        dataframe
    ) >= 500

    assert len(
        dataframe
    ) <= 2000


def test_required_columns_exist() -> None:
    dataframe = load_dataset(
        DATASET_PATH
    )

    required_columns = {
        "order_id",
        "order_date",
        "customer_id",
        "product",
        "category",
        "region",
        "quantity",
        "unit_price",
        "discount",
        "revenue",
        "cost",
        "profit",
        "salesperson",
    }

    assert required_columns.issubset(
        dataframe.columns
    )


def test_order_date_is_datetime() -> None:
    dataframe = load_dataset(
        DATASET_PATH
    )

    assert (
        pd.api.types
        .is_datetime64_any_dtype(
            dataframe[
                "order_date"
            ]
        )
    )


def test_inspect_dataset() -> None:
    result = inspect_dataset(
        DATASET_PATH
    )

    assert (
        result["dataset"]
        == "sales.csv"
    )

    assert (
        result["rows"]
        == 1337
    )

    assert (
        result["columns_count"]
        == 13
    )

    assert (
        "order_date"
        in result[
            "date_ranges"
        ]
    )

    assert (
        result[
            "missing_values"
        ]["discount"]
        == 12
    )

    assert (
        result[
            "duplicate_order_ids"
        ]
        == 0
    )

    assert (
        len(
            result[
                "sample_rows"
            ]
        )
        == 5
    )


def test_missing_dataset() -> None:
    missing_path = (
        PROJECT_ROOT
        / "data"
        / "does_not_exist.csv"
    )

    try:
        load_dataset(
            missing_path
        )

    except DatasetError:
        return

    raise AssertionError(
        "DatasetError was not raised."
    )
    