"""Tests for dataset filtering tools."""

from pathlib import Path

import pytest

from tools.filters import (
    FilterError,
    filter_dataset,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_filter_region_equals() -> None:
    result = filter_dataset(
        column="region",
        operator="equals",
        value="North",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["matching_rows"]
        > 0
    )

    for row in result["rows"]:
        assert (
            row["region"]
            == "North"
        )


def test_filter_category_equals() -> None:
    result = filter_dataset(
        column="category",
        operator="equals",
        value="Electronics",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["matching_rows"]
        > 0
    )


def test_filter_quantity_greater_than() -> None:
    result = filter_dataset(
        column="quantity",
        operator="greater_than",
        value=5,
        dataset_path=DATASET_PATH,
    )

    assert (
        result["matching_rows"]
        > 0
    )

    for row in result["rows"]:
        assert (
            row["quantity"]
            > 5
        )


def test_filter_product_contains() -> None:
    result = filter_dataset(
        column="product",
        operator="contains",
        value="phone",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["matching_rows"]
        > 0
    )


def test_invalid_column() -> None:
    with pytest.raises(
        FilterError
    ):
        filter_dataset(
            column="missing",
            operator="equals",
            value="test",
            dataset_path=DATASET_PATH,
        )


def test_invalid_operator() -> None:
    with pytest.raises(
        FilterError
    ):
        filter_dataset(
            column="region",
            operator="something_invalid",
            value="North",
            dataset_path=DATASET_PATH,
        )