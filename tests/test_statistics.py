"""Tests for statistical analysis tools."""

from pathlib import Path

import pytest

from tools.statistics import (
    StatisticsError,
    calculate_statistics,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_sum_revenue() -> None:
    result = calculate_statistics(
        operation="sum",
        column="revenue",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["operation"]
        == "sum"
    )

    assert (
        result["column"]
        == "revenue"
    )

    assert (
        result["group_by"]
        is None
    )

    assert (
        result["result"]
        > 0
    )


def test_mean_profit() -> None:
    result = calculate_statistics(
        operation="mean",
        column="profit",
        dataset_path=DATASET_PATH,
    )

    assert (
        isinstance(
            result["result"],
            float,
        )
    )


def test_count_orders() -> None:
    result = calculate_statistics(
        operation="count",
        column="order_id",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["result"]
        == 1337
    )


def test_group_revenue_by_category() -> None:
    result = calculate_statistics(
        operation="sum",
        column="revenue",
        group_by="category",
        dataset_path=DATASET_PATH,
    )

    groups = {
        item["group"]
        for item in result["result"]
    }

    assert (
        "Electronics"
        in groups
    )

    assert (
        "Furniture"
        in groups
    )

    assert (
        "Office Supplies"
        in groups
    )


def test_group_revenue_by_month() -> None:
    result = calculate_statistics(
        operation="sum",
        column="revenue",
        group_by="month",
        dataset_path=DATASET_PATH,
    )

    groups = [
        item["group"]
        for item in result["result"]
    ]

    assert (
        "2026-01"
        in groups
    )

    assert (
        "2026-09"
        in groups
    )

    assert (
        len(groups)
        == 9
    )


def test_percentage_change_by_month() -> None:
    result = calculate_statistics(
        operation=(
            "percentage_change"
        ),
        column="revenue",
        group_by="month",
        dataset_path=DATASET_PATH,
    )

    values = result["result"]

    assert (
        len(values)
        == 9
    )

    assert (
        values[0][
            "percentage_change"
        ]
        is None
    )

    assert (
        values[-1]["group"]
        == "2026-09"
    )


def test_invalid_operation() -> None:
    with pytest.raises(
        StatisticsError
    ):
        calculate_statistics(
            operation="average123",
            column="revenue",
            dataset_path=DATASET_PATH,
        )


def test_invalid_column() -> None:
    with pytest.raises(
        StatisticsError
    ):
        calculate_statistics(
            operation="sum",
            column="does_not_exist",
            dataset_path=DATASET_PATH,
        )


def test_non_numeric_sum() -> None:
    with pytest.raises(
        StatisticsError
    ):
        calculate_statistics(
            operation="sum",
            column="product",
            dataset_path=DATASET_PATH,
        )


def test_percentage_change_needs_grouping() -> None:
    with pytest.raises(
        StatisticsError
    ):
        calculate_statistics(
            operation=(
                "percentage_change"
            ),
            column="revenue",
            dataset_path=DATASET_PATH,
        )