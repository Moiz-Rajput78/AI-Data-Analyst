"""Tests for grouping and aggregation tools."""

from pathlib import Path

import pytest

from tools.aggregation import (
    AggregationError,
    group_and_aggregate,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_group_revenue_by_category() -> None:
    result = group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
        dataset_path=DATASET_PATH,
    )

    groups = {
        item["group"]
        for item in result["result"]
    }

    assert "Electronics" in groups
    assert "Furniture" in groups
    assert "Office Supplies" in groups


def test_group_revenue_by_region() -> None:
    result = group_and_aggregate(
        group_by="region",
        metric="revenue",
        aggregation="sum",
        dataset_path=DATASET_PATH,
    )

    assert (
        len(
            result["result"]
        )
        == 3
    )


def test_september_category_analysis() -> None:
    result = group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
        period="2026-09",
        dataset_path=DATASET_PATH,
    )

    assert (
        result["period"]
        == "2026-09"
    )

    assert (
        result["rows_analyzed"]
        > 0
    )


def test_invalid_group_column() -> None:
    with pytest.raises(
        AggregationError
    ):
        group_and_aggregate(
            group_by="missing_column",
            metric="revenue",
            dataset_path=DATASET_PATH,
        )


def test_invalid_metric() -> None:
    with pytest.raises(
        AggregationError
    ):
        group_and_aggregate(
            group_by="category",
            metric="missing_metric",
            dataset_path=DATASET_PATH,
        )


def test_non_numeric_metric_rejected() -> None:
    with pytest.raises(
        AggregationError
    ):
        group_and_aggregate(
            group_by="region",
            metric="product",
            aggregation="sum",
            dataset_path=DATASET_PATH,
        )