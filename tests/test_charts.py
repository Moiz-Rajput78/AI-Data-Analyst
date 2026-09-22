"""Tests for chart generation tools."""

from pathlib import Path

import pytest

from tools.charts import (
    ChartError,
    create_chart,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_line_chart(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="line",
        x="month",
        y="revenue",
        title="Monthly Revenue",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    chart_path = Path(
        result["output_path"]
    )

    assert chart_path.exists()

    assert (
        chart_path.suffix
        == ".png"
    )

    assert (
        result["chart_type"]
        == "line"
    )

    assert (
        result["points"]
        == 9
    )


def test_bar_chart(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="bar",
        x="category",
        y="revenue",
        title="Revenue by Category",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    assert Path(
        result["output_path"]
    ).exists()

    assert (
        result["points"]
        == 3
    )


def test_pie_chart(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="pie",
        x="region",
        y="revenue",
        title="Revenue by Region",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    assert Path(
        result["output_path"]
    ).exists()

    assert (
        result["chart_type"]
        == "pie"
    )

    assert (
        result["points"]
        == 3
    )


def test_scatter_chart(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="scatter",
        x="quantity",
        y="revenue",
        title="Quantity vs Revenue",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    assert Path(
        result["output_path"]
    ).exists()

    assert (
        result["chart_type"]
        == "scatter"
    )

    assert (
        result["points"]
        > 0
    )


def test_period_filter(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="bar",
        x="category",
        y="revenue",
        title="September Revenue by Category",
        period="2026-09",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    assert Path(
        result["output_path"]
    ).exists()

    assert (
        result["period"]
        == "2026-09"
    )


def test_custom_output_name(
    tmp_path: Path,
) -> None:
    result = create_chart(
        chart_type="bar",
        x="region",
        y="profit",
        title="Regional Profit",
        output_name="regional_profit_chart.png",
        dataset_path=DATASET_PATH,
        output_directory=tmp_path,
    )

    assert (
        result["filename"]
        == "regional_profit_chart.png"
    )


def test_invalid_chart_type(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ChartError
    ):
        create_chart(
            chart_type="histogram",
            x="category",
            y="revenue",
            title="Invalid Chart",
            dataset_path=DATASET_PATH,
            output_directory=tmp_path,
        )


def test_scatter_requires_numeric_x(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ChartError
    ):
        create_chart(
            chart_type="scatter",
            x="category",
            y="revenue",
            title="Invalid Scatter",
            dataset_path=DATASET_PATH,
            output_directory=tmp_path,
        )