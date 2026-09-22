"""Tests for the restricted Python executor."""

from pathlib import Path

import pytest

from tools.python_executor import (
    PythonExecutionError,
    run_python,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_simple_sum() -> None:
    result = run_python(
        code='df["revenue"].sum()',
        dataset_path=DATASET_PATH,
    )

    assert (
        result["status"]
        == "success"
    )

    assert (
        result["result"]
        > 0
    )


def test_groupby_category() -> None:
    result = run_python(
        code=(
            'df.groupby("category")'
            '["revenue"].sum()'
        ),
        dataset_path=DATASET_PATH,
    )

    output = result[
        "result"
    ]

    assert (
        output["type"]
        == "series"
    )

    groups = {
        row["index"]
        for row
        in output["data"]
    }

    assert (
        "Electronics"
        in groups
    )

    assert (
        "Furniture"
        in groups
    )


def test_monthly_analysis() -> None:
    result = run_python(
        code="""
monthly_sales = (
    df.groupby(
        df["order_date"].dt.to_period("M")
    )["revenue"].sum()
)

monthly_sales
""",
        dataset_path=DATASET_PATH,
    )

    output = result[
        "result"
    ]

    assert (
        output["type"]
        == "series"
    )

    assert (
        output["rows_total"]
        == 9
    )

    months = [
        row["index"]
        for row
        in output["data"]
    ]

    assert (
        "2026-01"
        in months
    )

    assert (
        "2026-09"
        in months
    )


def test_assignment_then_final_expression() -> None:
    result = run_python(
        code="""
total_revenue = df["revenue"].sum()
total_profit = df["profit"].sum()

{
    "revenue": total_revenue,
    "profit": total_profit
}
""",
        dataset_path=DATASET_PATH,
    )

    output = result[
        "result"
    ]

    assert (
        output["revenue"]
        > 0
    )

    assert (
        output["profit"]
        > 0
    )


def test_pandas_is_available_without_import() -> None:
    result = run_python(
        code="""
summary = pd.DataFrame(
    {
        "category": ["A", "B"],
        "value": [10, 20]
    }
)

summary
""",
        dataset_path=DATASET_PATH,
    )

    output = result[
        "result"
    ]

    assert (
        output["type"]
        == "dataframe"
    )

    assert (
        output["rows_total"]
        == 2
    )


def test_vectorized_period_comparison() -> None:
    result = run_python(
        code="""
working = df.copy()

working["month"] = (
    working["order_date"]
    .dt.to_period("M")
    .astype(str)
)

comparison = (
    working[
        working["month"].isin(
            ["2026-08", "2026-09"]
        )
    ]
    .groupby(
        ["category", "month"]
    )["revenue"]
    .sum()
    .unstack(
        fill_value=0
    )
)

comparison["percentage_change"] = (
    (
        comparison["2026-09"]
        - comparison["2026-08"]
    )
    / comparison["2026-08"]
    * 100
)

comparison.sort_values(
    "percentage_change"
)
""",
        dataset_path=DATASET_PATH,
    )

    output = result[
        "result"
    ]

    assert (
        output["type"]
        == "dataframe"
    )

    assert (
        output["rows_total"]
        == 3
    )

    assert (
        "percentage_change"
        in output["columns"]
    )


def test_import_is_blocked() -> None:
    with pytest.raises(
        PythonExecutionError
    ):
        run_python(
            code="""
import os
os.listdir(".")
""",
            dataset_path=DATASET_PATH,
        )


def test_open_is_blocked() -> None:
    with pytest.raises(
        PythonExecutionError
    ):
        run_python(
            code=(
                'open("secret.txt", "w")'
            ),
            dataset_path=DATASET_PATH,
        )


def test_for_loop_is_blocked() -> None:
    with pytest.raises(
        PythonExecutionError
    ):
        run_python(
            code="""
values = []

for value in df["revenue"]:
    values.append(value)

values
""",
            dataset_path=DATASET_PATH,
        )


def test_dunder_access_is_blocked() -> None:
    with pytest.raises(
        PythonExecutionError
    ):
        run_python(
            code="df.__class__",
            dataset_path=DATASET_PATH,
        )


def test_dataframe_file_write_is_blocked() -> None:
    with pytest.raises(
        PythonExecutionError
    ):
        run_python(
            code=(
                'df.to_csv("output.csv")'
            ),
            dataset_path=DATASET_PATH,
        )