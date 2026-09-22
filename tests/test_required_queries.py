"""Local evaluation tests for the five required assignment questions.

These tests do not call Hugging Face or Qwen.

Their purpose is to verify that the analytical tool layer can produce
the evidence required by each mandatory project question.
"""

from __future__ import annotations

from pathlib import Path

from tools.aggregation import group_and_aggregate
from tools.dataset import inspect_dataset
from tools.python_executor import run_python
from tools.statistics import calculate_statistics


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_required_query_1_why_sales_decreased_last_month() -> None:
    """
    Required question:

    "Why did sales decrease last month?"

    Verify that:
    - month-over-month revenue change can be calculated;
    - the latest month is September 2026;
    - September revenue declined from August;
    - Electronics is the largest category-level contributor
      to the decline.
    """

    monthly_change = calculate_statistics(
        operation="percentage_change",
        column="revenue",
        group_by="month",
        dataset_path=DATASET_PATH,
    )

    results = monthly_change[
        "result"
    ]

    assert len(
        results
    ) == 9

    august = next(
        row
        for row in results
        if row["group"] == "2026-08"
    )

    september = next(
        row
        for row in results
        if row["group"] == "2026-09"
    )

    assert (
        september["value"]
        < august["value"]
    )

    assert (
        september[
            "percentage_change"
        ]
        < 0
    )

    assert (
        round(
            september[
                "percentage_change"
            ],
            2,
        )
        == -15.13
    )

    category_comparison = run_python(
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

comparison["revenue_change"] = (
    comparison["2026-09"]
    - comparison["2026-08"]
)

comparison["percentage_change"] = (
    comparison["revenue_change"]
    / comparison["2026-08"]
    * 100
)

comparison.sort_values(
    "revenue_change"
)
""",
        dataset_path=DATASET_PATH,
    )

    output = category_comparison[
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

    largest_decline = (
        output["data"][0]
    )

    assert (
        largest_decline[
            "__index__"
        ]
        == "Electronics"
    )

    assert (
        largest_decline[
            "revenue_change"
        ]
        < 0
    )

    assert (
        round(
            largest_decline[
                "percentage_change"
            ],
            2,
        )
        == -25.02
    )


def test_required_query_2_category_with_most_revenue() -> None:
    """
    Required question:

    "Which category generated the most revenue?"
    """

    result = group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
        sort_descending=True,
        dataset_path=DATASET_PATH,
    )

    categories = result[
        "result"
    ]

    assert len(
        categories
    ) == 3

    top_category = categories[
        0
    ]

    assert (
        top_category[
            "group"
        ]
        == "Electronics"
    )

    assert (
        round(
            top_category[
                "value"
            ],
            2,
        )
        == 796047.76
    )

    assert (
        categories[0]["value"]
        > categories[1]["value"]
        > categories[2]["value"]
    )


def test_required_query_3_region_with_highest_profit_margin() -> None:
    """
    Required question:

    "Which region has the highest profit margin?"

    Profit margin is:

        total profit / total revenue * 100

    The calculation must be performed by Pandas rather than mentally
    by the model.
    """

    result = run_python(
        code="""
regional = (
    df.groupby("region")[
        ["revenue", "profit"]
    ]
    .sum()
)

regional["profit_margin"] = (
    regional["profit"]
    / regional["revenue"]
    * 100
)

regional.sort_values(
    "profit_margin",
    ascending=False
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
        "profit_margin"
        in output["columns"]
    )

    rows = output[
        "data"
    ]

    margins = [
        row[
            "profit_margin"
        ]
        for row in rows
    ]

    assert all(
        margin > 0
        for margin in margins
    )

    assert (
        margins
        == sorted(
            margins,
            reverse=True,
        )
    )

    highest_margin_region = (
        rows[0]["__index__"]
    )

    assert highest_margin_region in {
        "North",
        "South",
        "East",
    }


def test_required_query_4_unusual_sales_patterns() -> None:
    """
    Required question:

    "Are there any unusual sales patterns?"

    The generated dataset intentionally contains:

    - unusually large quantity orders;
    - missing discount values.

    Verify that the tools can discover them from the data.
    """

    quantity_outliers = run_python(
        code="""
unusual_orders = (
    df[
        df["quantity"] > 5
    ][
        [
            "order_id",
            "order_date",
            "product",
            "category",
            "region",
            "quantity",
            "revenue",
        ]
    ]
    .sort_values(
        "quantity",
        ascending=False
    )
)

unusual_orders
""",
        dataset_path=DATASET_PATH,
    )

    output = quantity_outliers[
        "result"
    ]

    assert (
        output["type"]
        == "dataframe"
    )

    assert (
        output["rows_total"]
        == 4
    )

    assert all(
        row["quantity"] > 5
        for row
        in output["data"]
    )

    inspection = inspect_dataset(
        dataset_path=DATASET_PATH,
        sample_rows=0,
    )

    assert (
        inspection[
            "missing_values"
        ]["discount"]
        == 12
    )


def test_required_query_5_management_summary_capabilities() -> None:
    """
    Required question:

    "Give me a management summary of the dataset."

    Verify that the system can obtain the major facts needed for
    a concise management-level summary.
    """

    inspection = inspect_dataset(
        dataset_path=DATASET_PATH,
        sample_rows=0,
    )

    total_revenue = calculate_statistics(
        operation="sum",
        column="revenue",
        dataset_path=DATASET_PATH,
    )

    total_profit = calculate_statistics(
        operation="sum",
        column="profit",
        dataset_path=DATASET_PATH,
    )

    category_revenue = (
        group_and_aggregate(
            group_by="category",
            metric="revenue",
            aggregation="sum",
            sort_descending=True,
            dataset_path=DATASET_PATH,
        )
    )

    monthly_revenue = (
        calculate_statistics(
            operation="percentage_change",
            column="revenue",
            group_by="month",
            dataset_path=DATASET_PATH,
        )
    )

    assert (
        inspection["rows"]
        == 1337
    )

    assert (
        inspection[
            "columns_count"
        ]
        == 13
    )

    assert (
        total_revenue[
            "result"
        ]
        > 0
    )

    assert (
        total_profit[
            "result"
        ]
        > 0
    )

    assert (
        category_revenue[
            "result"
        ][0]["group"]
        == "Electronics"
    )

    monthly_results = (
        monthly_revenue[
            "result"
        ]
    )

    assert (
        monthly_results[
            -1
        ]["group"]
        == "2026-09"
    )

    assert (
        monthly_results[
            -1
        ][
            "percentage_change"
        ]
        < 0
    )

    assert (
        inspection[
            "missing_values"
        ]["discount"]
        == 12
    )