"""Evaluate the five required AI Data Analyst assignment queries.

This evaluation is completely local.

It does NOT:

- call Hugging Face
- call Qwen
- consume inference credits

Instead, it verifies that the analytical tool layer can generate the
evidence required to answer all five mandatory assignment questions.

Run with:

    python -m evaluation.required_queries
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tools.aggregation import group_and_aggregate
from tools.dataset import inspect_dataset
from tools.python_executor import run_python
from tools.statistics import calculate_statistics


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "reports"
)


REQUIRED_QUESTIONS = [
    "Why did sales decrease last month?",
    "Which category generated the most revenue?",
    "Which region has the highest profit margin?",
    "Are there any unusual sales patterns?",
    "Give me a management summary of the dataset.",
]


class EvaluationError(RuntimeError):
    """Raised when the required-query evaluation cannot run."""


def _evaluate_sales_decrease(
    dataset_path: Path,
) -> dict[str, Any]:
    """Evaluate evidence for the monthly sales-decline question."""

    monthly = calculate_statistics(
        operation="percentage_change",
        column="revenue",
        group_by="month",
        dataset_path=dataset_path,
    )

    monthly_results = monthly[
        "result"
    ]

    if len(
        monthly_results
    ) < 2:
        raise EvaluationError(
            "At least two months are required for "
            "month-over-month evaluation."
        )

    previous_month = monthly_results[
        -2
    ]

    latest_month = monthly_results[
        -1
    ]

    category_analysis = run_python(
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
        dataset_path=dataset_path,
    )

    category_rows = (
        category_analysis[
            "result"
        ]["data"]
    )

    largest_decline = (
        category_rows[
            0
        ]
    )

    passed = (
        latest_month[
            "group"
        ]
        == "2026-09"
        and previous_month[
            "group"
        ]
        == "2026-08"
        and latest_month[
            "percentage_change"
        ]
        is not None
        and latest_month[
            "percentage_change"
        ]
        < 0
        and round(
            latest_month[
                "percentage_change"
            ],
            2,
        )
        == -15.13
        and largest_decline[
            "__index__"
        ]
        == "Electronics"
        and largest_decline[
            "revenue_change"
        ]
        < 0
    )

    return {
        "question": REQUIRED_QUESTIONS[0],
        "passed": passed,
        "evidence": {
            "previous_month": (
                previous_month[
                    "group"
                ]
            ),
            "previous_month_revenue": (
                previous_month[
                    "value"
                ]
            ),
            "latest_month": (
                latest_month[
                    "group"
                ]
            ),
            "latest_month_revenue": (
                latest_month[
                    "value"
                ]
            ),
            "percentage_change": (
                latest_month[
                    "percentage_change"
                ]
            ),
            "largest_category_decline": (
                largest_decline[
                    "__index__"
                ]
            ),
            "largest_category_change": (
                largest_decline[
                    "revenue_change"
                ]
            ),
            "largest_category_percentage_change": (
                largest_decline[
                    "percentage_change"
                ]
            ),
        },
    }


def _evaluate_top_revenue_category(
    dataset_path: Path,
) -> dict[str, Any]:
    """Evaluate evidence for highest-revenue category."""

    result = group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
        sort_descending=True,
        dataset_path=dataset_path,
    )

    rows = result[
        "result"
    ]

    top = rows[
        0
    ]

    passed = (
        top[
            "group"
        ]
        == "Electronics"
        and round(
            top[
                "value"
            ],
            2,
        )
        == 796047.76
    )

    return {
        "question": REQUIRED_QUESTIONS[1],
        "passed": passed,
        "evidence": {
            "top_category": (
                top[
                    "group"
                ]
            ),
            "top_revenue": (
                top[
                    "value"
                ]
            ),
            "category_count": len(
                rows
            ),
        },
    }


def _evaluate_profit_margin_region(
    dataset_path: Path,
) -> dict[str, Any]:
    """Evaluate evidence for highest regional profit margin."""

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
        dataset_path=dataset_path,
    )

    output = result[
        "result"
    ]

    rows = output[
        "data"
    ]

    if not rows:
        raise EvaluationError(
            "Regional profit-margin analysis returned no rows."
        )

    top = rows[
        0
    ]

    margins = [
        row[
            "profit_margin"
        ]
        for row in rows
    ]

    passed = (
        output[
            "type"
        ]
        == "dataframe"
        and output[
            "rows_total"
        ]
        == 3
        and "profit_margin"
        in output[
            "columns"
        ]
        and margins
        == sorted(
            margins,
            reverse=True,
        )
        and top[
            "__index__"
        ]
        in {
            "North",
            "South",
            "East",
        }
    )

    return {
        "question": REQUIRED_QUESTIONS[2],
        "passed": passed,
        "evidence": {
            "highest_margin_region": (
                top[
                    "__index__"
                ]
            ),
            "highest_profit_margin": (
                top[
                    "profit_margin"
                ]
            ),
            "regional_results": [
                {
                    "region": row[
                        "__index__"
                    ],
                    "revenue": row[
                        "revenue"
                    ],
                    "profit": row[
                        "profit"
                    ],
                    "profit_margin": row[
                        "profit_margin"
                    ],
                }
                for row in rows
            ],
        },
    }


def _evaluate_unusual_patterns(
    dataset_path: Path,
) -> dict[str, Any]:
    """Evaluate whether unusual patterns can be discovered."""

    unusual_orders = run_python(
        code="""
unusual = (
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

unusual
""",
        dataset_path=dataset_path,
    )

    unusual_output = unusual_orders[
        "result"
    ]

    inspection = inspect_dataset(
        dataset_path=dataset_path,
        sample_rows=0,
    )

    unusual_count = (
        unusual_output[
            "rows_total"
        ]
    )

    missing_discount = (
        inspection[
            "missing_values"
        ][
            "discount"
        ]
    )

    passed = (
        unusual_count
        == 4
        and missing_discount
        == 12
        and all(
            row[
                "quantity"
            ]
            > 5
            for row in unusual_output[
                "data"
            ]
        )
    )

    return {
        "question": REQUIRED_QUESTIONS[3],
        "passed": passed,
        "evidence": {
            "large_quantity_orders": (
                unusual_count
            ),
            "largest_quantity": max(
                row[
                    "quantity"
                ]
                for row in unusual_output[
                    "data"
                ]
            ),
            "missing_discount_values": (
                missing_discount
            ),
        },
    }


def _evaluate_management_summary(
    dataset_path: Path,
) -> dict[str, Any]:
    """Evaluate evidence needed for a management summary."""

    inspection = inspect_dataset(
        dataset_path=dataset_path,
        sample_rows=0,
    )

    revenue = calculate_statistics(
        operation="sum",
        column="revenue",
        dataset_path=dataset_path,
    )

    profit = calculate_statistics(
        operation="sum",
        column="profit",
        dataset_path=dataset_path,
    )

    categories = group_and_aggregate(
        group_by="category",
        metric="revenue",
        aggregation="sum",
        sort_descending=True,
        dataset_path=dataset_path,
    )

    monthly = calculate_statistics(
        operation="percentage_change",
        column="revenue",
        group_by="month",
        dataset_path=dataset_path,
    )

    latest_month = monthly[
        "result"
    ][
        -1
    ]

    passed = (
        inspection[
            "rows"
        ]
        == 1337
        and inspection[
            "columns_count"
        ]
        == 13
        and revenue[
            "result"
        ]
        > 0
        and profit[
            "result"
        ]
        > 0
        and categories[
            "result"
        ][
            0
        ][
            "group"
        ]
        == "Electronics"
        and latest_month[
            "percentage_change"
        ]
        is not None
        and latest_month[
            "percentage_change"
        ]
        < 0
    )

    return {
        "question": REQUIRED_QUESTIONS[4],
        "passed": passed,
        "evidence": {
            "rows": inspection[
                "rows"
            ],
            "columns": inspection[
                "columns_count"
            ],
            "total_revenue": revenue[
                "result"
            ],
            "total_profit": profit[
                "result"
            ],
            "top_revenue_category": (
                categories[
                    "result"
                ][
                    0
                ][
                    "group"
                ]
            ),
            "latest_month": (
                latest_month[
                    "group"
                ]
            ),
            "latest_month_change": (
                latest_month[
                    "percentage_change"
                ]
            ),
            "missing_discount_values": (
                inspection[
                    "missing_values"
                ][
                    "discount"
                ]
            ),
        },
    }


EVALUATORS: list[
    Callable[
        [Path],
        dict[str, Any],
    ]
] = [
    _evaluate_sales_decrease,
    _evaluate_top_revenue_category,
    _evaluate_profit_margin_region,
    _evaluate_unusual_patterns,
    _evaluate_management_summary,
]


def run_required_query_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Run all five required-query evidence evaluations."""

    resolved_dataset = (
        Path(
            dataset_path
        )
        .expanduser()
        .resolve()
    )

    if not resolved_dataset.exists():
        raise EvaluationError(
            f"Dataset not found: {resolved_dataset}"
        )

    results: list[
        dict[str, Any]
    ] = []

    for evaluator in EVALUATORS:
        try:
            result = evaluator(
                resolved_dataset
            )

        except Exception as exc:
            result = {
                "question": (
                    REQUIRED_QUESTIONS[
                        len(
                            results
                        )
                    ]
                ),
                "passed": False,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "evidence": {},
            }

        results.append(
            result
        )

    passed_count = sum(
        1
        for result in results
        if result[
            "passed"
        ]
    )

    total_count = len(
        results
    )

    return {
        "dataset": (
            resolved_dataset.name
        ),
        "dataset_path": str(
            resolved_dataset
        ),
        "passed": (
            passed_count
            == total_count
        ),
        "passed_count": (
            passed_count
        ),
        "total_count": (
            total_count
        ),
        "results": results,
    }


def save_evaluation_report(
    evaluation: dict[str, Any],
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    timestamp: datetime | None = None,
) -> Path:
    """Save required-query evaluation results as Markdown."""

    report_time = (
        timestamp
        if timestamp is not None
        else datetime.now()
    )

    directory = (
        Path(
            output_directory
        )
        .expanduser()
        .resolve()
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "required_query_evaluation_"
        + report_time.strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".md"
    )

    path = (
        directory
        / filename
    )

    lines = [
        "# AI Data Analyst — Required Query Evaluation",
        "",
        f"**Generated:** {report_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Dataset:** `{evaluation['dataset']}`",
        "",
        (
            f"**Overall:** "
            f"{evaluation['passed_count']}/"
            f"{evaluation['total_count']} PASS"
        ),
        "",
        "---",
        "",
    ]

    for index, result in enumerate(
        evaluation[
            "results"
        ],
        start=1,
    ):
        status = (
            "PASS"
            if result[
                "passed"
            ]
            else "FAIL"
        )

        lines.extend(
            [
                (
                    f"## {index}. "
                    f"{result['question']}"
                ),
                "",
                f"**Status:** {status}",
                "",
            ]
        )

        if result.get(
            "error"
        ):
            lines.extend(
                [
                    "**Error**",
                    "",
                    (
                        f"`{result['error']}`"
                    ),
                    "",
                ]
            )

        evidence = result.get(
            "evidence",
            {},
        )

        if evidence:
            lines.extend(
                [
                    "**Evidence**",
                    "",
                ]
            )

            for key, value in (
                evidence.items()
            ):
                readable_key = (
                    key.replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                lines.append(
                    f"- **{readable_key}:** {value}"
                )

            lines.append("")

        lines.extend(
            [
                "---",
                "",
            ]
        )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return path


def print_evaluation_summary(
    evaluation: dict[str, Any],
) -> None:
    """Print a clean terminal evaluation summary."""

    print()
    print("=" * 68)
    print(
        "AI DATA ANALYST - REQUIRED QUERY EVALUATION"
    )
    print("=" * 68)
    print()

    print(
        f"Dataset: {evaluation['dataset']}"
    )

    print()

    for index, result in enumerate(
        evaluation[
            "results"
        ],
        start=1,
    ):
        status = (
            "PASS"
            if result[
                "passed"
            ]
            else "FAIL"
        )

        print(
            f"{index}. {result['question']}"
        )

        print(
            f"   Evidence readiness: {status}"
        )

        if result.get(
            "error"
        ):
            print(
                f"   Error: {result['error']}"
            )

        print()

    print(
        "Overall: "
        f"{evaluation['passed_count']}/"
        f"{evaluation['total_count']} PASS"
    )

    print()


def main() -> None:
    """Run local required-query evaluation."""

    evaluation = (
        run_required_query_evaluation()
    )

    print_evaluation_summary(
        evaluation
    )

    report_path = (
        save_evaluation_report(
            evaluation
        )
    )

    print(
        "Evaluation report saved:"
    )

    print(
        report_path
    )

    print()

    if not evaluation[
        "passed"
    ]:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()