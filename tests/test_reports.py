"""Tests for Markdown report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agent.state import ToolExecution
from tools.reports import (
    ReportError,
    save_analysis_report,
)


def test_report_is_created(
    tmp_path: Path,
) -> None:
    result = save_analysis_report(
        question=(
            "Which category generated the most revenue?"
        ),
        answer=(
            "Electronics generated the most revenue "
            "at $796,047.76."
        ),
        dataset_name="sales.csv",
        output_directory=tmp_path,
        timestamp=datetime(
            2026,
            9,
            23,
            12,
            0,
            0,
        ),
    )

    report_path = Path(
        result["output_path"]
    )

    assert report_path.exists()

    assert (
        report_path.suffix
        == ".md"
    )


def test_report_contains_question_and_answer(
    tmp_path: Path,
) -> None:
    result = save_analysis_report(
        question="What is total revenue?",
        answer="Total revenue is $1,413,404.45.",
        dataset_name="sales.csv",
        output_directory=tmp_path,
        timestamp=datetime(
            2026,
            9,
            23,
            12,
            0,
            0,
        ),
    )

    content = Path(
        result["output_path"]
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "What is total revenue?"
        in content
    )

    assert (
        "$1,413,404.45"
        in content
    )

    assert (
        "sales.csv"
        in content
    )


def test_report_contains_tool_history(
    tmp_path: Path,
) -> None:
    history = [
        ToolExecution(
            name="calculate_statistics",
            arguments={
                "operation": "sum",
                "column": "revenue",
            },
            result={
                "operation": "sum",
                "column": "revenue",
                "result": 1413404.45,
            },
        ),
        ToolExecution(
            name="create_chart",
            arguments={
                "chart_type": "line",
                "x": "month",
                "y": "revenue",
            },
            result={
                "status": "success",
                "filename": (
                    "monthly_revenue_trend.png"
                ),
            },
        ),
    ]

    result = save_analysis_report(
        question="Analyze revenue.",
        answer="Revenue analysis completed.",
        dataset_name="sales.csv",
        tool_history=history,
        output_directory=tmp_path,
        timestamp=datetime(
            2026,
            9,
            23,
            12,
            0,
            0,
        ),
    )

    content = Path(
        result["output_path"]
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "calculate_statistics"
        in content
    )

    assert (
        "create_chart"
        in content
    )

    assert (
        "Completed"
        in content
    )


def test_report_records_tool_error(
    tmp_path: Path,
) -> None:
    history = [
        ToolExecution(
            name="create_chart",
            arguments={
                "x": "category",
                "y": "value",
            },
            result={
                "status": "error",
                "message": (
                    "Column 'value' does not exist."
                ),
            },
        )
    ]

    result = save_analysis_report(
        question="Create a chart.",
        answer=(
            "The requested chart could not be generated."
        ),
        dataset_name="sales.csv",
        tool_history=history,
        output_directory=tmp_path,
        timestamp=datetime(
            2026,
            9,
            23,
            12,
            0,
            0,
        ),
    )

    content = Path(
        result["output_path"]
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "Error"
        in content
    )

    assert (
        "Column 'value' does not exist."
        in content
    )


def test_report_filename_is_safe(
    tmp_path: Path,
) -> None:
    result = save_analysis_report(
        question=(
            "Why did sales decrease last month?"
        ),
        answer="Analysis complete.",
        dataset_name="sales.csv",
        output_directory=tmp_path,
        timestamp=datetime(
            2026,
            9,
            23,
            12,
            30,
            45,
        ),
    )

    assert (
        result["filename"]
        == (
            "20260923_123045_"
            "why_did_sales_decrease_last_month.md"
        )
    )


def test_empty_question_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ReportError
    ):
        save_analysis_report(
            question="",
            answer="Some answer.",
            dataset_name="sales.csv",
            output_directory=tmp_path,
        )


def test_empty_answer_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ReportError
    ):
        save_analysis_report(
            question="Some question?",
            answer="",
            dataset_name="sales.csv",
            output_directory=tmp_path,
        )