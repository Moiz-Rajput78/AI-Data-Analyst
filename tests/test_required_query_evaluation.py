"""Tests for the local required-query evaluation runner."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from evaluation.required_queries import (
    REQUIRED_QUESTIONS,
    run_required_query_evaluation,
    save_evaluation_report,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def test_evaluation_runs_all_required_questions() -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    assert (
        evaluation[
            "total_count"
        ]
        == 5
    )

    assert (
        len(
            evaluation[
                "results"
            ]
        )
        == 5
    )


def test_all_required_questions_pass() -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    assert (
        evaluation[
            "passed"
        ]
        is True
    )

    assert (
        evaluation[
            "passed_count"
        ]
        == 5
    )


def test_evaluation_contains_expected_questions() -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    actual_questions = [
        result[
            "question"
        ]
        for result in evaluation[
            "results"
        ]
    ]

    assert (
        actual_questions
        == REQUIRED_QUESTIONS
    )


def test_each_evaluation_has_evidence() -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    for result in evaluation[
        "results"
    ]:
        assert (
            result[
                "passed"
            ]
            is True
        )

        assert result[
            "evidence"
        ]


def test_evaluation_report_is_created(
    tmp_path: Path,
) -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    report_path = (
        save_evaluation_report(
            evaluation,
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
    )

    assert (
        report_path.exists()
    )

    assert (
        report_path.name
        == (
            "required_query_evaluation_"
            "20260923_120000.md"
        )
    )


def test_evaluation_report_contains_summary(
    tmp_path: Path,
) -> None:
    evaluation = (
        run_required_query_evaluation(
            DATASET_PATH
        )
    )

    report_path = (
        save_evaluation_report(
            evaluation,
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
    )

    content = report_path.read_text(
        encoding="utf-8"
    )

    assert (
        "5/5 PASS"
        in content
    )

    assert (
        "Why did sales decrease last month?"
        in content
    )

    assert (
        "Which category generated the most revenue?"
        in content
    )

    assert (
        "Which region has the highest profit margin?"
        in content
    )

    assert (
        "Are there any unusual sales patterns?"
        in content
    )

    assert (
        "Give me a management summary of the dataset."
        in content
    )