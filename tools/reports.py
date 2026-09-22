"""Report-saving utilities for the AI Data Analyst."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "reports"
)


class ReportError(RuntimeError):
    """Raised when an analysis report cannot be saved."""


def _safe_filename(
    text: str,
) -> str:
    """Convert question text into a filesystem-safe filename."""

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        text.strip().lower(),
    )

    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = "analysis_report"

    return cleaned[:70]


def _format_tool_history(
    tool_history: list[Any],
) -> str:
    """Create a readable Markdown summary of tool executions."""

    if not tool_history:
        return "No analytical tools were used."

    lines: list[str] = []

    for index, execution in enumerate(
        tool_history,
        start=1,
    ):
        name = getattr(
            execution,
            "name",
            "unknown_tool",
        )

        arguments = getattr(
            execution,
            "arguments",
            {},
        )

        result = getattr(
            execution,
            "result",
            None,
        )

        lines.append(
            f"### {index}. `{name}`"
        )

        lines.append("")
        lines.append("**Arguments**")
        lines.append("")
        lines.append("```text")
        lines.append(
            repr(arguments)
        )
        lines.append("```")
        lines.append("")

        if (
            isinstance(result, dict)
            and result.get("status") == "error"
        ):
            lines.append(
                f"**Status:** Error — {result.get('message', 'Unknown error')}"
            )
        else:
            lines.append(
                "**Status:** Completed"
            )

        lines.append("")

    return "\n".join(
        lines
    )


def save_analysis_report(
    question: str,
    answer: str,
    *,
    dataset_name: str,
    tool_history: list[Any] | None = None,
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """
    Save one completed analysis as a Markdown report.

    Parameters
    ----------
    question:
        User's analytical question.

    answer:
        Final evidence-based answer produced by the agent.

    dataset_name:
        Name of the dataset used for analysis.

    tool_history:
        Tool executions used for the current question.

    output_directory:
        Directory where reports are saved.

    timestamp:
        Optional timestamp, mainly useful for deterministic tests.
    """

    question = question.strip()
    answer = answer.strip()
    dataset_name = dataset_name.strip()

    if not question:
        raise ReportError(
            "Question cannot be empty."
        )

    if not answer:
        raise ReportError(
            "Answer cannot be empty."
        )

    if not dataset_name:
        raise ReportError(
            "Dataset name cannot be empty."
        )

    report_time = (
        timestamp
        if timestamp is not None
        else datetime.now()
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp_text = (
        report_time.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    filename = (
        f"{timestamp_text}_"
        f"{_safe_filename(question)}"
        ".md"
    )

    output_path = (
        output_directory
        / filename
    )

    tool_history = (
        tool_history
        if tool_history is not None
        else []
    )

    report_content = (
        "# AI Data Analyst Report\n\n"
        f"**Generated:** "
        f"{report_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"**Dataset:** `{dataset_name}`\n\n"
        "---\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "---\n\n"
        "## Final Analysis\n\n"
        f"{answer}\n\n"
        "---\n\n"
        "## Analytical Tools Used\n\n"
        f"{_format_tool_history(tool_history)}\n"
    )

    try:
        output_path.write_text(
            report_content,
            encoding="utf-8",
        )

    except OSError as exc:
        raise ReportError(
            f"Unable to save report: {exc}"
        ) from exc

    if not output_path.exists():
        raise ReportError(
            "Report generation completed but the file was not created."
        )

    return {
        "status": "success",
        "filename": output_path.name,
        "output_path": str(
            output_path
        ),
        "dataset": dataset_name,
        "question": question,
    }