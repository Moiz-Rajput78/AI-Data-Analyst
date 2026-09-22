"""CLI entry point for the AI Data Analyst Agent."""

from __future__ import annotations

from pathlib import Path

from agent.agent import (
    AgentError,
    DataAnalystAgent,
)
from agent.llm import (
    HuggingFaceConfigurationError,
)
from tools.reports import (
    ReportError,
    save_analysis_report,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def print_banner() -> None:
    """Display the CLI header."""

    print()
    print("=" * 68)
    print("AI DATA ANALYST AGENT")
    print("=" * 68)
    print()


def main() -> None:
    """Run the interactive command-line analyst."""

    print_banner()

    dataset_path = DEFAULT_DATASET

    print(
        f"Dataset: {dataset_path.name}"
    )

    print(
        f"Path: {dataset_path}"
    )

    print()
    print(
        "Type your analytical question."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print()

    try:
        agent = DataAnalystAgent(
            dataset_path=dataset_path
        )

    except (
        HuggingFaceConfigurationError,
        AgentError,
    ) as exc:
        print(
            f"Startup error: {exc}"
        )
        return

    while True:
        try:
            question = input(
                "Ask your question:\n> "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print(
                "\nGoodbye."
            )
            break

        if not question:
            print(
                "Please enter a question.\n"
            )
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:
            print(
                "Goodbye."
            )
            break

        print()
        print(
            "Analyzing..."
        )

        tool_history_start = len(
            agent.state.tool_history
        )

        try:
            answer = agent.ask(
                question,
                verbose=True,
            )

        except Exception as exc:
            print()
            print(
                f"Analysis failed: {exc}"
            )
            print()
            continue

        print()
        print("=" * 68)
        print("FINAL REPORT")
        print("=" * 68)
        print()
        print(
            answer
        )
        print()

        current_tool_history = (
            agent.state.tool_history[
                tool_history_start:
            ]
        )

        try:
            report = save_analysis_report(
                question=question,
                answer=answer,
                dataset_name=(
                    dataset_path.name
                ),
                tool_history=(
                    current_tool_history
                ),
            )

            print(
                "Report saved:"
            )

            print(
                report[
                    "output_path"
                ]
            )

        except ReportError as exc:
            print(
                f"Report save failed: {exc}"
            )

        print()


if __name__ == "__main__":
    main()