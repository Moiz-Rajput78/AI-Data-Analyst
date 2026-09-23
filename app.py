"""Streamlit interface for the AI Data Analyst Agent."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

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

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)

CHART_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "charts"
)


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)


def initialize_agent() -> DataAnalystAgent:
    """Create and cache the AI Data Analyst agent."""

    try:
        return DataAnalystAgent(
            dataset_path=DATASET_PATH
        )

    except (
        HuggingFaceConfigurationError,
        AgentError,
    ) as exc:
        st.error(
            f"Unable to initialize the AI agent: {exc}"
        )

        st.stop()


@st.cache_resource
def get_agent() -> DataAnalystAgent:
    """Return one agent instance for the Streamlit session."""

    return initialize_agent()


def get_latest_chart(
    previous_files: set[Path],
) -> Path | None:
    """Return a newly generated chart, if one exists."""

    if not CHART_DIRECTORY.exists():
        return None

    current_files = {
        path
        for path in CHART_DIRECTORY.glob(
            "*.png"
        )
        if path.is_file()
    }

    new_files = (
        current_files
        - previous_files
    )

    if not new_files:
        return None

    return max(
        new_files,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )


def main() -> None:
    """Render the Streamlit application."""

    st.title(
        "📊 AI Data Analyst Agent"
    )

    st.caption(
        "Ask business questions about the sales dataset. "
        "The AI dynamically selects analytical tools, "
        "examines evidence, and produces a grounded report."
    )

    st.divider()

    left_column, right_column = (
        st.columns(
            [
                2,
                1,
            ]
        )
    )

    with left_column:
        st.subheader(
            "Ask a Question"
        )

        question = st.text_area(
            "Analytical question",
            placeholder=(
                "Example: Why did sales decrease last month?"
            ),
            height=120,
        )

        analyze_button = st.button(
            "Analyze",
            type="primary",
            use_container_width=True,
        )

    with right_column:
        st.subheader(
            "Dataset"
        )

        st.write(
            f"**File:** `{DATASET_PATH.name}`"
        )

        if DATASET_PATH.exists():
            size_kb = (
                DATASET_PATH.stat().st_size
                / 1024
            )

            st.write(
                f"**Size:** {size_kb:.1f} KB"
            )

            st.success(
                "Dataset loaded"
            )

        else:
            st.error(
                "Dataset file not found."
            )

    st.divider()

    if not analyze_button:
        st.info(
            "Enter a question above and click Analyze."
        )

        st.markdown(
            """
### Example questions

- Why did sales decrease last month?
- Which category generated the most revenue?
- Which region has the highest profit margin?
- Are there any unusual sales patterns?
- Give me a management summary of the dataset.
"""
        )

        return

    question = question.strip()

    if not question:
        st.warning(
            "Please enter an analytical question."
        )

        return

    agent = get_agent()

    tool_history_start = len(
        agent.state.tool_history
    )

    previous_chart_files: set[
        Path
    ] = set()

    if CHART_DIRECTORY.exists():
        previous_chart_files = {
            path
            for path in CHART_DIRECTORY.glob(
                "*.png"
            )
            if path.is_file()
        }

    st.subheader(
        "Analysis"
    )

    with st.spinner(
        "Analyzing the dataset..."
    ):
        try:
            answer = agent.ask(
                question,
                verbose=False,
            )

        except Exception as exc:
            st.error(
                "The analysis could not be completed."
            )

            st.code(
                str(exc)
            )

            return

    st.success(
        "Analysis completed"
    )

    st.markdown(
        "### Final Report"
    )

    st.markdown(
        answer
    )

    current_tool_history = (
        agent.state.tool_history[
            tool_history_start:
        ]
    )

    if current_tool_history:
        with st.expander(
            "Analytical tools used",
            expanded=False,
        ):
            for index, execution in enumerate(
                current_tool_history,
                start=1,
            ):
                st.markdown(
                    f"**{index}. `{execution.name}`**"
                )

                st.json(
                    execution.arguments
                )

                if (
                    isinstance(
                        execution.result,
                        dict,
                    )
                    and execution.result.get(
                        "status"
                    )
                    == "error"
                ):
                    st.error(
                        execution.result.get(
                            "message",
                            "Tool returned an error.",
                        )
                    )

                else:
                    st.success(
                        "Completed"
                    )

    latest_chart = get_latest_chart(
        previous_chart_files
    )

    if (
        latest_chart is not None
        and latest_chart.exists()
    ):
        st.markdown(
            "### Generated Chart"
        )

        st.image(
            str(
                latest_chart
            ),
            use_container_width=True,
        )

        st.caption(
            latest_chart.name
        )

    try:
        report = save_analysis_report(
            question=question,
            answer=answer,
            dataset_name=(
                DATASET_PATH.name
            ),
            tool_history=(
                current_tool_history
            ),
        )

        st.caption(
            "Report saved locally as "
            f"`{Path(report['output_path']).name}`"
        )

    except ReportError as exc:
        st.warning(
            f"Analysis succeeded, but the report "
            f"could not be saved: {exc}"
        )


if __name__ == "__main__":
    main()