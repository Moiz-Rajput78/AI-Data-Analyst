"""Tests for agent final-answer evidence validation."""

from pathlib import Path

from agent.agent import DataAnalystAgent
from agent.state import AgentState, ToolExecution


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


def build_agent_without_api_init() -> DataAnalystAgent:
    """
    Construct an agent instance without initializing Qwen.

    These tests validate only local final-answer checking.
    """

    agent = object.__new__(
        DataAnalystAgent
    )

    agent.dataset_path = DATASET_PATH
    agent.state = AgentState()

    return agent


def test_detects_fake_chart() -> None:
    agent = (
        build_agent_without_api_init()
    )

    agent.state.tool_history.append(
        ToolExecution(
            name="group_and_aggregate",
            arguments={},
            result={
                "result": [
                    {
                        "group": "Electronics",
                        "value": 796047.76,
                    }
                ]
            },
        )
    )

    problems = (
        agent._validate_final_answer(
            (
                "Electronics generated $796,047.76.\n"
                "Charts:\n"
                "- category_revenue_comparison.png"
            ),
            question_tool_start=0,
        )
    )

    assert problems

    assert any(
        (
            "not generated"
            in problem.lower()
            or "not actually generated"
            in problem.lower()
        )
        for problem in problems
    )


def test_real_chart_is_allowed() -> None:
    agent = (
        build_agent_without_api_init()
    )

    agent.state.tool_history.append(
        ToolExecution(
            name="create_chart",
            arguments={},
            result={
                "status": "success",
                "filename": (
                    "category_revenue_comparison.png"
                ),
            },
        )
    )

    problems = (
        agent._validate_final_answer(
            (
                "Charts:\n"
                "- category_revenue_comparison.png"
            ),
            question_tool_start=0,
        )
    )

    assert not problems


def test_detects_unsupported_percentage() -> None:
    agent = (
        build_agent_without_api_init()
    )

    agent.state.tool_history.append(
        ToolExecution(
            name="group_and_aggregate",
            arguments={},
            result={
                "result": [
                    {
                        "group": "Electronics",
                        "value": 796047.76,
                    },
                    {
                        "group": "Furniture",
                        "value": 511179.56,
                    },
                ]
            },
        )
    )

    problems = (
        agent._validate_final_answer(
            (
                "Electronics contributed "
                "53% of revenue."
            ),
            question_tool_start=0,
        )
    )

    assert problems

    assert any(
        "percentage"
        in problem.lower()
        for problem in problems
    )


def test_supported_percentage_is_allowed() -> None:
    agent = (
        build_agent_without_api_init()
    )

    agent.state.tool_history.append(
        ToolExecution(
            name="run_python",
            arguments={},
            result={
                "status": "success",
                "result": {
                    "electronics_share": 56.3201,
                },
            },
        )
    )

    problems = (
        agent._validate_final_answer(
            (
                "Electronics contributed "
                "56.32% of revenue."
            ),
            question_tool_start=0,
        )
    )

    assert not problems


def test_percentage_rounding_is_allowed() -> None:
    agent = (
        build_agent_without_api_init()
    )

    agent.state.tool_history.append(
        ToolExecution(
            name="calculate_statistics",
            arguments={},
            result={
                "percentage_change": (
                    -15.131117523527692
                )
            },
        )
    )

    problems = (
        agent._validate_final_answer(
            (
                "Revenue decreased "
                "by 15.13%."
            ),
            question_tool_start=0,
        )
    )

    assert not problems