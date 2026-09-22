"""Tests for the dynamic AI Data Analyst agent loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.agent import (
    MAX_TOOL_CALLS_PER_QUESTION,
    DataAnalystAgent,
)
from agent.state import AgentState


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "sales.csv"
)


class FakeQwenClient:
    """Deterministic fake Qwen client for local agent-loop tests."""

    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.responses = responses
        self.call_index = 0
        self.received_messages: list[
            list[dict[str, Any]]
        ] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1600,
    ) -> Any:
        """
        Return the next prepared model response.

        The received conversation is stored so tests can verify
        that tool results were sent back to the model.
        """

        self.received_messages.append(
            list(messages)
        )

        if self.call_index >= len(
            self.responses
        ):
            raise RuntimeError(
                "FakeQwenClient ran out of prepared responses."
            )

        response = self.responses[
            self.call_index
        ]

        self.call_index += 1

        return response

    @staticmethod
    def get_message_content(
        message: Any,
    ) -> str:
        """Extract visible assistant content."""

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            return ""

        return str(
            content
        ).strip()


def make_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    tool_call_id: str = "tool-call-1",
) -> Any:
    """Create a fake Hugging Face-style tool call."""

    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(
            name=tool_name,
            arguments=arguments,
        ),
    )


def make_assistant_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    """Create a fake assistant message."""

    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )


def build_agent(
    responses: list[Any],
) -> DataAnalystAgent:
    """
    Construct an agent without initializing the real Hugging Face client.
    """

    agent = object.__new__(
        DataAnalystAgent
    )

    agent.dataset_path = (
        DATASET_PATH
    )

    agent.state = AgentState()

    agent.state.add_message(
        {
            "role": "system",
            "content": (
                "Test system prompt."
            ),
        }
    )

    agent.llm = FakeQwenClient(
        responses
    )

    return agent


def test_single_tool_call_then_final_answer() -> None:
    """
    The model should be able to:

    1. request one analytical tool,
    2. receive the result,
    3. return a final answer.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "group_and_aggregate",
                    {
                        "group_by": "category",
                        "metric": "revenue",
                        "aggregation": "sum",
                        "sort_descending": True,
                    },
                )
            ],
        ),
        make_assistant_response(
            content=(
                "Electronics generated the most revenue "
                "at $796,047.76."
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Which category generated the most revenue?",
        verbose=False,
    )

    assert (
        "Electronics"
        in answer
    )

    assert (
        "$796,047.76"
        in answer
    )

    assert (
        len(
            agent.state.tool_history
        )
        == 1
    )

    execution = (
        agent.state.tool_history[
            0
        ]
    )

    assert (
        execution.name
        == "group_and_aggregate"
    )

    assert (
        execution.result[
            "result"
        ][0]["group"]
        == "Electronics"
    )


def test_tool_result_is_returned_to_model() -> None:
    """
    Verify that a tool result becomes a tool-role message
    before the next model call.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "calculate_statistics",
                    {
                        "operation": (
                            "percentage_change"
                        ),
                        "column": "revenue",
                        "group_by": "month",
                    },
                )
            ],
        ),
        make_assistant_response(
            content=(
                "Revenue decreased by 15.13% "
                "in the latest month."
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    agent.ask(
        "Did revenue decrease last month?",
        verbose=False,
    )

    fake_client = agent.llm

    assert isinstance(
        fake_client,
        FakeQwenClient,
    )

    assert (
        len(
            fake_client.received_messages
        )
        == 2
    )

    second_request = (
        fake_client.received_messages[
            1
        ]
    )

    tool_messages = [
        message
        for message in second_request
        if message.get(
            "role"
        )
        == "tool"
    ]

    assert (
        len(
            tool_messages
        )
        == 1
    )

    assert (
        tool_messages[
            0
        ]["name"]
        == "calculate_statistics"
    )

    assert (
        "percentage_change"
        in tool_messages[
            0
        ]["content"]
    )


def test_multiple_tool_iterations() -> None:
    """
    Verify that the agent can continue through multiple
    model-selected analytical steps.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "calculate_statistics",
                    {
                        "operation": (
                            "percentage_change"
                        ),
                        "column": "revenue",
                        "group_by": "month",
                    },
                    tool_call_id="call-1",
                )
            ],
        ),
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "run_python",
                    {
                        "code": (
                            "working = df.copy()\n"
                            "working['month'] = "
                            "working['order_date']"
                            ".dt.to_period('M')"
                            ".astype(str)\n"
                            "summary = "
                            "working["
                            "working['month']"
                            ".isin(['2026-08','2026-09'])"
                            "]"
                            ".groupby(['category','month'])"
                            "['revenue'].sum()"
                            ".unstack(fill_value=0)\n"
                            "summary"
                        )
                    },
                    tool_call_id="call-2",
                )
            ],
        ),
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "create_chart",
                    {
                        "chart_type": "line",
                        "x": "month",
                        "y": "revenue",
                        "title": (
                            "Monthly Revenue Trend"
                        ),
                        "output_name": (
                            "monthly_revenue_trend.png"
                        ),
                    },
                    tool_call_id="call-3",
                )
            ],
        ),
        make_assistant_response(
            content=(
                "Revenue decreased by 15.13% in September 2026. "
                "The decline was concentrated most strongly in "
                "Electronics. "
                "Chart: monthly_revenue_trend.png"
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Why did sales decrease last month?",
        verbose=False,
    )

    assert (
        "15.13%"
        in answer
    )

    assert (
        "Electronics"
        in answer
    )

    assert (
        "monthly_revenue_trend.png"
        in answer
    )

    tool_names = [
        execution.name
        for execution
        in agent.state.tool_history
    ]

    assert tool_names == [
        "calculate_statistics",
        "run_python",
        "create_chart",
    ]


def test_fake_chart_in_final_answer_triggers_validation() -> None:
    """
    A final answer mentioning a chart that was never generated
    must be rejected and sent back to the model.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "group_and_aggregate",
                    {
                        "group_by": "category",
                        "metric": "revenue",
                        "aggregation": "sum",
                    },
                    tool_call_id="call-1",
                )
            ],
        ),
        make_assistant_response(
            content=(
                "Electronics generated the most revenue.\n"
                "Chart: fake_chart.png"
            ),
        ),
        make_assistant_response(
            content=(
                "Electronics generated the most revenue "
                "at $796,047.76."
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Which category generated the most revenue?",
        verbose=False,
    )

    assert (
        "fake_chart.png"
        not in answer
    )

    assert (
        "$796,047.76"
        in answer
    )

    fake_client = agent.llm

    assert isinstance(
        fake_client,
        FakeQwenClient,
    )

    assert (
        fake_client.call_index
        == 3
    )


def test_unsupported_percentage_triggers_validation() -> None:
    """
    A percentage not present in tool evidence must cause another
    model turn rather than being returned directly to the user.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "group_and_aggregate",
                    {
                        "group_by": "category",
                        "metric": "revenue",
                        "aggregation": "sum",
                    },
                )
            ],
        ),
        make_assistant_response(
            content=(
                "Electronics produced 53% of revenue."
            ),
        ),
        make_assistant_response(
            content=(
                "Electronics generated the highest revenue "
                "at $796,047.76."
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Which category generated the most revenue?",
        verbose=False,
    )

    assert (
        "53%"
        not in answer
    )

    assert (
        "$796,047.76"
        in answer
    )


def test_tool_call_budget_is_enforced() -> None:
    """
    Verify that the agent cannot execute more than the configured
    number of tools for one user question.
    """

    responses: list[Any] = []

    for index in range(
        MAX_TOOL_CALLS_PER_QUESTION
    ):
        responses.append(
            make_assistant_response(
                tool_calls=[
                    make_tool_call(
                        "calculate_statistics",
                        {
                            "operation": "sum",
                            "column": "revenue",
                        },
                        tool_call_id=(
                            f"call-{index}"
                        ),
                    )
                ],
            )
        )

    responses.append(
        make_assistant_response(
            content=(
                "Total revenue is $1,413,404.45."
            ),
        )
    )

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Analyze revenue.",
        verbose=False,
    )

    assert (
        "$1,413,404.45"
        in answer
    )

    assert (
        len(
            agent.state.tool_history
        )
        == MAX_TOOL_CALLS_PER_QUESTION
    )


def test_tool_error_is_added_to_history() -> None:
    """
    Tool failures should not crash the entire agent loop.
    The failed execution should remain visible in state.
    """

    responses = [
        make_assistant_response(
            tool_calls=[
                make_tool_call(
                    "create_chart",
                    {
                        "chart_type": "bar",
                        "x": "category",
                        "y": "value",
                        "title": "Invalid Chart",
                    },
                )
            ],
        ),
        make_assistant_response(
            content=(
                "The chart could not be generated because "
                "the requested y-axis was not a dataset column."
            ),
        ),
    ]

    agent = build_agent(
        responses
    )

    answer = agent.ask(
        "Create a category chart.",
        verbose=False,
    )

    assert (
        "could not be generated"
        in answer.lower()
    )

    assert (
        len(
            agent.state.tool_history
        )
        == 1
    )

    result = (
        agent.state.tool_history[
            0
        ].result
    )

    assert (
        result[
            "status"
        ]
        == "error"
    )

    assert (
        "does not exist"
        in result[
            "message"
        ]
    )