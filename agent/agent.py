"""Dynamic AI Data Analyst agent loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from agent.llm import QwenClient
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState

from tools.aggregation import group_and_aggregate
from tools.charts import create_chart
from tools.dataset import inspect_dataset
from tools.filters import filter_dataset
from tools.python_executor import run_python
from tools.statistics import calculate_statistics


MAX_AGENT_ITERATIONS = 10
MAX_EMPTY_RESPONSES = 2
MAX_TOOL_CALLS_PER_QUESTION = 6


ToolFunction = Callable[..., Any]


class AgentError(RuntimeError):
    """Raised when the AI Data Analyst agent cannot continue."""


TOOL_FUNCTIONS: dict[str, ToolFunction] = {
    "inspect_dataset": inspect_dataset,
    "calculate_statistics": calculate_statistics,
    "group_and_aggregate": group_and_aggregate,
    "filter_dataset": filter_dataset,
    "run_python": run_python,
    "create_chart": create_chart,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": (
                "Inspect the CSV dataset. Returns columns, data types, "
                "missing values, date ranges, numeric summaries, dataset "
                "size, and sample rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_rows": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_statistics",
            "description": (
                "Calculate mean, median, sum, min, max, standard deviation, "
                "count, or percentage change. Supports normal columns and "
                "special month/year/quarter grouping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "mean",
                            "median",
                            "sum",
                            "min",
                            "max",
                            "std",
                            "count",
                            "percentage_change",
                        ],
                    },
                    "column": {
                        "type": "string",
                    },
                    "group_by": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                },
                "required": [
                    "operation",
                    "column",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "group_and_aggregate",
            "description": (
                "Group the dataset by a business dimension and aggregate "
                "a metric. Suitable for straightforward comparisons by "
                "category, region, product, salesperson, or similar column."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                    },
                    "metric": {
                        "type": "string",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                            "count",
                        ],
                    },
                    "period": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                    "sort_descending": {
                        "type": "boolean",
                    },
                },
                "required": [
                    "group_by",
                    "metric",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_dataset",
            "description": (
                "Filter dataset rows using one condition for focused "
                "inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                    },
                    "operator": {
                        "type": "string",
                        "enum": [
                            "equals",
                            "not_equals",
                            "greater_than",
                            "greater_than_or_equal",
                            "less_than",
                            "less_than_or_equal",
                            "contains",
                        ],
                    },
                    "value": {},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [
                    "column",
                    "operator",
                    "value",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute restricted custom Pandas analysis. The environment "
                "already contains df and pd. Do not import anything. "
                "Imports, loops, filesystem access, network access, and OS "
                "operations are blocked. Prefer vectorized Pandas. This is "
                "the preferred tool for multi-dimensional comparisons when "
                "one calculation can replace many repeated tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                    },
                    "max_output_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": [
                    "code",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Generate a PNG chart directly from the original dataset. "
                "Supported types: line, bar, pie, scatter. The x and y "
                "arguments must reference real dataset columns, except x "
                "may also be month, year, or quarter. Valid numeric y "
                "examples include revenue, profit, cost, quantity, "
                "unit_price, and discount. Do not use tool-result keys such "
                "as value, change, or percentage_change as y unless they "
                "are actual CSV columns. For monthly revenue trends use "
                "chart_type='line', x='month', y='revenue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "line",
                            "bar",
                            "pie",
                            "scatter",
                        ],
                    },
                    "x": {
                        "type": "string",
                    },
                    "y": {
                        "type": "string",
                    },
                    "title": {
                        "type": "string",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": [
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                            "count",
                        ],
                    },
                    "period": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                    "output_name": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                },
                "required": [
                    "chart_type",
                    "x",
                    "y",
                    "title",
                ],
                "additionalProperties": False,
            },
        },
    },
]


class DataAnalystAgent:
    """Dynamic tool-calling AI Data Analyst."""

    def __init__(
        self,
        dataset_path: str | Path,
    ) -> None:
        self.dataset_path = (
            Path(dataset_path)
            .expanduser()
            .resolve()
        )

        if not self.dataset_path.exists():
            raise AgentError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.llm = QwenClient()
        self.state = AgentState()

        self.state.add_message(
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        )

    def _extract_tool_calls(
        self,
        message: Any,
    ) -> list[Any]:
        """Extract tool calls from a Qwen response."""

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if not tool_calls:
            return []

        return list(
            tool_calls
        )

    def _assistant_message_to_dict(
        self,
        message: Any,
    ) -> dict[str, Any]:
        """Convert a Hugging Face response into chat-message format."""

        result: dict[str, Any] = {
            "role": "assistant",
            "content": (
                getattr(
                    message,
                    "content",
                    None,
                )
                or ""
            ),
        }

        tool_calls = self._extract_tool_calls(
            message
        )

        if not tool_calls:
            return result

        serialized_calls: list[
            dict[str, Any]
        ] = []

        for tool_call in tool_calls:
            function = getattr(
                tool_call,
                "function",
                None,
            )

            arguments = getattr(
                function,
                "arguments",
                {},
            )

            if isinstance(
                arguments,
                str,
            ):
                arguments_text = arguments
            else:
                arguments_text = json.dumps(
                    arguments
                )

            serialized_calls.append(
                {
                    "id": str(
                        getattr(
                            tool_call,
                            "id",
                            "",
                        )
                    ),
                    "type": "function",
                    "function": {
                        "name": str(
                            getattr(
                                function,
                                "name",
                                "",
                            )
                        ),
                        "arguments": arguments_text,
                    },
                }
            )

        result["tool_calls"] = serialized_calls

        return result

    @staticmethod
    def _parse_arguments(
        arguments: Any,
    ) -> dict[str, Any]:
        """Normalize tool-call arguments."""

        if arguments is None:
            return {}

        if isinstance(
            arguments,
            dict,
        ):
            return arguments

        if isinstance(
            arguments,
            str,
        ):
            try:
                parsed = json.loads(
                    arguments
                )

            except json.JSONDecodeError as exc:
                raise AgentError(
                    "Model returned invalid JSON for tool arguments."
                ) from exc

            if not isinstance(
                parsed,
                dict,
            ):
                raise AgentError(
                    "Tool arguments must be a JSON object."
                )

            return parsed

        raise AgentError(
            "Unsupported tool argument format."
        )

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a model-selected analytical tool."""

        if name not in TOOL_FUNCTIONS:
            raise AgentError(
                f"Unknown tool requested: {name}"
            )

        arguments = dict(
            arguments
        )

        arguments["dataset_path"] = (
            self.dataset_path
        )

        function = TOOL_FUNCTIONS[
            name
        ]

        try:
            return function(
                **arguments
            )

        except Exception as exc:
            return {
                "status": "error",
                "tool": name,
                "error_type": (
                    type(exc).__name__
                ),
                "message": str(
                    exc
                ),
            }

    @staticmethod
    def _extract_percentage_numbers(
        text: str,
    ) -> list[float]:
        """Extract percentages appearing in visible final text."""

        matches = re.findall(
            r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*%",
            text,
        )

        return [
            float(value)
            for value in matches
        ]

    @staticmethod
    def _collect_numeric_values(
        value: Any,
    ) -> list[float]:
        """Collect numeric values recursively from tool evidence."""

        values: list[
            float
        ] = []

        if isinstance(
            value,
            bool,
        ):
            return values

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            values.append(
                float(value)
            )

            return values

        if isinstance(
            value,
            dict,
        ):
            for item in value.values():
                values.extend(
                    DataAnalystAgent
                    ._collect_numeric_values(
                        item
                    )
                )

            return values

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            for item in value:
                values.extend(
                    DataAnalystAgent
                    ._collect_numeric_values(
                        item
                    )
                )

        return values

    @staticmethod
    def _number_is_supported(
        number: float,
        evidence_numbers: list[float],
    ) -> bool:
        """Check whether a visible percentage is supported."""

        for evidence in evidence_numbers:
            tolerance = max(
                0.02,
                abs(number) * 0.002,
            )

            if abs(
                number - evidence
            ) <= tolerance:
                return True

            if abs(
                abs(number)
                - abs(evidence)
            ) <= tolerance:
                return True

        return False

    def _validate_final_answer(
        self,
        answer: str,
        question_tool_start: int,
    ) -> list[str]:
        """Validate important claims against tool evidence."""

        problems: list[
            str
        ] = []

        question_tools = (
            self.state.tool_history[
                question_tool_start:
            ]
        )

        generated_chart_files: set[
            str
        ] = set()

        evidence_numbers: list[
            float
        ] = []

        for execution in question_tools:
            evidence_numbers.extend(
                self._collect_numeric_values(
                    execution.result
                )
            )

            if execution.name != (
                "create_chart"
            ):
                continue

            if not isinstance(
                execution.result,
                dict,
            ):
                continue

            if execution.result.get(
                "status"
            ) == "error":
                continue

            filename = execution.result.get(
                "filename"
            )

            if filename:
                generated_chart_files.add(
                    str(filename)
                )

        mentioned_charts = set(
            re.findall(
                r"[\w\-]+\.png",
                answer,
                flags=re.IGNORECASE,
            )
        )

        unsupported_charts = (
            mentioned_charts
            - generated_chart_files
        )

        if unsupported_charts:
            problems.append(
                "The proposed final answer mentions chart file(s) "
                "that were not generated: "
                + ", ".join(
                    sorted(
                        unsupported_charts
                    )
                )
                + ". Call create_chart or remove those claims."
            )

        percentages = (
            self._extract_percentage_numbers(
                answer
            )
        )

        unsupported: list[
            float
        ] = []

        for percentage in percentages:
            if not self._number_is_supported(
                percentage,
                evidence_numbers,
            ):
                unsupported.append(
                    percentage
                )

        if unsupported:
            formatted = ", ".join(
                f"{value:g}%"
                for value
                in unsupported
            )

            problems.append(
                "The proposed final answer contains unsupported "
                f"percentage value(s): {formatted}. "
                "Calculate them using an analytical tool or remove them."
            )

        return problems

    def _add_validation_feedback(
        self,
        problems: list[str],
    ) -> None:
        """Return validation feedback to Qwen."""

        feedback = (
            "/no_think\n"
            "FINAL ANSWER VALIDATION FAILED.\n\n"
            "Correct these issue(s):\n\n"
        )

        for index, problem in enumerate(
            problems,
            start=1,
        ):
            feedback += (
                f"{index}. {problem}\n"
            )

        feedback += (
            "\nContinue with another analytical tool only if needed. "
            "Otherwise provide a corrected final answer."
        )

        self.state.add_message(
            {
                "role": "user",
                "content": feedback,
            }
        )

    def _add_empty_response_feedback(
        self,
    ) -> None:
        """Ask Qwen to continue after an empty turn."""

        self.state.add_message(
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Your previous turn returned neither a visible answer "
                    "nor a tool call. Continue now. Call one useful tool if "
                    "additional evidence is needed, otherwise produce the "
                    "final visible report."
                ),
            }
        )

    def _add_tool_budget_feedback(
        self,
    ) -> None:
        """Tell Qwen that no more analytical tools may be called."""

        self.state.add_message(
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "The tool-call budget for this question has been "
                    "reached. Do not call another tool. Use only the "
                    "evidence already collected and produce the best "
                    "grounded final answer now. Do not invent missing "
                    "facts or unsupported causes."
                ),
            }
        )

    def ask(
        self,
        question: str,
        *,
        verbose: bool = True,
    ) -> str:
        """Run the dynamic tool-calling loop for one question."""

        question = question.strip()

        if not question:
            raise AgentError(
                "Question cannot be empty."
            )

        self.state.reset_iterations()

        question_tool_start = len(
            self.state.tool_history
        )

        empty_responses = 0
        tool_calls_used = 0
        budget_feedback_sent = False

        self.state.add_message(
            {
                "role": "user",
                "content": (
                    question
                    + "\n\n/no_think"
                ),
            }
        )

        while (
            self.state.iterations
            < MAX_AGENT_ITERATIONS
        ):
            self.state.iterations += 1

            tool_choice: (
                str
                | dict[str, Any]
                | None
            ) = "auto"

            tools: list[
                dict[str, Any]
            ] | None = TOOL_SCHEMAS

            if (
                tool_calls_used
                >= MAX_TOOL_CALLS_PER_QUESTION
            ):
                tools = None
                tool_choice = None

                if not budget_feedback_sent:
                    self._add_tool_budget_feedback()
                    budget_feedback_sent = True

            response = self.llm.chat(
                self.state.messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.2,
                max_tokens=1600,
            )

            tool_calls = (
                self._extract_tool_calls(
                    response
                )
            )

            visible_content = (
                self.llm.get_message_content(
                    response
                )
            )

            if (
                not tool_calls
                and not visible_content
            ):
                empty_responses += 1

                if verbose:
                    print()
                    print(
                        "[Agent] Qwen returned an empty turn."
                    )
                    print(
                        "[Agent] Asking Qwen to continue..."
                    )

                if (
                    empty_responses
                    > MAX_EMPTY_RESPONSES
                ):
                    raise AgentError(
                        "Qwen repeatedly returned empty responses."
                    )

                self._add_empty_response_feedback()

                continue

            empty_responses = 0

            assistant_message = (
                self._assistant_message_to_dict(
                    response
                )
            )

            self.state.add_message(
                assistant_message
            )

            if not tool_calls:
                validation_problems = (
                    self._validate_final_answer(
                        visible_content,
                        question_tool_start,
                    )
                )

                if validation_problems:
                    if verbose:
                        print()
                        print(
                            "[Agent] Final answer failed "
                            "evidence validation."
                        )

                        for problem in (
                            validation_problems
                        ):
                            print(
                                f"[Agent] {problem}"
                            )

                    self._add_validation_feedback(
                        validation_problems
                    )

                    continue

                return visible_content

            for tool_call in tool_calls:
                if (
                    tool_calls_used
                    >= MAX_TOOL_CALLS_PER_QUESTION
                ):
                    break

                tool_calls_used += 1

                function = getattr(
                    tool_call,
                    "function",
                    None,
                )

                name = str(
                    getattr(
                        function,
                        "name",
                        "",
                    )
                )

                raw_arguments = getattr(
                    function,
                    "arguments",
                    {},
                )

                arguments = (
                    self._parse_arguments(
                        raw_arguments
                    )
                )

                if verbose:
                    print()
                    print(
                        f"[Agent] Tool: {name}"
                    )

                    if arguments:
                        print(
                            "[Agent] Arguments: "
                            + json.dumps(
                                arguments,
                                indent=2,
                            )
                        )

                result = self._execute_tool(
                    name,
                    arguments,
                )

                self.state.add_tool_execution(
                    name=name,
                    arguments=arguments,
                    result=result,
                )

                if verbose:
                    if (
                        isinstance(
                            result,
                            dict,
                        )
                        and result.get(
                            "status"
                        )
                        == "error"
                    ):
                        print(
                            "[Agent] Tool returned an error: "
                            + str(
                                result.get(
                                    "message",
                                    "Unknown error",
                                )
                            )
                        )
                    else:
                        print(
                            "[Agent] Tool completed."
                        )

                    print(
                        "[Agent] Tool calls used: "
                        f"{tool_calls_used}/"
                        f"{MAX_TOOL_CALLS_PER_QUESTION}"
                    )

                tool_call_id = str(
                    getattr(
                        tool_call,
                        "id",
                        "",
                    )
                )

                self.state.add_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": json.dumps(
                            result,
                            default=str,
                        ),
                    }
                )

        raise AgentError(
            "Maximum agent iterations reached before "
            "a valid final answer was produced."
        )