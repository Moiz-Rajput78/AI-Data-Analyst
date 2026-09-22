"""Conversation and execution state for the AI Data Analyst agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolExecution:
    """Record of one analytical tool execution."""

    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AgentState:
    """
    State maintained during one analytical conversation.

    This keeps the user question, chat history, and tool executions
    together while the agent performs multi-step analysis.
    """

    messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_history: list[ToolExecution] = field(
        default_factory=list
    )

    iterations: int = 0

    def add_message(
        self,
        message: dict[str, Any],
    ) -> None:
        """Add a chat message to conversation state."""

        self.messages.append(
            message
        )

    def add_tool_execution(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        """Record a completed tool execution."""

        self.tool_history.append(
            ToolExecution(
                name=name,
                arguments=arguments,
                result=result,
            )
        )

    def reset_iterations(self) -> None:
        """Reset iteration count for a new user question."""

        self.iterations = 0