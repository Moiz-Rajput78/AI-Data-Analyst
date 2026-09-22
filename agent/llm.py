"""Hugging Face Qwen client for the AI Data Analyst project."""

from __future__ import annotations

import os
from typing import Any, Iterable

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


class HuggingFaceConfigurationError(RuntimeError):
    """Raised when required Hugging Face configuration is missing."""


class QwenClient:
    """
    Wrapper around Hugging Face Inference Providers for Qwen.

    Qwen acts as the analyst/orchestrator.
    Python and Pandas perform the actual numerical analysis.
    """

    def __init__(self) -> None:
        load_dotenv()

        token = os.getenv(
            "HF_TOKEN",
            "",
        ).strip()

        model = os.getenv(
            "HF_MODEL",
            "Qwen/Qwen3-32B",
        ).strip()

        provider = os.getenv(
            "HF_PROVIDER",
            "auto",
        ).strip() or "auto"

        if not token:
            raise HuggingFaceConfigurationError(
                "Missing HF_TOKEN. "
                "Add your Hugging Face access token to the .env file."
            )

        self.model = model
        self.provider = provider

        self.client = InferenceClient(
            api_key=token,
            provider=provider,
        )

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1800,
    ) -> Any:
        """
        Send a chat-completion request to Qwen through Hugging Face.

        Qwen3 is requested in non-thinking mode because this project's
        visible agent loop performs explicit tool-based reasoning.

        If a selected provider ignores the provider-specific thinking
        option, the system prompt also contains Qwen's /no_think switch.
        """

        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }

        if tools:
            request[
                "tools"
            ] = tools

            if tool_choice is not None:
                request[
                    "tool_choice"
                ] = tool_choice

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    **request
                )
            )

        except Exception as first_error:
            """
            Some routed Hugging Face providers may not accept
            chat_template_kwargs even though other providers do.

            Retry without the provider-specific option. The /no_think
            instruction in the system prompt remains active.
            """

            fallback_request = dict(
                request
            )

            fallback_request.pop(
                "extra_body",
                None,
            )

            try:
                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        **fallback_request
                    )
                )

            except Exception:
                raise first_error

        if not response.choices:
            raise RuntimeError(
                "Hugging Face returned no completion choices."
            )

        return response.choices[
            0
        ].message

    @staticmethod
    def get_message_content(
        message: Any,
    ) -> str:
        """Extract visible final content from a Qwen response."""

        content = getattr(
            message,
            "content",
            None,
        )

        if content:
            return str(
                content
            ).strip()

        return ""

    @staticmethod
    def get_reasoning_content(
        message: Any,
    ) -> str:
        """
        Detect provider-returned reasoning content.

        We do not expose this to the user. It is only used internally
        to diagnose an otherwise empty model response.
        """

        reasoning = getattr(
            message,
            "reasoning_content",
            None,
        )

        if reasoning:
            return str(
                reasoning
            ).strip()

        return ""

    def test_connection(self) -> str:
        """Verify that Qwen is reachable through Hugging Face."""

        message = self.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        "You are performing an API connection test. "
                        "Give only the requested short final answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Reply exactly with: "
                        "Qwen connection successful"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=256,
        )

        content = (
            self.get_message_content(
                message
            )
        )

        if not content:
            raise RuntimeError(
                "Qwen returned no visible final answer "
                "during the connection test."
            )

        return content