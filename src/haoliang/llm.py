"""OpenAI-compatible chat client shared by the LLM adapters.

`summarize.py` and `edges_llm.py` talk to the same DeepSeek-compatible
endpoint. This module keeps the connection details in one place so a new
adapter (for example `LLMReviser`) does not copy the key/client boilerplate.

Tests do not need the `openai` package: they inject a fake object with a
`complete()` method that returns `ChatResult`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMSettings:
    """Connection and default generation settings for one model."""

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1600
    timeout: float = 90.0


@dataclass(frozen=True)
class ChatResult:
    """One model reply plus token usage, when the provider reports it."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Anything that can turn messages into one text reply."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        ...


def strip_code_fence(text: str) -> str:
    """Remove a Markdown fence that some models add around JSON."""
    content = text.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    return content


class OpenAICompatibleClient:
    """Thin wrapper around the OpenAI SDK.

    The SDK is imported lazily so the core package keeps working without the
    optional `openai` extra.
    """

    def __init__(self, settings: LLMSettings | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError("Install haoliang[openai] to call an LLM") from exc

        self.settings = settings or LLMSettings()
        key = os.environ.get(self.settings.api_key_env)
        if not key:
            raise RuntimeError(f"{self.settings.api_key_env} is not set")

        self.client = OpenAI(
            api_key=key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        kwargs: dict = {
            "model": self.settings.model,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self.settings.max_tokens if max_tokens is None else max_tokens
            ),
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        usage: dict[str, int] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(raw_usage, name, None)
                if value is not None:
                    usage[name] = int(value)
        return ChatResult(text=text, usage=usage)
