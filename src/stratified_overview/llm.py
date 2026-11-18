"""A minimal chat client for OpenAI-compatible endpoints such as OpenRouter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class RefusalError(RuntimeError):
    """Raised when the model declines to answer."""


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


class ChatClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 16_000) -> ChatResult: ...


class OpenRouterChatClient:
    """One system + user turn through OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 6,
    ) -> None:
        from openai import OpenAI

        self._model = model
        # Generous retries: the SDK backs off exponentially on connection errors
        # (including transient DNS failures), 429s and 5xx responses.
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            max_retries=max_retries,
        )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 16_000) -> ChatResult:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise RefusalError(f"{self._model} declined to respond")
        usage = response.usage
        return ChatResult(
            text=choice.message.content or "",
            model=response.model or self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            truncated=choice.finish_reason == "length",
        )
