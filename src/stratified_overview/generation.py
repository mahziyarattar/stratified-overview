"""Step 8: writing the overview from the sampled chunks."""

from __future__ import annotations

from typing import Protocol

from stratified_overview.llm import ChatClient, OpenRouterChatClient, RefusalError
from stratified_overview.models import Overview, OverviewRequest
from stratified_overview.tokens import ApproxTokenCounter, TokenCounter

# USD per million tokens (input, output) for prompts up to 100K tokens.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    # OpenRouter slugs, same list prices.
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "anthropic/claude-opus-5.5": (4.00, 20.00),
}

SYSTEM_PROMPT = """\
You write an overview of a document database for people and agents who will query it.
You receive excerpts sampled from groups of similar chunks, each with the group's share of the database.

Organize the overview by topic, not by group, and give each section a descriptive topic name. \
When a group mixes subjects, place each subject in the section where it belongs rather than \
creating a mixed or miscellaneous section; do the same for the group marked misc. For each \
section, describe the main subjects and the notable specifics the excerpts show (entities, \
events, dates).

Give a section's approximate share of the database only when the section covers a clearly \
defined body of content, as one rounded percentage; never list several percentages for one section.

Use only what the excerpts show. Do not mention groups or their numbers, and do not comment on \
boilerplate, formatting or data quality. End with two sentences summarizing the whole database."""


class OverviewRefusedError(RefusalError):
    """Raised when the model declines to write the overview."""


class OverviewGenerator(Protocol):
    def generate(self, request: OverviewRequest) -> Overview: ...


class OverviewPromptBuilder:
    def build(self, request: OverviewRequest) -> str:
        lines = [f'<database total_chunks="{request.total_chunks}">']
        for section in request.sections:
            kind = "misc" if section.is_misc else "topic"
            lines.append(f'<group kind="{kind}" share="{section.share_of_corpus:.1%}">')
            lines.extend(f"<excerpt>\n{text}\n</excerpt>" for text in section.excerpts)
            lines.append("</group>")
        lines.append("</database>")
        lines.append("Write the overview of this database.")
        return "\n".join(lines)


class ClaudeOverviewGenerator:
    """One Messages API call over all sampled chunks."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        effort: str = "medium",
        max_tokens: int = 16_000,
        client: object | None = None,
        prompt_builder: OverviewPromptBuilder | None = None,
    ) -> None:
        import anthropic

        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()
        self._prompt_builder = prompt_builder or OverviewPromptBuilder()

    def generate(self, request: OverviewRequest) -> Overview:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": self._effort},
            messages=[{"role": "user", "content": self._prompt_builder.build(request)}],
        )
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise OverviewRefusedError(f"model declined to write the overview (category: {category})")

        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            text += "\n\n[truncated: hit max_tokens]"
        usage = response.usage
        return Overview(
            text=text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=estimate_cost(self._model, usage.input_tokens, usage.output_tokens),
        )


class OpenRouterOverviewGenerator:
    """One call through OpenRouter's OpenAI-compatible API.

    Benchmark baselines (GraphRAG, LazyGraphRAG, RAPTOR) speak the same API,
    so every method can share one endpoint and one model.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-5.5",
        max_tokens: int = 16_000,
        client: ChatClient | None = None,
        prompt_builder: OverviewPromptBuilder | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or OpenRouterChatClient(model)
        self._prompt_builder = prompt_builder or OverviewPromptBuilder()

    def generate(self, request: OverviewRequest) -> Overview:
        result = self._client.complete(SYSTEM_PROMPT, self._prompt_builder.build(request), self._max_tokens)
        text = result.text + ("\n\n[truncated: hit max_tokens]" if result.truncated else "")
        return Overview(
            text=text,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_usd=estimate_cost(self._model, result.input_tokens, result.output_tokens),
        )


class PromptPreviewGenerator:
    """Dry run: builds the prompt and estimates its size without calling an API."""

    def __init__(
        self,
        counter: TokenCounter | None = None,
        prompt_builder: OverviewPromptBuilder | None = None,
        priced_as: str = "claude-haiku-5-5",
    ) -> None:
        self._counter = counter or ApproxTokenCounter()
        self._prompt_builder = prompt_builder or OverviewPromptBuilder()
        self._priced_as = priced_as

    def generate(self, request: OverviewRequest) -> Overview:
        prompt = self._prompt_builder.build(request)
        input_tokens = self._counter.count(SYSTEM_PROMPT) + self._counter.count(prompt)
        return Overview(
            text=prompt,
            model="dry-run",
            input_tokens=input_tokens,
            output_tokens=0,
            estimated_cost_usd=estimate_cost(self._priced_as, input_tokens, 0),
        )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICING_PER_MTOK.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
