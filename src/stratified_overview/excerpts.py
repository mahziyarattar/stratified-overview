"""How a chunk is shown to an LLM, shared by the overview and every benchmark method."""

from __future__ import annotations

from stratified_overview.models import Chunk
from stratified_overview.tokens import TokenCounter


class ExcerptFormatter:
    """The chunk headed by its document title, truncated so both fit ``max_tokens``."""

    def __init__(self, counter: TokenCounter, max_tokens: int) -> None:
        self._counter = counter
        self._max_tokens = max_tokens

    def format(self, chunk: Chunk) -> str:
        title = chunk.metadata.get("title")
        header = f"[{title}]\n" if title else ""
        room = max(1, self._max_tokens - self._counter.count(header))
        return header + self._counter.truncate(chunk.text, room)

    def tokens(self, chunk: Chunk) -> int:
        return self._counter.count(self.format(chunk))
