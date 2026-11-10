"""Splitting documents into chunks of bounded length."""

from __future__ import annotations

from collections.abc import Iterable

from stratified_overview.models import Chunk, Document
from stratified_overview.tokens import TokenCounter


class ParagraphChunker:
    """Packs whole paragraphs into chunks up to ``max_tokens``.

    A paragraph longer than the limit is split on word boundaries.
    """

    def __init__(self, max_tokens: int, counter: TokenCounter) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self._max_tokens = max_tokens
        self._counter = counter

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for i, text in enumerate(self._pack(self._pieces(doc.text))):
                chunks.append(
                    Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=text, metadata=dict(doc.metadata))
                )
        return chunks

    def _pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for paragraph in (p.strip() for p in text.split("\n")):
            if not paragraph:
                continue
            if self._counter.count(paragraph) <= self._max_tokens:
                pieces.append(paragraph)
            else:
                pieces.extend(self._split_words(paragraph))
        return pieces

    def _split_words(self, paragraph: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        for word in paragraph.split():
            candidate = " ".join([*current, word])
            if current and self._counter.count(candidate) > self._max_tokens:
                parts.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            parts.append(" ".join(current))
        return parts

    def _pack(self, pieces: list[str]) -> list[str]:
        packed: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n{piece}" if current else piece
            if current and self._counter.count(candidate) > self._max_tokens:
                packed.append(current)
                current = piece
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed
