"""Removing site boilerplate before chunking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from stratified_overview.models import Document


@dataclass(frozen=True, slots=True)
class CleaningOutcome:
    documents: list[Document]
    paragraphs_removed: int


class RepeatedParagraphFilter:
    """Drops paragraphs that appear in at least ``min_documents`` different documents.

    Newsletter sign-ups, share-button rows and "Advertisement" lines repeat
    across a site's articles; real content rarely does. Matching ignores case
    and whitespace.
    """

    def __init__(self, min_documents: int = 3) -> None:
        if min_documents < 2:
            raise ValueError("min_documents must be at least 2")
        self._min_documents = min_documents

    def clean(self, documents: Sequence[Document]) -> CleaningOutcome:
        frequency: Counter[str] = Counter()
        for doc in documents:
            frequency.update({_key(p) for p in _paragraphs(doc.text)})
        repeated = {key for key, count in frequency.items() if count >= self._min_documents}

        cleaned: list[Document] = []
        removed = 0
        for doc in documents:
            kept = [p for p in _paragraphs(doc.text) if _key(p) not in repeated]
            removed += len(_paragraphs(doc.text)) - len(kept)
            if kept:
                cleaned.append(replace(doc, text="\n".join(kept)))
        return CleaningOutcome(documents=cleaned, paragraphs_removed=removed)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


def _key(paragraph: str) -> str:
    return " ".join(paragraph.split()).lower()
