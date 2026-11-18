"""Setup shared by the pilot and the benchmark, so both see exactly the same chunks."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from stratified_overview.chunking import ParagraphChunker
from stratified_overview.cleaning import RepeatedParagraphFilter
from stratified_overview.corpora import MultiHopRagCorpus
from stratified_overview.models import Chunk, Document
from stratified_overview.tokens import DEFAULT_SCALES, TiktokenCounter, TokenCalibration


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    documents: list[Document]
    chunks: list[Chunk]
    paragraphs_removed: int


def prepare_multihop_corpus(model: str, chunk_tokens: int, limit: int | None = None) -> PreparedCorpus:
    """Load, clean and chunk MultiHop-RAG.

    Chunking uses the model's fixed default scale, never the live calibration:
    a calibration update must not move chunk boundaries, or every run would
    invalidate the embedding cache.
    """
    cleaning = RepeatedParagraphFilter(min_documents=3).clean(MultiHopRagCorpus().load(limit=limit))
    counter = TiktokenCounter(scale=DEFAULT_SCALES.get(model, 1.0))
    chunks = ParagraphChunker(chunk_tokens, counter).chunk(cleaning.documents)
    return PreparedCorpus(cleaning.documents, chunks, cleaning.paragraphs_removed)


def load_dotenv(path: Path) -> None:
    """Set variables from a KEY=VALUE file without overriding ones already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def check_token_estimate(
    calibration: TokenCalibration, model: str, counter: TiktokenCounter, texts: Sequence[str], billed: int
) -> str:
    """Compare the estimate for ``texts`` with the billed count and store the new scale."""
    raw = sum(counter.raw_count(text) for text in texts)
    estimated = round(raw * counter.scale)
    deviation = (billed - estimated) / billed
    scale = calibration.record(model, raw, billed)
    message = (
        f"token estimate {estimated:,} vs billed {billed:,} ({deviation:+.1%}); "
        f"scale {counter.scale:.3f} -> {scale:.3f} saved for next run"
    )
    if abs(deviation) > 0.10:
        logging.warning("%s -- budget was off by more than 10%%", message)
    return message
