"""Token counting used for budget planning.

No public tokenizer matches Claude 5.x models, so ``TiktokenCounter`` counts
with OpenAI's o200k tokenizer and multiplies by a per-model scale measured
from provider-billed counts. ``TokenCalibration`` stores that scale and
updates it after every real call, so estimates stay within a few percent at
no extra cost.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

# billed tokens / o200k tokens, measured on the MultiHop-RAG overview prompt
# (2026-11-10: 62,295 billed vs 38,229 o200k).
DEFAULT_SCALES: dict[str, float] = {
    "claude-haiku-5-5": 1.63,
    "anthropic/claude-haiku-5.5": 1.63,
}


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class ApproxTokenCounter:
    """Character-based estimate; dependency-free, for tests and rough checks."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self._chars_per_token)

    def truncate(self, text: str, max_tokens: int) -> str:
        limit = int(max_tokens * self._chars_per_token)
        if len(text) <= limit:
            return text
        cut = text[:limit]
        # Prefer ending on a word boundary.
        space = cut.rfind(" ")
        return cut[:space] if space > limit * 0.8 else cut


class TiktokenCounter:
    """o200k token counts scaled to the target model's tokenizer."""

    def __init__(self, scale: float = 1.0, encoding: str = "o200k_base") -> None:
        import tiktoken

        if scale <= 0:
            raise ValueError("scale must be positive")
        self._scale = scale
        self._encoding = tiktoken.get_encoding(encoding)

    @property
    def scale(self) -> float:
        return self._scale

    def raw_count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def count(self, text: str) -> int:
        return math.ceil(self.raw_count(text) * self._scale)

    def truncate(self, text: str, max_tokens: int) -> str:
        limit = math.floor(max_tokens / self._scale)
        ids = self._encoding.encode(text, disallowed_special=())
        return text if len(ids) <= limit else self._encoding.decode(ids[:limit])


class TokenCalibration:
    """Per-model scale (billed / o200k tokens), persisted as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def scale_for(self, model: str) -> float:
        entry = self._data.get(model)
        if entry:
            return float(entry["scale"])
        return DEFAULT_SCALES.get(model, 1.0)

    def record(self, model: str, raw_tokens: int, billed_tokens: int) -> float:
        """Store the scale measured on one call and return it."""
        if raw_tokens <= 0 or billed_tokens <= 0:
            raise ValueError("token counts must be positive")
        scale = billed_tokens / raw_tokens
        self._data[model] = {
            "scale": round(scale, 4),
            "raw_tokens": raw_tokens,
            "billed_tokens": billed_tokens,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        return scale
