"""Plain data objects passed between pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NOISE_LABEL = -1


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Cluster:
    """A group of chunks with the statistics allocation needs."""

    label: int
    member_indices: np.ndarray
    centroid: np.ndarray
    spread: float
    total_tokens: int
    quality: float | None = None
    is_misc: bool = False

    @property
    def size(self) -> int:
        return int(self.member_indices.size)

    @property
    def mean_chunk_tokens(self) -> float:
        return self.total_tokens / self.size if self.size else 0.0


@dataclass(frozen=True, slots=True)
class ClusterAllocation:
    label: int
    weight: float
    token_budget: int


@dataclass(frozen=True, slots=True)
class ClusterSample:
    """The chunks picked from one cluster, in selection order."""

    cluster: Cluster
    chunk_indices: tuple[int, ...]
    tokens_used: int


@dataclass(frozen=True, slots=True)
class OverviewSection:
    label: int
    share_of_corpus: float
    is_misc: bool
    excerpts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OverviewRequest:
    total_chunks: int
    sections: tuple[OverviewSection, ...]


@dataclass(frozen=True, slots=True)
class Overview:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None
