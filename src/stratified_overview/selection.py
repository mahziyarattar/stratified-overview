"""Step 7c: choosing which chunks represent each cluster."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from stratified_overview.models import Cluster, ClusterSample


class SelectionStrategy(Protocol):
    name: str

    def select(
        self,
        cluster: Cluster,
        embeddings: np.ndarray,
        token_lengths: np.ndarray,
        token_budget: int,
    ) -> ClusterSample: ...


class DiverseMMR:
    """Most typical chunk first, then each time the chunk most unlike those picked.

    Score = (1 - diversity) x similarity to centroid - diversity x max similarity
    to already-picked chunks (Maximal Marginal Relevance). Chunks at or above
    ``duplicate_threshold`` similarity to a picked chunk are skipped as
    near-duplicates. ``diversity=0`` reduces to centroid-only selection.
    """

    name = "diverse-mmr"

    def __init__(self, diversity: float = 0.5, duplicate_threshold: float | None = 0.95) -> None:
        if not 0 <= diversity <= 1:
            raise ValueError("diversity must be in [0, 1]")
        self._diversity = diversity
        self._duplicate_threshold = duplicate_threshold

    def select(
        self,
        cluster: Cluster,
        embeddings: np.ndarray,
        token_lengths: np.ndarray,
        token_budget: int,
    ) -> ClusterSample:
        members = cluster.member_indices
        vectors = embeddings[members]
        lengths = token_lengths[members]
        typicality = vectors @ cluster.centroid
        max_sim = np.full(members.size, -np.inf)
        taken = np.zeros(members.size, dtype=bool)
        picked: list[int] = []
        remaining = token_budget

        while True:
            candidates = ~taken & (lengths <= remaining)
            if self._duplicate_threshold is not None:
                candidates &= max_sim < self._duplicate_threshold
            if not candidates.any():
                break
            redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
            score = (1 - self._diversity) * typicality - self._diversity * redundancy
            best = int(np.argmax(np.where(candidates, score, -np.inf)))
            taken[best] = True
            picked.append(best)
            remaining -= int(lengths[best])
            max_sim = np.maximum(max_sim, vectors @ vectors[best])

        return ClusterSample(
            cluster=cluster,
            chunk_indices=tuple(int(members[i]) for i in picked),
            tokens_used=token_budget - remaining,
        )


class CentroidOnly(DiverseMMR):
    """Picks the chunks nearest the centroid, ignoring redundancy (ablation baseline)."""

    name = "centroid-only"

    def __init__(self) -> None:
        super().__init__(diversity=0.0, duplicate_threshold=None)
