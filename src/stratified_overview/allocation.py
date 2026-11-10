"""Step 7b: dividing the token budget across clusters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from stratified_overview.models import Cluster, ClusterAllocation

_EPSILON = 1e-9


class AllocationStrategy(Protocol):
    name: str

    def weights(self, clusters: Sequence[Cluster]) -> np.ndarray: ...


class EqualWeights:
    name = "equal"

    def weights(self, clusters: Sequence[Cluster]) -> np.ndarray:
        return np.ones(len(clusters))


class SizeProportional:
    name = "size"

    def weights(self, clusters: Sequence[Cluster]) -> np.ndarray:
        return np.array([c.size for c in clusters], dtype=float)


class SqrtSizeSpread:
    """Dampened Neyman allocation: weight = sqrt(size) x spread."""

    name = "sqrt-size-x-spread"

    def weights(self, clusters: Sequence[Cluster]) -> np.ndarray:
        return np.array([np.sqrt(c.size) * c.spread for c in clusters], dtype=float)


class TokenAllocator:
    """Reserves the minimums first, then shares the rest by weight.

    A cluster's reserve is ``min_chunks`` times its mean chunk length, so
    short chunks do not tie up budget at the per-chunk cap. Each cluster is
    capped at ``max_cluster_share`` of the budget (or its reserve, if larger)
    and at the tokens it actually holds. Budget a capped cluster cannot use
    flows to the others, so the total never exceeds the budget and only stays
    below it when every cluster is capped.
    """

    def __init__(self, strategy: AllocationStrategy, budget_tokens: int, max_cluster_share: float) -> None:
        self._strategy = strategy
        self._budget = budget_tokens
        self._max_share = max_cluster_share

    def allocate(self, clusters: Sequence[Cluster], min_chunks: int) -> list[ClusterAllocation]:
        if not clusters:
            return []
        holdings = np.array([c.total_tokens for c in clusters], dtype=float)
        mean_lengths = np.array([c.mean_chunk_tokens for c in clusters], dtype=float)
        reserve = np.minimum(np.ceil(min_chunks * mean_lengths), holdings)
        caps = np.minimum(holdings, np.maximum(reserve, self._max_share * self._budget))
        weights = np.clip(self._strategy.weights(clusters), 0.0, None) + _EPSILON

        alloc = reserve.copy()
        remaining = self._budget - alloc.sum()
        open_ = alloc < caps
        while remaining > 1 and open_.any():
            share = remaining * weights[open_] / weights[open_].sum()
            given = np.minimum(share, caps[open_] - alloc[open_])
            alloc[open_] += given
            remaining -= given.sum()
            open_ = alloc < caps - _EPSILON
            if given.sum() < _EPSILON:
                break

        return [
            ClusterAllocation(label=c.label, weight=float(w), token_budget=int(a))
            for c, w, a in zip(clusters, weights, np.floor(alloc), strict=True)
        ]
