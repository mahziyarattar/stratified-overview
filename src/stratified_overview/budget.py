"""Step 7a: making sure the per-cluster minimums fit the token budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stratified_overview.clustering import ClusterMerger, cluster_labels
from stratified_overview.config import OverviewConfig


@dataclass(frozen=True, slots=True)
class BudgetFit:
    labels: np.ndarray
    min_chunks_per_cluster: int
    steps: tuple[str, ...]


def reserve_tokens(labels: np.ndarray, token_lengths: np.ndarray, min_chunks: int) -> float:
    """Tokens the minimums need: min_chunks x each cluster's mean chunk length."""
    return sum(min_chunks * float(token_lengths[labels == k].mean()) for k in cluster_labels(labels))


class BudgetFitter:
    """Applies the conflict rule.

    If the minimums (min chunks x each cluster's mean chunk length) exceed the
    budget: first lower the minimum to 1 chunk, then merge the smallest
    cluster into its nearest neighbor until the minimums fit, never going
    below ``min_clusters``. ``OverviewConfig`` validation (budget >= min
    clusters x max tokens per chunk) guarantees this ends with a fit. The misc
    cluster, if any, is never merged away.
    """

    def __init__(self, config: OverviewConfig) -> None:
        self._config = config

    def fit(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        token_lengths: np.ndarray,
        misc_label: int | None = None,
    ) -> BudgetFit:
        budget = self._config.budget_tokens
        min_chunks = self._config.min_chunks_per_cluster
        steps: list[str] = []

        if reserve_tokens(labels, token_lengths, min_chunks) <= budget:
            return BudgetFit(labels=labels, min_chunks_per_cluster=min_chunks, steps=())

        if min_chunks > 1:
            min_chunks = 1
            steps.append("minimums exceeded the budget; lowered min chunks per cluster to 1")

        merger = ClusterMerger(embeddings)
        protected = () if misc_label is None else (misc_label,)
        merges = 0
        while (
            reserve_tokens(labels, token_lengths, min_chunks) > budget
            and len(cluster_labels(labels)) > self._config.min_clusters
        ):
            merged = merger.merge_smallest(labels, protected=protected)
            if np.array_equal(merged, labels):
                break
            labels, merges = merged, merges + 1
        if merges:
            steps.append(
                f"merged the {merges} smallest clusters into neighbors -> {len(cluster_labels(labels))} clusters"
            )

        return BudgetFit(labels=labels, min_chunks_per_cluster=min_chunks, steps=tuple(steps))
