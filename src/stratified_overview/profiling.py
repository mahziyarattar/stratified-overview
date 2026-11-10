"""Turning labels into Cluster objects with centroid, spread and token mass."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from stratified_overview.clustering import cluster_labels
from stratified_overview.embedding import l2_normalize
from stratified_overview.models import Cluster


class ClusterProfiler:
    """Computes the statistics allocation and selection rely on.

    Spread is the mean cosine distance of members to the centroid, measured
    in the original embedding space.
    """

    def __init__(self, embeddings: np.ndarray, token_lengths: np.ndarray) -> None:
        self._embeddings = embeddings
        self._token_lengths = token_lengths

    def profile(
        self,
        labels: np.ndarray,
        misc_label: int | None = None,
        quality: Mapping[int, float] | None = None,
    ) -> list[Cluster]:
        quality = quality or {}
        clusters = [self._profile_one(labels, k, k == misc_label, quality.get(k)) for k in cluster_labels(labels)]
        return sorted(clusters, key=lambda c: c.size, reverse=True)

    def _profile_one(self, labels: np.ndarray, label: int, is_misc: bool, quality: float | None) -> Cluster:
        members = np.flatnonzero(labels == label)
        vectors = self._embeddings[members]
        centroid = l2_normalize(vectors.mean(axis=0, keepdims=True))[0]
        spread = float(np.mean(1.0 - vectors @ centroid))
        return Cluster(
            label=label,
            member_indices=members,
            centroid=centroid,
            spread=max(spread, 0.0),
            total_tokens=int(self._token_lengths[members].sum()),
            quality=quality,
            is_misc=is_misc,
        )
