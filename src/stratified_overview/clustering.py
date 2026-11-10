"""Steps 3-4: density clustering and keeping the cluster count within bounds."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np

from stratified_overview.embedding import l2_normalize
from stratified_overview.models import NOISE_LABEL


class Clusterer(Protocol):
    def fit_predict(self, vectors: np.ndarray) -> np.ndarray: ...

    def describe(self) -> str: ...


@dataclass(frozen=True, slots=True)
class HdbscanClusterer:
    min_cluster_size: int
    min_samples: int | None = None
    selection_method: Literal["eom", "leaf"] = "eom"
    epsilon: float = 0.0

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.selection_method,
            cluster_selection_epsilon=self.epsilon,
            copy=True,
        )
        return model.fit_predict(vectors)

    def finer(self) -> HdbscanClusterer | None:
        """Return a variant that tends to find more clusters, or None at the limit."""
        if self.selection_method == "eom":
            return replace(self, selection_method="leaf")
        if self.min_cluster_size > 2:
            return replace(self, min_cluster_size=max(2, self.min_cluster_size // 2))
        return None

    def coarser(self, factor: float = 1.5) -> HdbscanClusterer:
        """Return a variant that tends to find fewer, larger clusters."""
        return replace(self, selection_method="eom", min_cluster_size=math.ceil(self.min_cluster_size * factor))

    def describe(self) -> str:
        return (
            f"HDBSCAN(min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, selection={self.selection_method})"
        )


@dataclass(frozen=True, slots=True)
class KMeansClusterer:
    n_clusters: int
    random_state: int | None = 42

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        return model.fit_predict(vectors)

    def describe(self) -> str:
        return f"KMeans(k={self.n_clusters})"


def cluster_labels(labels: np.ndarray, exclude: Collection[int] = ()) -> list[int]:
    """Distinct non-noise labels, optionally skipping some."""
    return [int(x) for x in np.unique(labels) if x != NOISE_LABEL and x not in exclude]


class ClusterMerger:
    """Folds the smallest cluster into its most similar neighbor.

    Merging the smallest (rather than the closest pair) keeps large clusters
    from chaining into grab-bags. Centroids are taken in the original
    (normalized) embedding space, where cosine similarity is meaningful.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings

    def merge_smallest(self, labels: np.ndarray, protected: Collection[int] = ()) -> np.ndarray:
        ids = cluster_labels(labels, protected)
        if len(ids) < 2:
            return labels
        sizes = np.array([(labels == k).sum() for k in ids])
        smallest = int(np.argmin(sizes))
        centroids = l2_normalize(np.stack([self._embeddings[labels == k].mean(axis=0) for k in ids]))
        sims = centroids @ centroids[smallest]
        sims[smallest] = -np.inf
        merged = labels.copy()
        merged[merged == ids[smallest]] = ids[int(np.argmax(sims))]
        return merged


@dataclass(frozen=True, slots=True)
class ClusteringOutcome:
    labels: np.ndarray
    steps: tuple[str, ...]

    @property
    def n_clusters(self) -> int:
        return len(cluster_labels(self.labels))


class ClusterCountEnforcer:
    """Runs HDBSCAN and adjusts until the cluster count lies in [min, max].

    Too few clusters: finer HDBSCAN settings, then k-means with k = min.
    Too many: re-run HDBSCAN with a larger minimum cluster size, keeping the
    last result that stays at or above min; merge the smallest clusters only
    if that still leaves too many.
    """

    def __init__(
        self,
        clusterer: HdbscanClusterer,
        min_clusters: int,
        max_clusters: int,
        random_state: int | None = 42,
        max_refinements: int = 3,
        max_coarsenings: int = 6,
    ) -> None:
        self._clusterer = clusterer
        self._min = min_clusters
        self._max = max_clusters
        self._random_state = random_state
        self._max_refinements = max_refinements
        self._max_coarsenings = max_coarsenings

    def cluster(self, reduced: np.ndarray, embeddings: np.ndarray) -> ClusteringOutcome:
        clusterer = self._clusterer
        labels = clusterer.fit_predict(reduced)
        steps = [self._step("", clusterer, labels)]

        labels, clusterer = self._refine(reduced, labels, clusterer, steps)
        if len(cluster_labels(labels)) < self._min:
            fallback = KMeansClusterer(self._min, self._random_state)
            steps.append(f"still too few; fell back to {fallback.describe()}")
            return ClusteringOutcome(labels=fallback.fit_predict(reduced), steps=tuple(steps))

        labels = self._coarsen(reduced, labels, clusterer, steps)
        merger = ClusterMerger(embeddings)
        merges = 0
        while len(cluster_labels(labels)) > self._max:
            labels = merger.merge_smallest(labels)
            merges += 1
        if merges:
            steps.append(f"still too many; merged the {merges} smallest clusters -> {self._max} clusters")
        return ClusteringOutcome(labels=labels, steps=tuple(steps))

    def _refine(
        self, reduced: np.ndarray, labels: np.ndarray, clusterer: HdbscanClusterer, steps: list[str]
    ) -> tuple[np.ndarray, HdbscanClusterer]:
        for _ in range(self._max_refinements):
            if len(cluster_labels(labels)) >= self._min:
                break
            finer = clusterer.finer()
            if finer is None:
                break
            clusterer, labels = finer, finer.fit_predict(reduced)
            steps.append(self._step("too few; ", clusterer, labels))
        return labels, clusterer

    def _coarsen(
        self, reduced: np.ndarray, labels: np.ndarray, clusterer: HdbscanClusterer, steps: list[str]
    ) -> np.ndarray:
        for _ in range(self._max_coarsenings):
            if len(cluster_labels(labels)) <= self._max:
                break
            coarser = clusterer.coarser()
            candidate = coarser.fit_predict(reduced)
            if len(cluster_labels(candidate)) < self._min:
                steps.append(self._step("too many; ", coarser, candidate) + " (below min, kept previous)")
                break
            clusterer, labels = coarser, candidate
            steps.append(self._step("too many; ", clusterer, labels))
        return labels

    @staticmethod
    def _step(prefix: str, clusterer: HdbscanClusterer, labels: np.ndarray) -> str:
        noise = float(np.mean(labels == NOISE_LABEL))
        return f"{prefix}{clusterer.describe()} -> {len(cluster_labels(labels))} clusters, {noise:.0%} noise"
