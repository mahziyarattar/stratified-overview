"""Step 6 (optional): grading how coherent each cluster is."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from stratified_overview.clustering import cluster_labels
from stratified_overview.models import NOISE_LABEL


class QualityScorer(Protocol):
    def score(self, labels: np.ndarray, embeddings: np.ndarray) -> dict[int, float]: ...


class SilhouetteScorer:
    """Mean silhouette per cluster, on a random sample to stay fast.

    The full silhouette compares every pair of points, so large corpora are
    scored on ``sample_size`` chunks. Clusters absent from the sample get no
    score.
    """

    def __init__(self, sample_size: int = 5_000, random_state: int | None = 42) -> None:
        self._sample_size = sample_size
        self._random_state = random_state

    def score(self, labels: np.ndarray, embeddings: np.ndarray) -> dict[int, float]:
        from sklearn.metrics import silhouette_samples

        keep = np.flatnonzero(labels != NOISE_LABEL)
        if len(cluster_labels(labels)) < 2:
            return {}
        if keep.size > self._sample_size:
            rng = np.random.default_rng(self._random_state)
            keep = rng.choice(keep, size=self._sample_size, replace=False)
        sample_labels = labels[keep]
        if np.unique(sample_labels).size < 2:
            return {}
        values = silhouette_samples(embeddings[keep], sample_labels, metric="cosine")
        return {int(k): float(values[sample_labels == k].mean()) for k in np.unique(sample_labels)}
