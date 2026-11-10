"""Step 5: deciding what happens to chunks HDBSCAN labeled as noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stratified_overview.clustering import cluster_labels
from stratified_overview.embedding import l2_normalize
from stratified_overview.models import NOISE_LABEL


@dataclass(frozen=True, slots=True)
class NoiseOutcome:
    labels: np.ndarray
    noise_count: int
    misc_label: int | None = None


class NoiseHandler(Protocol):
    def handle(self, labels: np.ndarray, embeddings: np.ndarray) -> NoiseOutcome: ...


class AssignToNearest:
    """Moves each noise chunk into the cluster whose centroid is most similar."""

    def handle(self, labels: np.ndarray, embeddings: np.ndarray) -> NoiseOutcome:
        noise = labels == NOISE_LABEL
        ids = cluster_labels(labels)
        if not noise.any() or not ids:
            return NoiseOutcome(labels=labels, noise_count=int(noise.sum()))
        centroids = l2_normalize(np.stack([embeddings[labels == k].mean(axis=0) for k in ids]))
        nearest = np.argmax(embeddings[noise] @ centroids.T, axis=1)
        assigned = labels.copy()
        assigned[noise] = np.asarray(ids)[nearest]
        return NoiseOutcome(labels=assigned, noise_count=int(noise.sum()))


class KeepAsMisc:
    """Groups all noise chunks into one extra "misc" cluster."""

    def handle(self, labels: np.ndarray, embeddings: np.ndarray) -> NoiseOutcome:
        noise = labels == NOISE_LABEL
        if not noise.any():
            return NoiseOutcome(labels=labels, noise_count=0)
        misc = int(labels.max()) + 1
        grouped = labels.copy()
        grouped[noise] = misc
        return NoiseOutcome(labels=grouped, noise_count=int(noise.sum()), misc_label=misc)


class DropNoise:
    """Leaves noise chunks out of the overview (an ablation, not a default)."""

    def handle(self, labels: np.ndarray, embeddings: np.ndarray) -> NoiseOutcome:
        return NoiseOutcome(labels=labels, noise_count=int((labels == NOISE_LABEL).sum()))
