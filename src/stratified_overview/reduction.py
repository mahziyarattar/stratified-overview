"""Step 2: reducing embeddings to a few dimensions for density clustering."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Reducer(Protocol):
    def fit_transform(self, vectors: np.ndarray) -> np.ndarray: ...


class UmapReducer:
    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        metric: str = "cosine",
        random_state: int | None = 42,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._random_state = random_state

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        import umap

        model = umap.UMAP(
            n_components=self._n_components,
            n_neighbors=min(self._n_neighbors, len(vectors) - 1),
            min_dist=0.0,
            metric=self._metric,
            random_state=self._random_state,
        )
        return model.fit_transform(vectors)


class IdentityReducer:
    """Skips reduction; useful for tests and already low-dimensional data."""

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return vectors
