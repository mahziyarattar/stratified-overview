from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from stratified_overview.embedding import l2_normalize
from stratified_overview.models import Chunk


class FakeEmbedder:
    """Returns precomputed vectors in order, ignoring the texts."""

    name = "fake"

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        assert len(texts) == len(self._vectors)
        return self._vectors


def make_blobs(sizes: Sequence[int], dim: int = 16, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors grouped around orthogonal directions; returns (vectors, true labels)."""
    rng = np.random.default_rng(seed)
    vectors, labels = [], []
    for k, size in enumerate(sizes):
        center = np.zeros(dim)
        center[k] = 1.0
        vectors.append(center + noise * rng.standard_normal((size, dim)))
        labels.append(np.full(size, k))
    return l2_normalize(np.vstack(vectors)), np.concatenate(labels)


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs([120, 90, 60, 40, 30])


@pytest.fixture
def chunks_for():
    def build(n: int, words: int = 80) -> list[Chunk]:
        return [
            Chunk(id=f"c{i}", doc_id=f"d{i}", text=" ".join(["word"] * words), metadata={"title": f"Doc {i}"})
            for i in range(n)
        ]

    return build
