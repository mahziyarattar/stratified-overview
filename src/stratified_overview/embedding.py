"""Step 1: turning chunk texts into unit-length vectors."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so Euclidean distance tracks cosine."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (no API key needed)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self._model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


class CachedEmbedder:
    """Wraps an embedder and stores results on disk, keyed by model and texts."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    @property
    def name(self) -> str:
        return self._inner.name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        path = self._cache_dir / f"embeddings-{self._key(texts)}.npy"
        if path.exists():
            return np.load(path)
        vectors = self._inner.embed(texts)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def _key(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256(self._inner.name.encode())
        for text in texts:
            digest.update(b"\x00")
            digest.update(text.encode())
        return digest.hexdigest()[:16]
