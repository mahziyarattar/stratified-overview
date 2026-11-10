"""Pipeline settings and their upfront validation."""

from __future__ import annotations

import math
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when a combination of settings can never be satisfied."""


@dataclass(frozen=True, slots=True)
class OverviewConfig:
    """Structural settings shared by the pipeline steps.

    Strategy-specific knobs (diversity strength, noise policy, allocation
    strategy) live on the strategy objects injected into the pipeline.
    """

    budget_tokens: int = 40_000
    max_tokens_per_chunk: int = 512
    min_chunks_per_cluster: int = 2
    max_cluster_share: float = 0.4
    min_clusters: int = 5
    max_clusters: int = 50
    min_cluster_size: int | None = None
    min_samples: int = 5
    umap_components: int = 5
    umap_neighbors: int = 15
    random_state: int = 42

    def __post_init__(self) -> None:
        self._validate()

    def resolved_min_cluster_size(self, n_chunks: int) -> int:
        """Return the configured size, or 0.5% of the corpus with a floor of 10."""
        if self.min_cluster_size is not None:
            return self.min_cluster_size
        return max(10, math.ceil(0.005 * n_chunks))

    def _validate(self) -> None:
        positive = {
            "budget_tokens": self.budget_tokens,
            "max_tokens_per_chunk": self.max_tokens_per_chunk,
            "min_chunks_per_cluster": self.min_chunks_per_cluster,
            "min_clusters": self.min_clusters,
            "min_samples": self.min_samples,
            "umap_components": self.umap_components,
            "umap_neighbors": self.umap_neighbors,
        }
        for name, value in positive.items():
            if value < 1:
                raise ConfigError(f"{name} must be at least 1, got {value}")

        if self.min_cluster_size is not None and self.min_cluster_size < 2:
            raise ConfigError("min_cluster_size must be at least 2")
        if self.min_clusters > self.max_clusters:
            raise ConfigError(
                f"min_clusters ({self.min_clusters}) exceeds max_clusters ({self.max_clusters})"
            )
        if not 0 < self.max_cluster_share <= 1:
            raise ConfigError("max_cluster_share must be in (0, 1]")

        # The budget fitter lowers the per-cluster minimum to 1 chunk and merges
        # clusters, but never below min_clusters. This bound guarantees a fit.
        floor = self.min_clusters * self.max_tokens_per_chunk
        if self.budget_tokens < floor:
            raise ConfigError(
                f"budget_tokens ({self.budget_tokens}) is below "
                f"min_clusters x max_tokens_per_chunk ({floor}); "
                "raise the budget or lower min_clusters / max_tokens_per_chunk"
            )
