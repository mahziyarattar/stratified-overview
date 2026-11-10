"""Wires the steps into one run: chunks in, overview out."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from stratified_overview.allocation import AllocationStrategy, SqrtSizeSpread, TokenAllocator
from stratified_overview.budget import BudgetFitter
from stratified_overview.clustering import ClusterCountEnforcer, HdbscanClusterer
from stratified_overview.config import OverviewConfig
from stratified_overview.embedding import Embedder, l2_normalize
from stratified_overview.generation import OverviewGenerator
from stratified_overview.models import (
    Chunk,
    Cluster,
    ClusterAllocation,
    ClusterSample,
    Overview,
    OverviewRequest,
    OverviewSection,
)
from stratified_overview.noise import AssignToNearest, NoiseHandler
from stratified_overview.profiling import ClusterProfiler
from stratified_overview.quality import QualityScorer, SilhouetteScorer
from stratified_overview.reduction import Reducer, UmapReducer
from stratified_overview.selection import DiverseMMR, SelectionStrategy
from stratified_overview.tokens import ApproxTokenCounter, TokenCounter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    config: OverviewConfig
    chunks: tuple[Chunk, ...]
    clustering_steps: tuple[str, ...]
    budget_steps: tuple[str, ...]
    noise_count: int
    min_chunks_per_cluster: int
    clusters: tuple[Cluster, ...]
    allocations: tuple[ClusterAllocation, ...]
    samples: tuple[ClusterSample, ...]
    request: OverviewRequest
    overview: Overview
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_selected(self) -> int:
        return sum(s.tokens_used for s in self.samples)


class CorpusOverviewPipeline:
    """Runs steps 1-8. Every step is an injected object, so ablations swap one argument."""

    def __init__(
        self,
        config: OverviewConfig,
        embedder: Embedder,
        generator: OverviewGenerator,
        reducer: Reducer,
        count_enforcer: ClusterCountEnforcer,
        noise_handler: NoiseHandler,
        allocation: AllocationStrategy,
        selection: SelectionStrategy,
        quality_scorer: QualityScorer | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._generator = generator
        self._reducer = reducer
        self._count_enforcer = count_enforcer
        self._noise_handler = noise_handler
        self._allocation = allocation
        self._selection = selection
        self._quality_scorer = quality_scorer
        self._counter = token_counter or ApproxTokenCounter()

    @classmethod
    def with_defaults(
        cls,
        config: OverviewConfig,
        embedder: Embedder,
        generator: OverviewGenerator,
        n_chunks: int,
        **overrides: object,
    ) -> CorpusOverviewPipeline:
        """The configuration the article proposes; pass keyword overrides for ablations."""
        clusterer = HdbscanClusterer(
            min_cluster_size=config.resolved_min_cluster_size(n_chunks),
            min_samples=config.min_samples,
        )
        components: dict[str, object] = {
            "reducer": UmapReducer(
                n_components=config.umap_components,
                n_neighbors=config.umap_neighbors,
                random_state=config.random_state,
            ),
            "count_enforcer": ClusterCountEnforcer(
                clusterer, config.min_clusters, config.max_clusters, config.random_state
            ),
            "noise_handler": AssignToNearest(),
            "allocation": SqrtSizeSpread(),
            "selection": DiverseMMR(),
            "quality_scorer": SilhouetteScorer(random_state=config.random_state),
        }
        components.update(overrides)
        return cls(config=config, embedder=embedder, generator=generator, **components)  # type: ignore[arg-type]

    def run(self, chunks: Sequence[Chunk]) -> PipelineResult:
        if not chunks:
            raise ValueError("no chunks to summarize")
        timings: dict[str, float] = {}
        texts = [c.text for c in chunks]

        with _timed(timings, "embed"):
            embeddings = l2_normalize(np.asarray(self._embedder.embed(texts), dtype=np.float32))
        with _timed(timings, "reduce"):
            reduced = self._reducer.fit_transform(embeddings)
        with _timed(timings, "cluster"):
            clustering = self._count_enforcer.cluster(reduced, embeddings)
        noise = self._noise_handler.handle(clustering.labels, embeddings)

        # Budget what is actually sent: the truncated chunk plus its title header.
        token_lengths = np.array([self._counter.count(self._excerpt(c)) for c in chunks])
        fit = BudgetFitter(self._config).fit(noise.labels, embeddings, token_lengths, noise.misc_label)
        with _timed(timings, "quality"):
            quality = self._quality_scorer.score(fit.labels, embeddings) if self._quality_scorer else None
        clusters = ClusterProfiler(embeddings, token_lengths).profile(fit.labels, noise.misc_label, quality)

        allocator = TokenAllocator(self._allocation, self._config.budget_tokens, self._config.max_cluster_share)
        allocations = allocator.allocate(clusters, fit.min_chunks_per_cluster)
        samples = [
            self._selection.select(cluster, embeddings, token_lengths, alloc.token_budget)
            for cluster, alloc in zip(clusters, allocations, strict=True)
        ]

        request = self._build_request(chunks, clusters, samples)
        with _timed(timings, "generate"):
            overview = self._generator.generate(request)

        return PipelineResult(
            config=self._config,
            chunks=tuple(chunks),
            clustering_steps=clustering.steps,
            budget_steps=fit.steps,
            noise_count=noise.noise_count,
            min_chunks_per_cluster=fit.min_chunks_per_cluster,
            clusters=tuple(clusters),
            allocations=tuple(allocations),
            samples=tuple(samples),
            request=request,
            overview=overview,
            timings_s=timings,
        )

    def _build_request(
        self, chunks: Sequence[Chunk], clusters: Sequence[Cluster], samples: Sequence[ClusterSample]
    ) -> OverviewRequest:
        sections = tuple(
            OverviewSection(
                label=cluster.label,
                share_of_corpus=cluster.size / len(chunks),
                is_misc=cluster.is_misc,
                excerpts=tuple(self._excerpt(chunks[i]) for i in sample.chunk_indices),
            )
            for cluster, sample in zip(clusters, samples, strict=True)
        )
        return OverviewRequest(total_chunks=len(chunks), sections=sections)

    def _excerpt(self, chunk: Chunk) -> str:
        """The chunk headed by its document title, truncated so both fit the per-chunk cap."""
        title = chunk.metadata.get("title")
        header = f"[{title}]\n" if title else ""
        room = max(1, self._config.max_tokens_per_chunk - self._counter.count(header))
        return header + self._counter.truncate(chunk.text, room)


class _timed:
    def __init__(self, timings: dict[str, float], name: str) -> None:
        self._timings, self._name = timings, name

    def __enter__(self) -> None:
        self._start = time.perf_counter()
        logger.info("step %s: started", self._name)

    def __exit__(self, *exc: object) -> None:
        self._timings[self._name] = time.perf_counter() - self._start
        logger.info("step %s: %.1fs", self._name, self._timings[self._name])
