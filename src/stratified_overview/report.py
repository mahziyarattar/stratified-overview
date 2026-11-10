"""Human-readable pilot report: what the clusters look like and what was sampled."""

from __future__ import annotations

from typing import Any

from stratified_overview.pipeline import PipelineResult

_PREVIEW_CHARS = 160


class PilotReport:
    def __init__(self, result: PipelineResult) -> None:
        self._r = result

    def summary(self) -> dict[str, Any]:
        r = self._r
        n = len(r.chunks)
        return {
            "chunks": n,
            "clusters": len(r.clusters),
            "noise_chunks": r.noise_count,
            "noise_share": r.noise_count / n,
            "min_chunks_per_cluster": r.min_chunks_per_cluster,
            "budget_tokens": r.config.budget_tokens,
            "tokens_selected": r.tokens_selected,
            "chunks_selected": sum(len(s.chunk_indices) for s in r.samples),
            "clustering_steps": list(r.clustering_steps),
            "budget_steps": list(r.budget_steps),
            "overview_model": r.overview.model,
            "overview_input_tokens": r.overview.input_tokens,
            "overview_output_tokens": r.overview.output_tokens,
            "estimated_cost_usd": r.overview.estimated_cost_usd,
            "timings_s": {k: round(v, 2) for k, v in r.timings_s.items()},
        }

    def to_markdown(self, include_overview: bool = True) -> str:
        r = self._r
        s = self.summary()
        cost = s["estimated_cost_usd"]
        lines = [
            "# Pilot report",
            "",
            f"- Chunks: {s['chunks']}; clusters: {s['clusters']}",
            f"- Noise before handling: {s['noise_chunks']} chunks ({s['noise_share']:.1%})",
            f"- Selected: {s['chunks_selected']} chunks, {s['tokens_selected']:,} of "
            f"{s['budget_tokens']:,} budget tokens (min {s['min_chunks_per_cluster']} per cluster)",
            f"- Overview: {s['overview_model']}, {s['overview_input_tokens']:,} in / "
            f"{s['overview_output_tokens']:,} out tokens"
            + (f", about ${cost:.4f}" if cost is not None else ""),
            f"- Timings (s): {s['timings_s']}",
            "",
            "## Clustering steps",
            "",
            *(f"1. {step}" for step in r.clustering_steps),
            *(f"1. {step}" for step in r.budget_steps),
            "",
            "## Clusters",
            "",
            "| Cluster | Chunks | Share | Spread | Silhouette | Token budget | Picked | Example titles |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        n = len(r.chunks)
        for cluster, alloc, sample in zip(r.clusters, r.allocations, r.samples, strict=True):
            name = f"{cluster.label}{' (misc)' if cluster.is_misc else ''}"
            quality = f"{cluster.quality:.2f}" if cluster.quality is not None else "-"
            lines.append(
                f"| {name} | {cluster.size} | {cluster.size / n:.1%} | {cluster.spread:.3f} | {quality} "
                f"| {alloc.token_budget:,} | {len(sample.chunk_indices)} | {self._titles(sample.chunk_indices)} |"
            )

        lines += ["", "## Most typical chunk per cluster", ""]
        for sample in r.samples:
            if sample.chunk_indices:
                text = " ".join(r.chunks[sample.chunk_indices[0]].text.split())
                lines.append(f"- **{sample.cluster.label}:** {text[:_PREVIEW_CHARS]}...")

        if include_overview and r.overview.model != "dry-run":
            lines += ["", "## Overview", "", r.overview.text]
        return "\n".join(lines) + "\n"

    def _titles(self, indices: tuple[int, ...], limit: int = 3) -> str:
        seen: list[str] = []
        for i in indices:
            title = str(self._r.chunks[i].metadata.get("title", "")).replace("|", "/")
            if title and title not in seen:
                seen.append(title)
            if len(seen) == limit:
                break
        return "; ".join(seen) or "-"
