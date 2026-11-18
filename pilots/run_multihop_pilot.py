"""Pilot: run the pipeline on the MultiHop-RAG news corpus and report what it found.

Dry run (no API key, no cost): clusters, allocation and the prompt that would be sent.
    python pilots/run_multihop_pilot.py --dry-run

Full run (one Claude Haiku 5.5 call). Reads OPENROUTER_API_KEY or
ANTHROPIC_API_KEY from the environment or from a .env file in the working
directory; --provider picks one explicitly:
    python pilots/run_multihop_pilot.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from stratified_overview.allocation import EqualWeights, SizeProportional, SqrtSizeSpread
from stratified_overview.config import OverviewConfig
from stratified_overview.embedding import CachedEmbedder, SentenceTransformerEmbedder
from stratified_overview.experiment import check_token_estimate, load_dotenv, prepare_multihop_corpus
from stratified_overview.generation import (
    SYSTEM_PROMPT,
    ClaudeOverviewGenerator,
    OpenRouterOverviewGenerator,
    OverviewGenerator,
    OverviewPromptBuilder,
    PromptPreviewGenerator,
)
from stratified_overview.noise import AssignToNearest, DropNoise, KeepAsMisc
from stratified_overview.pipeline import CorpusOverviewPipeline
from stratified_overview.report import PilotReport
from stratified_overview.selection import CentroidOnly, DiverseMMR
from stratified_overview.tokens import TiktokenCounter, TokenCalibration

ALLOCATIONS = {"sqrt-spread": SqrtSizeSpread, "size": SizeProportional, "equal": EqualWeights}
NOISE_HANDLERS = {"assign": AssignToNearest, "misc": KeepAsMisc, "drop": DropNoise}
SELECTIONS = {"diverse": DiverseMMR, "centroid": CentroidOnly}
DEFAULT_MODELS = {"openrouter": "anthropic/claude-haiku-5.5", "anthropic": "claude-haiku-5-5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="build the prompt but do not call the API")
    parser.add_argument("--provider", choices=["auto", *DEFAULT_MODELS], default="auto")
    parser.add_argument("--model", default=None, help="defaults to Claude Haiku 5.5 for the provider")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--budget", type=int, default=40_000, help="total tokens the overview sees")
    parser.add_argument("--chunk-tokens", type=int, default=512, help="chunk size and per-chunk cap")
    parser.add_argument("--min-clusters", type=int, default=5)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--allocation", choices=ALLOCATIONS, default="sqrt-spread")
    parser.add_argument("--noise", choices=NOISE_HANDLERS, default="assign")
    parser.add_argument("--selection", choices=SELECTIONS, default="diverse")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N articles")
    parser.add_argument("--out", type=Path, default=Path("pilot_output"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    return parser.parse_args()


def resolve_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    raise SystemExit("No API key: set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, or use --dry-run")


def resolve_model(args: argparse.Namespace) -> tuple[str, str]:
    """Return (provider, model); a dry run prices with the provider it would use."""
    try:
        provider = resolve_provider(args.provider)
    except SystemExit:
        if not args.dry_run:
            raise
        provider = "openrouter"
    return provider, args.model or DEFAULT_MODELS[provider]


def build_generator(
    args: argparse.Namespace, provider: str, model: str, counter: TiktokenCounter
) -> OverviewGenerator:
    if args.dry_run:
        return PromptPreviewGenerator(counter, priced_as=model)
    if provider == "openrouter":
        return OpenRouterOverviewGenerator(model=model)
    return ClaudeOverviewGenerator(model=model)


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    config = OverviewConfig(
        budget_tokens=args.budget,
        max_tokens_per_chunk=args.chunk_tokens,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    provider, model = resolve_model(args)
    calibration = TokenCalibration(args.cache_dir / "token_calibration.json")
    counter = TiktokenCounter(scale=calibration.scale_for(model))
    logging.info("counting tokens for %s with scale %.3f", model, counter.scale)

    corpus = prepare_multihop_corpus(model, args.chunk_tokens, limit=args.limit)
    chunks = corpus.chunks
    logging.info(
        "loaded %d articles; removed %d boilerplate paragraphs -> %d chunks",
        len(corpus.documents),
        corpus.paragraphs_removed,
        len(chunks),
    )

    embedder = CachedEmbedder(SentenceTransformerEmbedder(args.embedding_model), args.cache_dir)
    generator = build_generator(args, provider, model, counter)
    pipeline = CorpusOverviewPipeline.with_defaults(
        config,
        embedder,
        generator,
        n_chunks=len(chunks),
        allocation=ALLOCATIONS[args.allocation](),
        noise_handler=NOISE_HANDLERS[args.noise](),
        selection=SELECTIONS[args.selection](),
        token_counter=counter,
    )
    result = pipeline.run(chunks)

    report = PilotReport(result)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    if args.dry_run:
        # The preview generator returns the rendered prompt as its text.
        (args.out / "prompt.txt").write_text(result.overview.text, encoding="utf-8")
    else:
        (args.out / "overview.md").write_text(result.overview.text, encoding="utf-8")
        prompt = OverviewPromptBuilder().build(result.request)
        texts = [SYSTEM_PROMPT, prompt]
        logging.info(check_token_estimate(calibration, model, counter, texts, result.overview.input_tokens))

    print(report.to_markdown(include_overview=not args.dry_run))
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
