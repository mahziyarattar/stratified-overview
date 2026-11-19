"""Answer the AutoQ questions with every method, writing the files AutoE reads.

Methods share the corpus, embeddings, answering model and context budget;
only the choice of context differs:

    top-k-rag            chunks most similar to the question
    kmeans-centroids     nearest-to-centroid chunks of a fixed-k k-means (query-agnostic)
    stratified-overview  our prebuilt overview plus the stratified sample behind it

Resumable: answers already on disk are skipped. Smoke test first:
    python benchmarks/run_answers.py --limit 2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from stratified_overview.answering import (
    AnswerStore,
    BenchmarkRunner,
    KMeansCentroidSelector,
    OverviewSelector,
    QuestionAnswerer,
    TopKSelector,
    load_autoq_questions,
)
from stratified_overview.config import OverviewConfig
from stratified_overview.embedding import CachedEmbedder, SentenceTransformerEmbedder, l2_normalize
from stratified_overview.excerpts import ExcerptFormatter
from stratified_overview.experiment import load_dotenv, prepare_multihop_corpus
from stratified_overview.generation import OpenRouterOverviewGenerator, estimate_cost
from stratified_overview.llm import OpenRouterChatClient
from stratified_overview.pipeline import CorpusOverviewPipeline
from stratified_overview.tokens import TiktokenCounter, TokenCalibration

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=Path("benchmarks/autoq/output"))
    parser.add_argument("--question-sets", nargs="+", default=["data_global", "activity_global", "data_local"])
    parser.add_argument("--limit", type=int, default=None, help="answer only the first N questions per set")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/answers"))
    parser.add_argument("--model", default="anthropic/claude-haiku-5.5")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--budget", type=int, default=40_000, help="context tokens per method")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--kmeans-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    return parser.parse_args()


def build_overview(args, corpus, embedder, counter) -> dict:
    """Run our pipeline once and keep the overview and its sample on disk."""
    path = args.out / "overview.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    config = OverviewConfig(budget_tokens=args.budget, max_tokens_per_chunk=args.chunk_tokens)
    pipeline = CorpusOverviewPipeline.with_defaults(
        config,
        embedder,
        OpenRouterOverviewGenerator(model=args.model),
        n_chunks=len(corpus.chunks),
        token_counter=counter,
    )
    result = pipeline.run(corpus.chunks)
    record = {
        "overview": result.overview.text,
        "excerpts": [text for section in result.request.sections for text in section.excerpts],
        "input_tokens": result.overview.input_tokens,
        "output_tokens": result.overview.output_tokens,
        "estimated_cost_usd": result.overview.estimated_cost_usd,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def copy_assertions(questions_dir: Path, out: Path, question_sets: list[str]) -> None:
    """Place AutoQ's fact lists where AutoE's assertion scoring looks: <out>/<set>_assertions.json."""
    for question_set in question_sets:
        source = questions_dir / f"{question_set}_questions" / "assertions.json"
        if source.exists():
            shutil.copyfile(source, out / f"{question_set}_assertions.json")


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    counter = TiktokenCounter(scale=TokenCalibration(args.cache_dir / "token_calibration.json").scale_for(args.model))
    corpus = prepare_multihop_corpus(args.model, args.chunk_tokens)
    embedder = CachedEmbedder(SentenceTransformerEmbedder(args.embedding_model), args.cache_dir)
    embeddings = l2_normalize(np.asarray(embedder.embed([c.text for c in corpus.chunks]), dtype=np.float32))
    excerpts = ExcerptFormatter(counter, args.chunk_tokens)

    overview = build_overview(args, corpus, embedder, counter)
    selectors = [
        TopKSelector(corpus.chunks, embeddings, embedder, excerpts, args.budget, query_prefix=BGE_QUERY_PREFIX),
        KMeansCentroidSelector(corpus.chunks, embeddings, excerpts, args.budget, n_clusters=args.kmeans_k),
        OverviewSelector(overview["overview"], overview["excerpts"]),
    ]
    questions = [q for qs in args.question_sets for q in load_autoq_questions(args.questions, qs, args.limit)]
    logging.info("%d questions x %d methods", len(questions), len(selectors))

    runner = BenchmarkRunner(
        QuestionAnswerer(OpenRouterChatClient(args.model)), selectors, AnswerStore(args.out), workers=args.workers
    )
    answers = runner.run(questions)
    copy_assertions(args.questions, args.out, args.question_sets)

    usage: dict[str, dict[str, float]] = {}
    for answer in answers:
        row = usage.setdefault(answer.method, {"answers": 0, "input_tokens": 0, "output_tokens": 0})
        row["answers"] += 1
        row["input_tokens"] += answer.input_tokens
        row["output_tokens"] += answer.output_tokens
    for row in usage.values():
        row["estimated_cost_usd"] = estimate_cost(args.model, row["input_tokens"], row["output_tokens"]) or 0.0
    summary = {"overview": {k: overview[k] for k in ("input_tokens", "output_tokens", "estimated_cost_usd")}, **usage}
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
