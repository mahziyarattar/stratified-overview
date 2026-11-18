"""Benchmark answering: three ways to pick context, one shared way to answer.

Every method gets the same context budget and the same answering model, so
the comparison isolates how chunks are chosen.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from stratified_overview.embedding import Embedder, l2_normalize
from stratified_overview.excerpts import ExcerptFormatter
from stratified_overview.llm import ChatClient
from stratified_overview.models import Chunk

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """\
Answer the question using only the material provided from a document database. \
The question may be about the database as a whole: draw on all relevant material, \
name specific entities, events and dates, and say what the material does not cover \
when that matters for the answer. Do not mention excerpts, the overview, or how the \
material was selected."""


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str
    question_set: str


@dataclass(frozen=True, slots=True)
class Context:
    excerpts: tuple[str, ...]
    overview: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    question_id: str
    question_text: str
    answer: str
    method: str
    input_tokens: int
    output_tokens: int


def load_autoq_questions(output_dir: Path, question_set: str, limit: int | None = None) -> list[Question]:
    """Read AutoQ's ``<set>_questions/selected_questions.json``."""
    path = output_dir / f"{question_set}_questions" / "selected_questions.json"
    records = json.loads(path.read_text(encoding="utf-8"))[:limit]
    return [Question(id=r["id"], text=r["text"], question_set=question_set) for r in records]


class ContextSelector(Protocol):
    name: str

    def select(self, question: str) -> Context: ...


def _fill_budget(
    order: Sequence[int], chunks: Sequence[Chunk], costs: np.ndarray, excerpts: ExcerptFormatter, budget: int
) -> tuple[str, ...]:
    """Take chunks in the given order while they fit, until no chunk could fit."""
    picked: list[str] = []
    remaining = budget
    smallest = int(costs.min()) if costs.size else 0
    for i in order:
        if costs[i] <= remaining:
            picked.append(excerpts.format(chunks[i]))
            remaining -= int(costs[i])
        if remaining < smallest:
            break
    return tuple(picked)


class TopKSelector:
    """Plain RAG: the chunks most similar to the question, until the budget is full.

    ``query_prefix`` is the instruction some embedding models expect on queries
    (BGE: "Represent this sentence for searching relevant passages: ").
    """

    name = "top-k-rag"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        excerpts: ExcerptFormatter,
        budget_tokens: int,
        query_prefix: str = "",
    ) -> None:
        self._chunks = chunks
        self._embeddings = embeddings
        self._embedder = embedder
        self._query_prefix = query_prefix
        self._excerpts = excerpts
        self._budget = budget_tokens
        self._costs = np.array([excerpts.tokens(c) for c in chunks])

    def select(self, question: str) -> Context:
        vector = self._embedder.embed([self._query_prefix + question])
        query = l2_normalize(np.asarray(vector, dtype=np.float32))[0]
        order = np.argsort(-(self._embeddings @ query))
        return Context(excerpts=_fill_budget(order, self._chunks, self._costs, self._excerpts, self._budget))


class KMeansCentroidSelector:
    """Fixed-k k-means baseline: nearest-to-centroid chunks, round-robin across clusters.

    Query-agnostic: computed once and reused for every question.
    """

    name = "kmeans-centroids"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        excerpts: ExcerptFormatter,
        budget_tokens: int,
        n_clusters: int = 20,
        random_state: int | None = 42,
    ) -> None:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(embeddings)
        distances = model.transform(embeddings)
        queues = [
            list(members[np.argsort(distances[members, k])])
            for k in range(n_clusters)
            if (members := np.flatnonzero(model.labels_ == k)).size
        ]
        order: list[int] = []
        while any(queues):
            order.extend(queue.pop(0) for queue in queues if queue)
        costs = np.array([excerpts.tokens(c) for c in chunks])
        self._context = Context(excerpts=_fill_budget(order, chunks, costs, excerpts, budget_tokens))

    def select(self, question: str) -> Context:
        return self._context


class OverviewSelector:
    """Our method: the prebuilt overview plus the stratified sample it was written from."""

    name = "stratified-overview"

    def __init__(self, overview: str, excerpts: Sequence[str]) -> None:
        self._context = Context(excerpts=tuple(excerpts), overview=overview)

    def select(self, question: str) -> Context:
        return self._context


class QuestionAnswerer:
    def __init__(self, client: ChatClient, max_tokens: int = 16_000) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def answer(self, question: Question, context: Context, method: str) -> Answer:
        result = self._client.complete(ANSWER_SYSTEM_PROMPT, self.build_prompt(question, context), self._max_tokens)
        return Answer(
            question_id=question.id,
            question_text=question.text,
            answer=result.text,
            method=method,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    @staticmethod
    def build_prompt(question: Question, context: Context) -> str:
        parts = []
        if context.overview:
            parts.append(f"<overview>\n{context.overview}\n</overview>")
        parts.append("<excerpts>")
        parts.extend(f"<excerpt>\n{text}\n</excerpt>" for text in context.excerpts)
        parts.append("</excerpts>")
        parts.append(f"<question>\n{question.text}\n</question>")
        return "\n".join(parts)


class AnswerStore:
    """Answers on disk as ``<base>/<method>/<question_set>.json``, the layout AutoE reads."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def load(self, method: str, question_set: str) -> dict[str, dict]:
        path = self._path(method, question_set)
        if not path.exists():
            return {}
        return {record["question_id"]: record for record in json.loads(path.read_text(encoding="utf-8"))}

    def save(self, method: str, question_set: str, records: dict[str, dict]) -> None:
        path = self._path(method, question_set)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(records.values()), indent=2, ensure_ascii=False), encoding="utf-8")

    def _path(self, method: str, question_set: str) -> Path:
        return self._base / method / f"{question_set}.json"


class BenchmarkRunner:
    """Answers every question with every method; resumes by skipping stored answers."""

    def __init__(
        self,
        answerer: QuestionAnswerer,
        selectors: Sequence[ContextSelector],
        store: AnswerStore,
        workers: int = 8,
        on_answer: Callable[[Answer], None] | None = None,
    ) -> None:
        self._answerer = answerer
        self._selectors = selectors
        self._store = store
        self._workers = workers
        self._on_answer = on_answer

    def run(self, questions: Sequence[Question]) -> list[Answer]:
        answers: list[Answer] = []
        for selector in self._selectors:
            for question_set in sorted({q.question_set for q in questions}):
                answers.extend(self._run_set(selector, [q for q in questions if q.question_set == question_set]))
        return answers

    def _run_set(self, selector: ContextSelector, questions: Sequence[Question]) -> list[Answer]:
        question_set = questions[0].question_set
        records = self._store.load(selector.name, question_set)
        todo = [q for q in questions if q.id not in records]
        logger.info("%s / %s: %d to answer, %d stored", selector.name, question_set, len(todo), len(records))

        def work(question: Question) -> Answer:
            return self._answerer.answer(question, selector.select(question.text), selector.name)

        answers: list[Answer] = []
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            for answer in pool.map(work, todo):
                records[answer.question_id] = asdict(answer)
                answers.append(answer)
                if self._on_answer:
                    self._on_answer(answer)
                self._store.save(selector.name, question_set, records)
        return answers
