import numpy as np

from stratified_overview.answering import (
    AnswerStore,
    BenchmarkRunner,
    KMeansCentroidSelector,
    OverviewSelector,
    Question,
    QuestionAnswerer,
    TopKSelector,
)
from stratified_overview.embedding import l2_normalize
from stratified_overview.excerpts import ExcerptFormatter
from stratified_overview.llm import ChatResult
from stratified_overview.models import Chunk
from stratified_overview.tokens import ApproxTokenCounter


class EchoClient:
    """Answers with the number of excerpts it was shown."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=16_000):
        self.calls += 1
        return ChatResult(text=f"{user.count('<excerpt>')} excerpts", model="echo", input_tokens=10, output_tokens=2)


class QueryEmbedder:
    name = "query"

    def embed(self, texts):
        return np.array([[1.0, 0.0, 0.0]])


def corpus():
    vectors = l2_normalize(np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    chunks = [Chunk(id=str(i), doc_id=str(i), text=" ".join(["w"] * 40), metadata={"title": f"T{i}"}) for i in range(4)]
    return chunks, vectors, ExcerptFormatter(ApproxTokenCounter(), 512)


def test_top_k_takes_most_similar_within_budget():
    chunks, vectors, excerpts = corpus()
    one_chunk = excerpts.tokens(chunks[0])
    context = TopKSelector(chunks, vectors, QueryEmbedder(), excerpts, budget_tokens=2 * one_chunk).select("q")
    assert [e.split("\n")[0] for e in context.excerpts] == ["[T0]", "[T1]"]


def test_kmeans_context_is_query_agnostic_and_spans_clusters():
    chunks, vectors, excerpts = corpus()
    selector = KMeansCentroidSelector(chunks, vectors, excerpts, budget_tokens=10_000, n_clusters=3)
    assert selector.select("a") == selector.select("b")
    assert len(selector.select("a").excerpts) == 4


def test_overview_context_is_included_in_prompt():
    context = OverviewSelector("THE OVERVIEW", ["e1", "e2"]).select("q")
    prompt = QuestionAnswerer.build_prompt(Question("1", "What is covered?", "s"), context)
    assert prompt.index("THE OVERVIEW") < prompt.index("<excerpts>") < prompt.index("What is covered?")


def test_runner_resumes_from_stored_answers(tmp_path):
    client = EchoClient()
    store = AnswerStore(tmp_path)
    runner = BenchmarkRunner(QuestionAnswerer(client), [OverviewSelector("o", ["e"])], store, workers=2)
    questions = [Question(str(i), f"q{i}", "data_global") for i in range(3)]

    runner.run(questions)
    runner.run(questions)

    assert client.calls == 3
    stored = store.load("stratified-overview", "data_global")
    assert set(stored) == {"0", "1", "2"} and stored["0"]["answer"] == "1 excerpts"
