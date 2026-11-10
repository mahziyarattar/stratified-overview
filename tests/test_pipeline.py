from conftest import FakeEmbedder, make_blobs

from stratified_overview.chunking import ParagraphChunker
from stratified_overview.config import OverviewConfig
from stratified_overview.generation import PromptPreviewGenerator
from stratified_overview.models import Document
from stratified_overview.pipeline import CorpusOverviewPipeline
from stratified_overview.reduction import IdentityReducer
from stratified_overview.report import PilotReport
from stratified_overview.tokens import ApproxTokenCounter


def test_end_to_end_on_synthetic_clusters(chunks_for):
    # Enough noise that members are not near-duplicates of each other.
    vectors, _ = make_blobs([120, 90, 60, 40, 30], noise=0.2)
    chunks = chunks_for(len(vectors))
    config = OverviewConfig(budget_tokens=4_000, max_tokens_per_chunk=200, min_clusters=3, max_clusters=10)
    pipeline = CorpusOverviewPipeline.with_defaults(
        config,
        FakeEmbedder(vectors),
        PromptPreviewGenerator(),
        n_chunks=len(chunks),
        reducer=IdentityReducer(),
    )

    result = pipeline.run(chunks)

    assert len(result.clusters) == 5
    assert result.tokens_selected <= config.budget_tokens
    assert all(len(s.chunk_indices) >= result.min_chunks_per_cluster for s in result.samples)
    assert result.overview.text.count("<group ") == len(result.clusters)
    assert "id=" not in result.overview.text  # cluster numbers never reach the model
    assert "| Cluster |" in PilotReport(result).to_markdown()


def test_chunker_respects_max_tokens():
    counter = ApproxTokenCounter()
    text = "\n".join(["short paragraph."] * 50 + [" ".join(["long"] * 400)])
    chunks = ParagraphChunker(100, counter).chunk([Document(id="d", text=text)])
    assert all(counter.count(c.text) <= 100 for c in chunks)
    assert len(chunks) > 1
