"""Loaders for benchmark corpora."""

from __future__ import annotations

from stratified_overview.models import Document

_METADATA_FIELDS = ("title", "source", "published_at", "category", "url", "author")


class MultiHopRagCorpus:
    """The MultiHop-RAG news corpus (Tang and Yang, 2024), licensed ODC-By.

    Downloaded from the Hugging Face Hub on first use and cached by
    ``datasets``. Attribution: https://huggingface.co/datasets/yixuantt/MultiHopRAG
    """

    dataset_id = "yixuantt/MultiHopRAG"
    config_name = "corpus"

    def load(self, limit: int | None = None) -> list[Document]:
        from datasets import load_dataset

        loaded = load_dataset(self.dataset_id, self.config_name)
        split = loaded[next(iter(loaded))] if hasattr(loaded, "keys") else loaded
        documents: list[Document] = []
        for i, row in enumerate(split):
            if limit is not None and i >= limit:
                break
            text = row.get("body") or row.get("text") or ""
            if not text.strip():
                continue
            metadata = {k: row[k] for k in _METADATA_FIELDS if row.get(k)}
            documents.append(Document(id=f"mhr-{i}", text=text, metadata=metadata))
        return documents
