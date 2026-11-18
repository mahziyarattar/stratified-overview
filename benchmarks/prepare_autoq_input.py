"""Export the cleaned MultiHop-RAG articles as AutoQ input (one JSON file per article).

Questions are generated from the same text the methods see: the boilerplate
filter runs before export.

    python benchmarks/prepare_autoq_input.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stratified_overview.cleaning import RepeatedParagraphFilter
from stratified_overview.corpora import MultiHopRagCorpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/autoq/input"))
    args = parser.parse_args()

    cleaning = RepeatedParagraphFilter(min_documents=3).clean(MultiHopRagCorpus().load())
    args.out.mkdir(parents=True, exist_ok=True)
    for doc in cleaning.documents:
        record = {
            "id": doc.id,
            "text": doc.text,
            "title": doc.metadata.get("title", ""),
            "published_at": str(doc.metadata.get("published_at", "")),
            "source": doc.metadata.get("source", ""),
        }
        (args.out / f"{doc.id}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(cleaning.documents)} articles to {args.out} ({cleaning.paragraphs_removed} paragraphs removed)")


if __name__ == "__main__":
    main()
