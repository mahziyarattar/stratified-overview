# stratified-overview

Cheap corpus-level overviews for RAG. The pipeline clusters every chunk, divides a
token budget across clusters by size and spread (a dampened Neyman allocation), picks
diverse chunks inside each cluster, and asks one LLM call to write an overview of what
the database contains.

## Pipeline

| Step | Module | Default |
| --- | --- | --- |
| 1. Embed and normalize | `embedding.py` | `BAAI/bge-small-en-v1.5` (local) |
| 2. Reduce dimensions | `reduction.py` | UMAP to 5 dims |
| 3-4. Cluster, keep count in bounds | `clustering.py` | HDBSCAN; finer settings, k-means fallback, or merges |
| 5. Handle noise | `noise.py` | assign to nearest cluster |
| 6. Score quality | `quality.py` | per-cluster silhouette (sampled) |
| 7a. Fit the budget | `budget.py` | lower min to 1 chunk, then merge smallest |
| 7b. Allocate tokens | `allocation.py` | reserve minimums, share the rest by sqrt(size) x spread, cap at 40% |
| 7c. Select chunks | `selection.py` | most typical first, then MMR with near-duplicate skipping |
| 8. Write the overview | `generation.py` | Claude Haiku 5.5, one call |

Every step is an object injected into `CorpusOverviewPipeline`, so an ablation swaps
one argument:

```python
pipeline = CorpusOverviewPipeline.with_defaults(
    config, embedder, generator, n_chunks=len(chunks),
    selection=CentroidOnly(),  # ablation: no diversity
)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[pilot,dev]"
```

## Pilot on MultiHop-RAG

Dry run: downloads the corpus and embedding model, clusters, allocates, and writes the
prompt that would be sent. No API key, no cost.

```bash
python pilots/run_multihop_pilot.py --dry-run
```

Full run: one Claude Haiku 5.5 call. Needs `ANTHROPIC_API_KEY` (or an `ant auth login`
profile).

```bash
python pilots/run_multihop_pilot.py
```

Outputs go to `pilot_output/`: `report.md` (clusters, allocation, typical chunks),
`summary.json`, and `prompt.txt` or `overview.md`. Embeddings are cached in `.cache/`.

What to check in the report:

- Do the clusters look like topics, or like formatting and boilerplate?
- What share of chunks did HDBSCAN label as noise?
- Does the overview read as a fair summary of the corpus?

## Tests

```bash
pytest
```

## Data

MultiHop-RAG (Tang and Yang, 2024) is licensed ODC-By:
https://huggingface.co/datasets/yixuantt/MultiHopRAG
