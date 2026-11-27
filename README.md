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

## Evaluation

`benchmarks/` compares three ways to give an answering model context on MultiHop-RAG:
this overview pipeline, a k-means-centroid baseline at the same budget, and top-k
retrieval. Questions and scoring come from Microsoft
[BenchmarkQED](https://github.com/microsoft/benchmark-qed) (MIT). AutoQ generated 50
questions in each of three sets (`data_local`, `data_global`, `activity_global`). AutoE
scored the answers with an LLM judge: pairwise on comprehensiveness, diversity,
empowerment, and relevance (4 trials, counterbalanced order), and against assertions for
`data_global`. The prompts in `benchmarks/*/prompts/` are adapted from BenchmarkQED.

Before trusting the judge, an A/A check (`benchmarks/autoe/aa_check/`) compared the
overview answers with an identical copy; every criterion came out at 0.5.
`bounded_judge.py` caps concurrent judge requests so rate limits do not drop calls. The
judge endpoint and model are read from `JUDGE_API_BASE` and `JUDGE_MODEL`.

Result: top-k retrieval won clearly on every criterion and question set, and on
assertions (pass rate 0.86 vs 0.63 on `data_global`). Against the k-means baseline the
overview did better on comprehensiveness and diversity for `activity_global` and worse
on relevance for `data_local`. Per-question scores are in `benchmarks/autoe/*/output/`.

## Tests

```bash
pytest
```

## Data

MultiHop-RAG (Tang and Yang, 2024) is licensed ODC-By:
https://huggingface.co/datasets/yixuantt/MultiHopRAG

## License

MIT. See `LICENSE`.
