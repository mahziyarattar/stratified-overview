import numpy as np

from stratified_overview.allocation import EqualWeights, SqrtSizeSpread, TokenAllocator
from stratified_overview.models import Cluster


def cluster(label: int, size: int, spread: float, tokens_per_chunk: int = 500) -> Cluster:
    return Cluster(
        label=label,
        member_indices=np.arange(size),
        centroid=np.zeros(4),
        spread=spread,
        total_tokens=size * tokens_per_chunk,
    )


# The running example: tech, sports, finance, boilerplate, space.
EXAMPLE = [
    cluster(0, 4000, 0.40),
    cluster(1, 3000, 0.35),
    cluster(2, 2000, 0.38),
    cluster(3, 900, 0.05),
    cluster(4, 100, 0.30),
]


def allocator(budget=10_000, share=1.0, strategy=None) -> TokenAllocator:
    return TokenAllocator(strategy or SqrtSizeSpread(), budget, share)


def by_label(allocations):
    return {a.label: a.token_budget for a in allocations}


def test_total_matches_budget_when_uncapped():
    allocations = allocator().allocate(EXAMPLE, min_chunks=2)
    assert sum(a.token_budget for a in allocations) <= 10_000
    assert sum(a.token_budget for a in allocations) >= 10_000 - len(EXAMPLE)  # flooring loss only


def test_every_cluster_keeps_its_reserve():
    budgets = by_label(allocator().allocate(EXAMPLE, min_chunks=2))
    assert all(tokens >= 2 * 500 for tokens in budgets.values())


def test_spread_out_cluster_beats_tight_one_of_similar_size():
    budgets = by_label(allocator(budget=20_000).allocate(EXAMPLE, min_chunks=1))
    # Boilerplate (900 chunks, spread 0.05) gets less than space (100 chunks, spread 0.30).
    assert budgets[3] < budgets[4]


def test_share_cap_redistributes_leftover():
    budgets = by_label(allocator(budget=10_000, share=0.3).allocate(EXAMPLE, min_chunks=1))
    assert max(budgets.values()) <= 3_000
    assert sum(budgets.values()) >= 10_000 - len(EXAMPLE)


def test_cluster_never_gets_more_than_it_holds():
    small = cluster(9, 2, 0.9)  # holds 1,000 tokens
    budgets = by_label(allocator(budget=10_000, strategy=EqualWeights()).allocate([small, EXAMPLE[0]], 1))
    assert budgets[9] == 1_000


def test_reserve_uses_actual_chunk_lengths():
    short = cluster(5, 50, 0.2, tokens_per_chunk=100)
    budgets = by_label(allocator(budget=10_000, share=0.01).allocate([short, EXAMPLE[0]], min_chunks=2))
    # The share cap (100 tokens) is below the reserve, so the reserve is all it gets.
    assert budgets[5] == 200
