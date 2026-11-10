import numpy as np

from stratified_overview.embedding import l2_normalize
from stratified_overview.models import Cluster
from stratified_overview.selection import CentroidOnly, DiverseMMR


def setup():
    # 0: on the centroid; 1: near-duplicate of 0; 2: a bit off-centre; 3: a different direction.
    vectors = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.01, 0.0],
                [1.0, 0.3, 0.0],
                [0.7, 0.0, 0.7],
            ]
        )
    )
    centroid = l2_normalize(vectors.mean(axis=0, keepdims=True))[0]
    cluster = Cluster(
        label=0, member_indices=np.arange(4), centroid=centroid, spread=0.1, total_tokens=400
    )
    return cluster, vectors, np.full(4, 100)


def test_diverse_skips_duplicates_and_reaches_other_direction():
    cluster, vectors, lengths = setup()
    sample = DiverseMMR(diversity=0.5, duplicate_threshold=0.95).select(cluster, vectors, lengths, 300)
    assert 1 not in sample.chunk_indices or 0 not in sample.chunk_indices
    assert 3 in sample.chunk_indices


def test_centroid_only_takes_most_typical_in_order():
    cluster, vectors, lengths = setup()
    sample = CentroidOnly().select(cluster, vectors, lengths, 200)
    typicality = vectors @ cluster.centroid
    assert list(sample.chunk_indices) == list(np.argsort(-typicality)[:2])


def test_respects_token_budget():
    cluster, vectors, lengths = setup()
    sample = DiverseMMR().select(cluster, vectors, lengths, 250)
    assert sample.tokens_used <= 250
    assert len(sample.chunk_indices) == 2
