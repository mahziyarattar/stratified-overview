import numpy as np

from stratified_overview.budget import BudgetFitter
from stratified_overview.clustering import ClusterCountEnforcer, HdbscanClusterer, cluster_labels
from stratified_overview.config import OverviewConfig
from stratified_overview.models import NOISE_LABEL
from stratified_overview.noise import AssignToNearest, KeepAsMisc


def test_hdbscan_finds_the_blobs(blobs):
    vectors, truth = blobs
    outcome = ClusterCountEnforcer(HdbscanClusterer(min_cluster_size=10), 2, 50).cluster(vectors, vectors)
    assert outcome.n_clusters == len(np.unique(truth))


def test_too_many_clusters_trigger_coarser_hdbscan(blobs):
    vectors, _ = blobs
    outcome = ClusterCountEnforcer(HdbscanClusterer(min_cluster_size=10), 2, 3).cluster(vectors, vectors)
    assert outcome.n_clusters == 3
    assert any(step.startswith("too many") for step in outcome.steps)


def test_falls_back_to_kmeans_when_too_few(blobs):
    vectors, _ = blobs
    # A min size larger than any blob leaves HDBSCAN with almost nothing.
    enforcer = ClusterCountEnforcer(HdbscanClusterer(min_cluster_size=200), 8, 50, max_refinements=1)
    outcome = enforcer.cluster(vectors, vectors)
    assert outcome.n_clusters == 8
    assert any("KMeans" in step for step in outcome.steps)


def test_assign_to_nearest_moves_noise_into_true_cluster(blobs):
    vectors, truth = blobs
    labels = truth.copy()
    labels[[0, 130]] = NOISE_LABEL
    outcome = AssignToNearest().handle(labels, vectors)
    assert outcome.noise_count == 2
    assert outcome.labels[0] == truth[0] and outcome.labels[130] == truth[130]


def test_keep_as_misc_adds_one_cluster(blobs):
    vectors, truth = blobs
    labels = truth.copy()
    labels[:5] = NOISE_LABEL
    outcome = KeepAsMisc().handle(labels, vectors)
    assert outcome.misc_label == truth.max() + 1
    assert len(cluster_labels(outcome.labels)) == len(np.unique(truth)) + 1


def test_budget_fitter_lowers_minimum_then_merges():
    vectors = np.eye(12)
    labels = np.arange(12)  # 12 singleton clusters
    config = OverviewConfig(budget_tokens=2_048, max_tokens_per_chunk=256, min_chunks_per_cluster=2, min_clusters=4)
    fit = BudgetFitter(config).fit(labels, vectors, np.full(12, 256))
    assert fit.min_chunks_per_cluster == 1
    assert len(cluster_labels(fit.labels)) == 8  # 8 x 256 = 2,048 fits exactly


def test_budget_fitter_counts_actual_chunk_lengths():
    labels = np.arange(12)
    config = OverviewConfig(budget_tokens=2_048, max_tokens_per_chunk=256, min_chunks_per_cluster=2, min_clusters=4)
    fit = BudgetFitter(config).fit(labels, np.eye(12), np.full(12, 100))
    # 12 x 1 x 100 = 1,200 tokens fits once the minimum drops to 1; no merges needed.
    assert fit.min_chunks_per_cluster == 1
    assert len(cluster_labels(fit.labels)) == 12


def test_budget_fitter_leaves_fitting_config_alone(blobs):
    vectors, truth = blobs
    fit = BudgetFitter(OverviewConfig()).fit(truth, vectors, np.full(len(truth), 512))
    assert fit.min_chunks_per_cluster == 2 and fit.steps == ()
    assert np.array_equal(fit.labels, truth)
