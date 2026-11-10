import pytest

from stratified_overview.config import ConfigError, OverviewConfig


def test_defaults_are_valid():
    config = OverviewConfig()
    assert config.budget_tokens >= config.min_clusters * config.max_tokens_per_chunk


def test_rejects_budget_below_minimum_cluster_floor():
    with pytest.raises(ConfigError, match="below min_clusters"):
        OverviewConfig(budget_tokens=2_000, max_tokens_per_chunk=512, min_clusters=5)


def test_rejects_inverted_cluster_bounds():
    with pytest.raises(ConfigError, match="exceeds max_clusters"):
        OverviewConfig(min_clusters=10, max_clusters=5)


@pytest.mark.parametrize("share", [0.0, 1.5])
def test_rejects_invalid_share(share):
    with pytest.raises(ConfigError):
        OverviewConfig(max_cluster_share=share)


def test_min_cluster_size_scales_with_corpus():
    config = OverviewConfig()
    assert config.resolved_min_cluster_size(100) == 10
    assert config.resolved_min_cluster_size(10_000) == 50
    assert OverviewConfig(min_cluster_size=7).resolved_min_cluster_size(10_000) == 7
