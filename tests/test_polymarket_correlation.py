"""Tests for measuring how much of a book is really one bet.

The central claim these pin is a negative one: price co-movement does not imply
joint settlement, so a book held to settlement cannot be sized from it. That
claim came out of measurement rather than reasoning, and it is easy to undo by
someone who "fixes" the clustering to feed sizing, so the evidence lives here.
"""

import numpy as np
import pandas as pd
import pytest

from quant_system.polymarket.correlation import (
    block_correlation,
    cluster_markets,
    clusters_from_events,
    concentration_report,
    correlation_matrix,
    effective_independent_bets,
    logit_price_changes,
    outcome_concordance,
)
from quant_system.polymarket.kelly import allocate_exposure
from quant_system.polymarket.simulation import PanelSpec, simulate_panel

IDS = [f"m{i}" for i in range(4)]


def test_independent_positions_count_as_themselves():
    assert effective_independent_bets(pd.Series(0.05, index=IDS)) == pytest.approx(4.0)


def test_perfectly_correlated_positions_are_one_bet():
    ones = pd.DataFrame(1.0, index=IDS, columns=IDS)
    assert effective_independent_bets(pd.Series(0.05, index=IDS), ones) == pytest.approx(1.0)


def test_partial_correlation_lands_between():
    matrix = pd.DataFrame(0.5, index=IDS, columns=IDS)
    np.fill_diagonal(matrix.values, 1.0)
    # (sum w)^2 / (w'Cw) = 16 / (4 + 12*0.5)
    assert effective_independent_bets(pd.Series(0.05, index=IDS), matrix) == pytest.approx(1.6)


def test_opposite_sides_of_correlated_markets_hedge_rather_than_concentrate():
    ones = pd.DataFrame(1.0, index=IDS, columns=IDS)
    sides = {"m0": "yes", "m1": "yes", "m2": "no", "m3": "no"}
    # A concentration measure blind to the side held would call this the most
    # dangerous book possible; it carries no directional risk at all.
    assert effective_independent_bets(pd.Series(0.05, index=IDS), ones, sides) == float("inf")


def test_an_empty_book_has_no_bets():
    assert effective_independent_bets(pd.Series(dtype=float)) == 0.0


def test_block_correlation_only_links_group_members():
    groups = pd.Series({"a": "G1", "b": "G1", "c": "G2"})
    matrix = block_correlation(groups, 0.3)
    assert matrix.loc["a", "b"] == pytest.approx(0.3)
    assert matrix.loc["a", "c"] == 0.0
    assert matrix.loc["a", "a"] == 1.0


def _settled(panel):
    return panel.groupby("market_id").agg(label=("label", "first"), group=("group", "first"))


def test_concordance_detects_a_grouping_that_predicts_joint_settlement():
    panel = simulate_panel(
        PanelSpec(n_markets=600, observations_per_market=8, n_themes=8, theme_strength=0.7, seed=42)
    )
    settled = _settled(panel)
    result = outcome_concordance(settled["label"], settled["group"], n_permutations=200, seed=3)
    assert result["within_concordance"] > result["cross_concordance"]
    assert result["implied_within_correlation"] > 0.05
    assert result["p_value"] < 0.01


def test_concordance_clears_a_grouping_that_predicts_nothing():
    panel = simulate_panel(
        PanelSpec(n_markets=600, observations_per_market=8, n_themes=8, theme_strength=0.7, seed=42)
    )
    settled = _settled(panel)
    # Same group sizes, membership shuffled: the structure is gone, the shape is not.
    shuffled = pd.Series(
        np.random.default_rng(7).permutation(settled["group"].to_numpy()), index=settled.index
    )
    result = outcome_concordance(settled["label"], shuffled, n_permutations=200, seed=3)
    assert abs(result["implied_within_correlation"]) < 0.05
    assert result["p_value"] > 0.05


def test_an_independent_panel_shows_no_settlement_structure():
    panel = simulate_panel(PanelSpec(n_markets=600, observations_per_market=8, seed=42))
    settled = _settled(panel)
    result = outcome_concordance(settled["label"], settled["group"], n_permutations=100, seed=3)
    # Every market is its own group, so there are no within-group pairs at all.
    assert not np.isfinite(result["within_concordance"]) or result["p_value"] > 0.05


def test_price_clusters_do_not_predict_joint_settlement():
    # The finding that decided the design. Clustering on price co-movement is
    # left available for the mark-to-market case and must never be wired into
    # sizing for a book held to settlement without this test being re-run.
    panel = simulate_panel(
        PanelSpec(n_markets=400, observations_per_market=12, horizon_days=90,
                  n_themes=4, theme_strength=0.7, seed=42)
    )
    corr = correlation_matrix(logit_price_changes(panel), min_overlap=8)
    clusters = cluster_markets(corr, threshold=0.5)
    settled = _settled(panel)
    result = outcome_concordance(
        settled["label"], pd.Series(clusters), n_permutations=200, seed=3
    )
    assert abs(result["implied_within_correlation"]) < 0.05
    assert result["p_value"] > 0.05


def test_price_clustering_groups_markets_that_are_not_related():
    # And the reason it must not be trusted: on a panel with no planted
    # structure at all it still puts nearly every market into a cluster.
    panel = simulate_panel(
        PanelSpec(n_markets=120, observations_per_market=14, horizon_days=90, seed=42)
    )
    corr = correlation_matrix(logit_price_changes(panel), min_overlap=8)
    clusters = cluster_markets(corr, threshold=0.5)
    sizes = pd.Series(list(clusters.values())).value_counts()
    assert (sizes > 1).sum() >= 1
    assert sizes.max() > 5


def test_changes_are_measured_on_a_common_grid():
    # Two markets quoted on different days share no timestamp, so differencing
    # the pivoted panel directly yields nothing and every correlation comes back
    # undefined -- which a caller would read as "uncorrelated".
    rows = []
    for market, offset in (("a", 0), ("b", 1)):
        for step in range(10):
            rows.append(
                {
                    "market_id": market,
                    "timestamp": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=offset + 2 * step),
                    "price": 0.4 + 0.02 * step,
                }
            )
    changes = logit_price_changes(pd.DataFrame(rows))
    assert not changes.empty
    overlap = changes.notna().all(axis=1).sum()
    assert overlap >= 5


def test_forward_fill_stops_at_a_market_s_last_quote():
    rows = [
        {"market_id": "short", "timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "price": 0.4},
        {"market_id": "short", "timestamp": pd.Timestamp("2026-01-02", tz="UTC"), "price": 0.5},
        {"market_id": "long", "timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "price": 0.3},
        {"market_id": "long", "timestamp": pd.Timestamp("2026-01-10", tz="UTC"), "price": 0.6},
    ]
    changes = logit_price_changes(pd.DataFrame(rows))
    # A settled market must not contribute a flat tail that drags correlations
    # toward zero for the rest of the panel.
    assert changes["short"].loc[pd.Timestamp("2026-01-05", tz="UTC") :].isna().all()


def test_clusters_from_events_never_leaves_a_market_ungrouped():
    mapping = clusters_from_events({"a": "E1", "b": "E1", "c": ""})
    assert mapping["a"] == mapping["b"] == "E1"
    assert mapping["c"] == "c"


def test_concentration_report_separates_count_from_value():
    matrix = block_correlation(pd.Series({m: "G" for m in IDS}), 0.9)
    report = concentration_report(
        pd.Series(0.05, index=IDS), matrix, {m: "G" for m in IDS}
    )
    assert report["n_positions"] == 4.0
    assert report["n_clusters"] == 1.0
    assert report["effective_independent_bets"] < 2.0
    assert report["largest_cluster_exposure"] == pytest.approx(0.20)


def test_cluster_cap_limits_a_correlated_group():
    fractions = {f"m{i}": 0.02 for i in range(10)}
    clusters = {f"m{i}": "ONE" for i in range(10)}
    scaled = allocate_exposure(fractions, 0.50, clusters, max_cluster_exposure=0.08)
    assert sum(scaled.values()) == pytest.approx(0.08)


def test_cluster_cap_leaves_a_diversified_book_alone():
    fractions = {f"m{i}": 0.02 for i in range(10)}
    clusters = {f"m{i}": f"G{i}" for i in range(10)}
    scaled = allocate_exposure(fractions, 0.50, clusters, max_cluster_exposure=0.08)
    assert scaled == pytest.approx(fractions)


def test_the_aggregate_cap_still_applies_after_cluster_trimming():
    fractions = {f"m{i}": 0.05 for i in range(10)}
    clusters = {f"m{i}": f"G{i % 2}" for i in range(10)}
    scaled = allocate_exposure(fractions, 0.10, clusters, max_cluster_exposure=0.08)
    assert sum(scaled.values()) == pytest.approx(0.10)
    by_group = {}
    for market, value in scaled.items():
        by_group[clusters[market]] = by_group.get(clusters[market], 0.0) + value
    assert max(by_group.values()) <= 0.08 + 1e-9
