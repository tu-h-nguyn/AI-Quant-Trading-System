import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from quant_system.polymarket.client import PolymarketClient, SnapshotStore, parse_book
from quant_system.polymarket.features import build_snapshot_features, design_matrix
from quant_system.polymarket.markets import group_by_event, parse_market, parse_markets
from quant_system.polymarket.metrics import (
    brier_skill_score,
    edge_realization,
    forecast_report,
)
from quant_system.polymarket.model import blend_with_market, market_anchored_model
from quant_system.polymarket.simulation import PanelSpec, simulate_panel
from quant_system.polymarket.validation import (
    resolution_aware_folds,
    resolution_aware_walk_forward,
)


class FakeTransport:
    """In-memory stand-in for the venue, so tests never touch the network."""

    def __init__(self, pages=1):
        self.pages = pages
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append(url)
        if url.endswith("/markets"):
            if params["offset"] >= self.pages * 2:
                return []
            return [_gamma_record(1, "t1", "t2"), _gamma_record(2, "t3", "t4")]
        if url.endswith("/book"):
            return {
                "asset_id": params["token_id"],
                "bids": [{"price": "0.40", "size": "100"}],
                "asks": [{"price": "0.42", "size": "50"}],
            }
        if url.endswith("/prices-history"):
            return {"history": [{"t": 1700000000, "p": 0.5}, {"t": 1700086400, "p": 0.6}]}
        return {}

    def post_json(self, url, body):
        raise RuntimeError("batch endpoint unavailable")


def _gamma_record(index, yes_token, no_token, closed=False, prices=("0.6", "0.4")):
    return {
        "id": str(index),
        "question": f"Question {index}?",
        "slug": f"question-{index}",
        "conditionId": f"0xcond{index}",
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([yes_token, no_token]),
        "outcomePrices": json.dumps(list(prices)),
        "closed": closed,
        "endDate": "2026-01-01T00:00:00Z",
        "volumeNum": 12_345,
        "events": [{"id": "ev1", "slug": "an-event", "negRisk": True}],
    }


def test_gamma_json_encoded_fields_are_decoded():
    market = parse_market(_gamma_record(1, "t1", "t2"))
    assert market.is_binary
    assert market.yes_outcome.token_id == "t1"
    assert market.no_outcome.token_id == "t2"
    assert market.neg_risk and market.event_id == "ev1"
    assert market.end_date == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_settled_prices_recover_the_binary_label():
    won = parse_market(_gamma_record(1, "t1", "t2", closed=True, prices=("1", "0")))
    lost = parse_market(_gamma_record(2, "t3", "t4", closed=True, prices=("0", "1")))
    assert won.is_resolved and won.yes_label == 1
    assert lost.is_resolved and lost.yes_label == 0


def test_an_open_market_has_no_label():
    market = parse_market(_gamma_record(1, "t1", "t2", closed=False, prices=("0.6", "0.4")))
    assert not market.is_resolved and market.yes_label is None


def test_malformed_records_are_dropped_rather_than_raising():
    assert parse_market({"id": "1", "outcomes": "not json"}) is None
    assert parse_markets([{"id": "1"}, _gamma_record(2, "a", "b")]) != []
    assert len(parse_markets([{"id": "1"}, _gamma_record(2, "a", "b")])) == 1


def test_days_to_resolution_is_floored_at_zero():
    market = parse_market(_gamma_record(1, "t1", "t2"))
    after = datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert market.days_to_resolution(after) == 0.0
    before = datetime(2025, 12, 25, tzinfo=timezone.utc)
    assert market.days_to_resolution(before) == pytest.approx(7.0)


def test_client_paginates_and_falls_back_when_batch_books_are_unavailable():
    client = PolymarketClient(transport=FakeTransport(), page_size=2)
    markets = client.fetch_markets(max_markets=4)
    assert len(markets) == 2
    assert client.order_book("t1").best_ask == 0.42
    assert sorted(client.order_books(["t1", "t2"])) == ["t1", "t2"]
    assert client.price_history("t1")[0]["p"] == 0.5


def test_books_are_sorted_and_filtered_on_parse():
    book = parse_book(
        {
            "asset_id": "t1",
            "bids": [{"price": "0.30", "size": "5"}, {"price": "0.40", "size": "5"}],
            "asks": [{"price": "0.60", "size": "0"}, {"price": "0.50", "size": "5"}],
        }
    )
    assert book.best_bid == 0.40
    assert book.best_ask == 0.50  # the zero-size level is discarded
    assert len(book.asks) == 1


def test_grouping_marks_a_partition_only_when_every_member_is_neg_risk():
    markets = parse_markets([_gamma_record(1, "a", "b"), _gamma_record(2, "c", "d")])
    groups = group_by_event(markets)
    assert len(groups) == 1 and groups[0].size == 2 and groups[0].exhaustive


def test_snapshot_store_round_trips(tmp_path):
    store = SnapshotStore(tmp_path)
    store.save("markets", [{"id": "1"}])
    assert store.load("markets") == [{"id": "1"}]
    with pytest.raises(FileNotFoundError):
        store.load("absent")


def test_snapshot_records_the_clock_a_replay_must_be_judged_by(tmp_path):
    # A replay evaluated at wall time would reject markets as "resolves too
    # soon" that were perfectly tradable when the data was captured.
    store = SnapshotStore(tmp_path)
    store.save("markets", [{"id": "1"}])
    captured = store.captured_at("markets")
    assert captured is not None and captured.tzinfo is not None
    assert (datetime.now(timezone.utc) - captured).total_seconds() < 60


def _panel():
    rows = []
    for market in range(4):
        for step in range(10):
            rows.append(
                {
                    "market_id": f"m{market}",
                    "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=step),
                    "price": 0.4 + 0.01 * step,
                    "best_bid": 0.39 + 0.01 * step,
                    "best_ask": 0.41 + 0.01 * step,
                    "bid_depth": 100.0,
                    "ask_depth": 80.0,
                    "end_date": pd.Timestamp("2026-02-01", tz="UTC"),
                    "extra": float(step),
                    "label": market % 2,
                }
            )
    return pd.DataFrame(rows)


def test_history_derived_features_are_lagged_and_per_market():
    frame = build_snapshot_features(_panel(), momentum_windows=(1,), volatility_window=0)
    first_rows = frame.groupby("market_id")["feat_momentum_1"].apply(lambda s: s.iloc[:2].isna().all())
    # One observation is lost to the difference and one more to the lag, so a
    # market's opening rows can never carry its own future.
    assert first_rows.all()
    assert frame.index.is_monotonic_increasing and frame.index.is_unique


def test_passthrough_columns_are_promoted_and_validated():
    frame = build_snapshot_features(_panel(), passthrough_columns=["extra"])
    assert "feat_extra" in frame.columns
    with pytest.raises(ValueError):
        build_snapshot_features(_panel(), passthrough_columns=["absent"])


def test_design_matrix_drops_warm_up_rows_rather_than_imputing_them():
    X, y, meta = design_matrix(build_snapshot_features(_panel()))
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(meta)
    assert X.index.equals(y.index)


def test_brier_skill_is_zero_against_itself_and_positive_for_a_better_forecast():
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    price = pd.Series([0.5] * 8)
    assert brier_skill_score(y, price, price) == pytest.approx(0.0)
    better = pd.Series([0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
    assert brier_skill_score(y, better, price) > 0


def test_forecast_report_compares_against_the_market_not_a_coin_flip():
    rng = np.random.default_rng(0)
    price = pd.Series(rng.uniform(0.1, 0.9, 400))
    y = pd.Series((rng.uniform(size=400) < price).astype(int))
    report = forecast_report(y, price, price)
    assert report["brier_skill_vs_market"] == pytest.approx(0.0)
    assert report["n_observations"] == 400


def test_edge_realization_reports_a_standard_error_for_a_flat_regressor():
    rng = np.random.default_rng(1)
    predicted = pd.Series(0.05 + rng.normal(0, 1e-4, 100))
    realized = pd.Series(rng.normal(0, 0.5, 100))
    stats = edge_realization(predicted, realized)
    assert abs(stats["slope_t_stat"]) < 3.0


def test_anchored_model_reproduces_the_market_when_the_penalty_is_infinite():
    rng = np.random.default_rng(2)
    price = rng.uniform(0.1, 0.9, 300)
    X = pd.DataFrame(
        {"feat_logit_price": np.log(price / (1 - price)), "feat_noise": rng.normal(size=300)}
    )
    y = pd.Series((rng.uniform(size=300) < price).astype(int))
    fitted = market_anchored_model(alpha=1e9).fit(X, y)
    assert np.allclose(fitted.predict_proba(X)[:, 1], price, atol=1e-6)


def test_anchored_model_is_cloneable_and_selects_its_own_penalty():
    rng = np.random.default_rng(3)
    price = rng.uniform(0.2, 0.8, 400)
    X = pd.DataFrame(
        {"feat_logit_price": np.log(price / (1 - price)), "feat_a": rng.normal(size=400)}
    )
    y = pd.Series((rng.uniform(size=400) < price).astype(int))
    fitted = market_anchored_model().fit(X, y)
    assert fitted.alpha_ in market_anchored_model().alpha_grid
    assert isinstance(clone(fitted), type(fitted))


def test_anchored_model_requires_its_anchor_column():
    X = pd.DataFrame({"feat_a": [0.0, 1.0, 0.0, 1.0]})
    with pytest.raises(ValueError):
        market_anchored_model().fit(X, pd.Series([0, 1, 0, 1]))


def test_blending_fully_toward_the_market_reproduces_the_price():
    price = pd.Series([0.2, 0.5, 0.8])
    blended = blend_with_market(pd.Series([0.9, 0.9, 0.9]), price, 1.0)
    assert np.allclose(blended.to_numpy(), price.to_numpy())


def _walk_forward_panel():
    panel = simulate_panel(PanelSpec(n_markets=120, observations_per_market=8, seed=5))
    frame = build_snapshot_features(panel, momentum_windows=(1,), volatility_window=0)
    X, y, _ = design_matrix(frame)
    return frame, X, y


def test_no_market_appears_on_both_sides_of_any_fold():
    frame, X, y = _walk_forward_panel()
    ids = frame.loc[X.index, "market_id"].to_numpy()
    folds = list(
        resolution_aware_folds(
            frame.loc[X.index, "timestamp"],
            frame.loc[X.index, "end_date"],
            frame.loc[X.index, "market_id"],
            100,
            100,
        )
    )
    assert folds, "expected at least one usable fold"
    for train, test in folds:
        assert not set(ids[train]) & set(ids[test])


def test_training_never_uses_a_label_before_it_was_knowable():
    frame, X, y = _walk_forward_panel()
    timestamps = frame.loc[X.index, "timestamp"]
    resolutions = frame.loc[X.index, "end_date"]
    groups = frame.loc[X.index, "market_id"]
    for train, test in resolution_aware_folds(timestamps, resolutions, groups, 100, 100):
        decision_time = timestamps.iloc[test[0]]
        assert resolutions.iloc[train].max() <= decision_time


def test_a_row_observed_after_its_market_resolved_cannot_leak_into_training():
    # A venue's stated end date routinely precedes its last recorded trade, so a
    # row can be eligible by resolution time while sitting inside the block being
    # scored. Resolution time alone does not exclude it; position and group must.
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"))
    resolutions = pd.Series([pd.Timestamp("2024-01-01", tz="UTC")] * 30)
    resolutions.iloc[20:] = pd.Timestamp("2024-03-01", tz="UTC")
    groups = pd.Series([f"m{index // 3}" for index in range(30)])

    folds = list(resolution_aware_folds(timestamps, resolutions, groups, 10, 1))
    assert folds
    for train, test in folds:
        assert not set(train) & set(test)
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        assert train.max() < test.min()


def test_the_guard_holds_on_a_panel_that_trades_past_its_stated_close(): 
    # The synthetic panel used by the study has every observation before its
    # market's end date, which is why it never exposed the leak. Generate the
    # realistic shape explicitly and check the guard against it.
    panel = simulate_panel(
        PanelSpec(n_markets=200, observations_per_market=10,
                  late_observation_fraction=0.4, seed=9)
    )
    assert (panel["timestamp"] > panel["end_date"]).mean() > 0.1
    frame = build_snapshot_features(panel, momentum_windows=(1,), volatility_window=0)
    X, y, _ = design_matrix(frame)
    ids = frame.loc[X.index, "market_id"]
    folds = list(
        resolution_aware_folds(
            frame.loc[X.index, "timestamp"], frame.loc[X.index, "end_date"], ids, 200, 100
        )
    )
    assert folds
    for train, test in folds:
        assert not set(ids.iloc[train]) & set(ids.iloc[test])


def test_embargo_shrinks_the_training_set():
    frame, X, y = _walk_forward_panel()
    timestamps, resolutions = frame.loc[X.index, "timestamp"], frame.loc[X.index, "end_date"]
    groups = frame.loc[X.index, "market_id"]
    without = [len(t) for t, _ in resolution_aware_folds(timestamps, resolutions, groups, 100, 50)]
    with_embargo = [
        len(t)
        for t, _ in resolution_aware_folds(
            timestamps, resolutions, groups, 100, 50, embargo_days=30
        )
    ]
    assert sum(with_embargo) < sum(without)


def test_walk_forward_returns_chronologically_ordered_out_of_sample_forecasts():
    frame, X, y = _walk_forward_panel()
    probability = resolution_aware_walk_forward(
        market_anchored_model(alpha=1.0),
        X,
        y,
        frame.loc[X.index, "timestamp"],
        frame.loc[X.index, "end_date"],
        frame.loc[X.index, "market_id"],
        test_window=100,
        min_train_size=100,
    )
    assert not probability.empty
    assert probability.index.is_monotonic_increasing
    assert probability.between(0, 1).all()
    assert probability.index.isin(X.index).all()


def test_unsorted_timestamps_are_rejected_rather_than_silently_mis_split():
    frame, X, y = _walk_forward_panel()
    reversed_timestamps = frame.loc[X.index, "timestamp"].iloc[::-1].reset_index(drop=True)
    reversed_timestamps.index = X.index
    with pytest.raises(ValueError):
        list(
            resolution_aware_folds(
                reversed_timestamps,
                frame.loc[X.index, "end_date"],
                frame.loc[X.index, "market_id"],
                100,
                50,
            )
        )
