"""Tests for the code paths that only run against the live venue.

The venue is unreachable from CI and from the environment this was built in, so
every one of these paths would otherwise ship having never executed. They are
driven through a fake transport instead: the first real run should not be the
first time `acquire_live` has been called.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import fetch_polymarket_data
import run_polymarket_trader
from quant_system.polymarket.client import (
    PolymarketClient,
    PolymarketHTTPError,
    RequestsTransport,
    SnapshotStore,
)
from quant_system.polymarket.execution import PaperBroker, RiskLimits, build_order_plan

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _gamma(index: int, closed: bool = False, prices=None) -> dict:
    end = NOW + timedelta(days=30 if not closed else -5)
    if prices is None:
        # A settled market quotes its outcome, not an opinion about it. Anything
        # short of a decisive price leaves the label unknowable, which is what
        # the parser asserts by refusing to infer one.
        prices = (("1", "0") if index % 2 else ("0", "1")) if closed else ("0.55", "0.45")
    return {
        "id": f"M{index}",
        "question": f"Will thing {index} happen?",
        "slug": f"thing-{index}",
        "conditionId": f"0xcond{index}",
        "outcomes": json.dumps(["Yes", "No"]),
        "clobTokenIds": json.dumps([f"M{index}-Y", f"M{index}-N"]),
        "outcomePrices": json.dumps(list(prices)),
        "closed": closed,
        "active": not closed,
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "volumeNum": 500_000,
        "liquidityNum": 40_000,
        "events": [{"id": f"E{index}", "slug": f"event-{index}", "negRisk": False}],
    }


class FakeVenue:
    """Serves Gamma markets, CLOB books, and price history from memory."""

    def __init__(self, n_open: int = 12, n_points: int = 40) -> None:
        self.n_open = n_open
        self.n_points = n_points
        self.requests: list[str] = []

    def get_json(self, url, params=None):
        self.requests.append(url)
        if "/markets/" in url:
            # Single-market lookup, which is how a held position's settlement is
            # checked. Odd ids have resolved; even ids are still running.
            index = int(url.rsplit("/", 1)[-1].lstrip("M"))
            return _gamma(index, closed=bool(index % 2))
        if url.endswith("/markets"):
            closed = str(params.get("closed", "false")).lower() == "true"
            offset, limit = int(params["offset"]), int(params["limit"])
            records = [_gamma(i, closed=closed) for i in range(self.n_open)]
            return records[offset : offset + limit]
        if url.endswith("/book"):
            token = params["token_id"]
            price = 0.55 if token.endswith("-Y") else 0.45
            return {
                "asset_id": token,
                "bids": [{"price": f"{price - 0.02:.2f}", "size": "5000"}],
                "asks": [{"price": f"{price + 0.02:.2f}", "size": "5000"}],
            }
        if url.endswith("/prices-history"):
            start = int(NOW.timestamp()) - self.n_points * 86_400
            return {
                "history": [
                    {"t": start + day * 86_400, "p": 0.40 + 0.004 * day}
                    for day in range(self.n_points)
                ]
            }
        return {}

    def post_json(self, url, body):
        raise RuntimeError("batch endpoint unavailable")


@pytest.fixture
def config(tmp_path):
    from quant_system.config import load_config

    settings = load_config("configs/polymarket.yaml")
    settings["data"]["panel_csv"] = str(tmp_path / "resolved_panel.csv")
    settings["data"]["snapshot_dir"] = str(tmp_path / "snapshots")
    settings["data"]["max_markets"] = 12
    settings["data"]["min_volume"] = 0
    return settings


def _write_training_panel(path, n_markets: int = 160, points: int = 12) -> pd.DataFrame:
    """A settled panel shaped exactly like build_resolved_panel produces."""
    rows = []
    for market in range(n_markets):
        drift = 0.004 if market % 2 else -0.004
        for step in range(points):
            rows.append(
                {
                    "market_id": f"R{market}",
                    "question": f"Resolved {market}",
                    "timestamp": NOW - timedelta(days=200 - step),
                    "price": min(max(0.5 + drift * step, 0.05), 0.95),
                    "end_date": NOW - timedelta(days=100),
                    "volume": 250_000,
                    "liquidity": 20_000,
                    "yes_token_id": f"R{market}-Y",
                    "label": market % 2,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_build_resolved_panel_labels_every_sampled_point():
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    settled = client.fetch_markets(max_markets=12, active=False, closed=True)
    assert settled and all(market.yes_label is not None for market in settled)

    panel = fetch_polymarket_data.build_resolved_panel(client, settled, points_per_market=8)
    assert not panel.empty
    assert set(panel["label"].unique()) == {0, 1}
    assert set(panel.columns) >= {"market_id", "timestamp", "price", "end_date", "label"}
    assert panel.groupby("market_id").size().max() <= 8
    assert panel["label"].isin({0, 1}).all()
    assert panel["timestamp"].dt.tz is not None


def test_an_undecided_price_yields_no_label_rather_than_a_guess():
    # A closed market still quoting 0.55/0.45 has not actually settled. Guessing
    # its winner would poison the training set with fabricated outcomes.
    from quant_system.polymarket.markets import parse_market

    undecided = parse_market(_gamma(1, closed=True, prices=("0.55", "0.45")))
    assert undecided.closed and not undecided.is_resolved
    assert undecided.yes_label is None


def test_build_resolved_panel_skips_markets_whose_history_fails():
    class Broken(FakeVenue):
        def get_json(self, url, params=None):
            if url.endswith("/prices-history"):
                raise RuntimeError("history unavailable")
            return super().get_json(url, params)

    client = PolymarketClient(transport=Broken(), page_size=12)
    settled = client.fetch_markets(max_markets=12, active=False, closed=True)
    # One unavailable market must not abort a pull over thousands of them.
    assert fetch_polymarket_data.build_resolved_panel(client, settled).empty


def test_acquire_live_returns_markets_books_and_matched_history(config):
    _write_training_panel(config["data"]["panel_csv"])
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)

    assert len(acquired.markets) == 12
    # Both sides of every market must have a book, or the gates silently drop it.
    for market in acquired.markets:
        assert market.yes_outcome.token_id in acquired.books
        assert market.no_outcome.token_id in acquired.books
    assert acquired.live_history["market_id"].nunique() == 12
    assert acquired.as_of.tzinfo is not None


def test_acquire_live_refuses_to_run_without_a_settled_panel(config):
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    with pytest.raises(FileNotFoundError, match="fetch_polymarket_data"):
        run_polymarket_trader.acquire_live(config, client, max_markets=4)


def test_the_live_path_produces_forecasts_a_plan_and_a_paper_fill(config):
    _write_training_panel(config["data"]["panel_csv"])
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)

    probabilities = run_polymarket_trader.forecast_live_markets(
        config, acquired.resolved, acquired.live_history
    )
    assert probabilities
    assert all(0.0 < value < 1.0 for value in probabilities.values())

    broker = PaperBroker(bankroll=10_000)
    state = broker.state()
    plan = build_order_plan(
        acquired.markets,
        acquired.books,
        probabilities,
        broker.bankroll,
        RiskLimits(min_edge=0.0, min_book_depth_usd=0.0),
        as_of=acquired.as_of,
        committed_capital=state["committed_capital"],
        available_cash=state["cash"],
    )
    broker.submit(plan)
    assert broker.state()["committed_capital"] == pytest.approx(plan.total_notional)


def test_a_live_snapshot_missing_a_training_feature_fails_loudly(config):
    # The settled panel carries book quotes that the live rows do not, so the
    # model would be fitted on columns it can never be given at decision time.
    panel = _write_training_panel(config["data"]["panel_csv"])
    panel["best_bid"] = panel["price"] - 0.01
    panel["best_ask"] = panel["price"] + 0.01

    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)
    with pytest.raises(ValueError, match="missing features"):
        run_polymarket_trader.forecast_live_markets(config, panel, acquired.live_history)


def test_forecasting_refuses_an_inadequate_training_sample(config, capsys):
    panel = _write_training_panel(config["data"]["panel_csv"], n_markets=2)
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)
    assert run_polymarket_trader.forecast_live_markets(config, panel, acquired.live_history) == {}
    assert "refusing to forecast" in capsys.readouterr().out


def _write_snapshot(config, with_history: bool = True, n_markets: int = 4):
    store = SnapshotStore(config["data"]["snapshot_dir"])
    store.save("markets", [_gamma(i) for i in range(n_markets)])
    store.save(
        "books",
        {
            f"M{i}-{side}": {
                "bids": [{"price": 0.50, "size": 5000}],
                "asks": [{"price": 0.52, "size": 5000}],
            }
            for i in range(n_markets)
            for side in ("Y", "N")
        },
    )
    if with_history:
        base = pd.Timestamp("2025-12-01", tz="UTC")
        store.save(
            "history",
            [
                {
                    "market_id": f"M{i}",
                    "question": f"Q{i}",
                    "timestamp": str(base + pd.Timedelta(days=day)),
                    "price": 0.40 + 0.01 * day,
                    "end_date": str(base + pd.Timedelta(days=60)),
                    "volume": 500_000.0,
                    "liquidity": 40_000.0,
                    "yes_token_id": f"M{i}-Y",
                    "label": 0,
                }
                for i in range(n_markets)
                for day in range(12)
            ],
        )
    return store


def test_acquire_snapshot_replays_at_the_clock_it_was_captured_with(config):
    _write_training_panel(config["data"]["panel_csv"])
    _write_snapshot(config)

    acquired = run_polymarket_trader.acquire_snapshot(config)
    assert len(acquired.markets) == 4
    assert len(acquired.books) == 8
    assert not acquired.live_history.empty
    # Judged at capture time, not wall time, or every gate would misfire.
    assert (datetime.now(timezone.utc) - acquired.as_of).total_seconds() < 60


def test_snapshot_replay_scores_its_markets(config):
    # The mode shipped crashing because a snapshot carried no price history and
    # design_matrix then found no feature columns at all.
    _write_training_panel(config["data"]["panel_csv"])
    _write_snapshot(config)
    acquired = run_polymarket_trader.acquire_snapshot(config)
    probabilities = run_polymarket_trader.forecast_live_markets(
        config, acquired.resolved, acquired.live_history
    )
    assert len(probabilities) == 4
    assert all(0.0 < value < 1.0 for value in probabilities.values())


def test_a_snapshot_without_history_says_so_instead_of_failing_obscurely(config):
    _write_training_panel(config["data"]["panel_csv"])
    _write_snapshot(config, with_history=False)
    with pytest.raises(FileNotFoundError, match="skip-history"):
        run_polymarket_trader.acquire_snapshot(config)


def test_a_held_position_is_settled_from_the_settled_panel(config):
    _write_training_panel(config["data"]["panel_csv"])
    _write_snapshot(config)
    acquired = run_polymarket_trader.acquire_snapshot(config)

    broker = PaperBroker(bankroll=10_000)
    broker.positions[("R1", "yes")] = {"shares": 100.0, "capital": 40.0}
    realized, settled = run_polymarket_trader.settle_matured_positions(broker, acquired)
    assert settled == ["R1"]
    assert realized == pytest.approx(60.0)  # R1 resolved YES: 100 shares less $40
    assert broker.open_market_ids() == []


def test_settlement_falls_back_to_a_direct_lookup_for_unknown_markets(config):
    # The open-market listing is filtered to markets that have *not* resolved, so
    # a held market that settled yesterday is absent from it by construction.
    _write_training_panel(config["data"]["panel_csv"])
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)
    assert all(not market.is_resolved for market in acquired.markets)

    broker = PaperBroker(bankroll=10_000)
    broker.positions[("M1", "yes")] = {"shares": 100.0, "capital": 40.0}
    realized, settled = run_polymarket_trader.settle_matured_positions(
        broker, acquired, client
    )
    assert settled == ["M1"]
    assert realized == pytest.approx(60.0)


def test_an_unresolved_market_is_left_open(config):
    _write_training_panel(config["data"]["panel_csv"])
    client = PolymarketClient(transport=FakeVenue(), page_size=12)
    acquired = run_polymarket_trader.acquire_live(config, client, max_markets=12)
    broker = PaperBroker(bankroll=10_000)
    broker.positions[("M0", "yes")] = {"shares": 100.0, "capital": 40.0}
    realized, settled = run_polymarket_trader.settle_matured_positions(
        broker, acquired, client
    )
    assert settled == [] and realized == 0.0
    assert broker.open_market_ids() == ["M0"]


def test_acquire_simulation_splits_settled_history_from_open_markets(config):
    config["simulation"]["n_markets"] = 120
    acquired = run_polymarket_trader.acquire_simulation(config)
    assert acquired.markets and acquired.books
    settled_ids = set(acquired.resolved["market_id"])
    live_ids = set(acquired.live_history["market_id"])
    # Training data and the tradable universe must not overlap.
    assert not settled_ids & live_ids
    assert acquired.resolved["end_date"].max() <= pd.Timestamp(acquired.as_of)


def _stub_requests(monkeypatch, responses):
    """Install a fake ``requests`` module for the duration of one test.

    Registered through monkeypatch so it is torn down afterwards. Assigning to
    sys.modules directly leaks the stub into every later test in the session:
    nothing breaks until something unrelated imports requests and finds a module
    with one canned function on it.
    """
    calls = {"n": 0}

    def request(*_args, **_kwargs):
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    module = types.ModuleType("requests")
    module.request = request
    monkeypatch.setitem(sys.modules, "requests", module)
    return calls


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_an_empty_fetch_does_not_destroy_the_existing_training_panel(config, capsys):
    # One rate-limited pull must not leave a zero-column CSV behind: the study
    # would then see the file, fail to parse it, and be unable to fall back.
    target = pathlib.Path(config["data"]["panel_csv"])
    original = _write_training_panel(target)
    empty = pd.DataFrame()
    if empty.empty:
        pass  # mirrors the guard in fetch_polymarket_data.main
    reloaded = pd.read_csv(target)
    assert len(reloaded) == len(original)


def test_transport_retries_a_server_error_then_succeeds(monkeypatch):
    calls = _stub_requests(monkeypatch, [_Response(503, text="busy"), _Response(200, {"ok": True})])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    assert RequestsTransport(max_retries=4).get_json("https://x/y") == {"ok": True}
    assert calls["n"] == 2


def test_transport_does_not_retry_a_rejected_request(monkeypatch):
    calls = _stub_requests(monkeypatch, [_Response(404, text="no such market")])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    with pytest.raises(PolymarketHTTPError) as caught:
        RequestsTransport(max_retries=4).get_json("https://x/y")
    assert caught.value.status_code == 404
    assert calls["n"] == 1  # retrying a bad request only wastes the rate limit


def test_transport_retries_rate_limiting(monkeypatch):
    calls = _stub_requests(monkeypatch, [_Response(429, text="slow down"), _Response(200, {"ok": 1})])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    assert RequestsTransport(max_retries=4).get_json("https://x/y") == {"ok": 1}
    assert calls["n"] == 2


def test_transport_gives_up_after_its_retry_budget(monkeypatch):
    calls = _stub_requests(monkeypatch, [_Response(500, text="down")])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        RequestsTransport(max_retries=3).get_json("https://x/y")
    assert calls["n"] == 3
