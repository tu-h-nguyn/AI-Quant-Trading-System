import json

import pytest

from quant_system.polymarket.arbitrage import (
    opportunities_frame,
    scan_binary_complement,
    scan_binary_mint_and_sell,
    scan_group_dutch_book,
    scan_markets,
    scan_neg_risk_no_basket,
)
from quant_system.polymarket.markets import MarketGroup, parse_market
from quant_system.polymarket.orderbook import OrderBook
from quant_system.polymarket.simulation import simulate_arbitrage_snapshot


def _market(index=1, yes="y", no="n", event="e1", neg_risk=True):
    return parse_market(
        {
            "id": str(index),
            "question": f"Q{index}",
            "slug": f"q{index}",
            "conditionId": f"0x{index}",
            "outcomes": json.dumps(["Yes", "No"]),
            "clobTokenIds": json.dumps([yes, no]),
            "outcomePrices": json.dumps(["0.5", "0.5"]),
            "closed": False,
            "events": [{"id": event, "slug": "event", "negRisk": neg_risk}],
        }
    )


def test_complement_basket_profit_matches_the_hand_calculation():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [(0.40, 100)], [(0.45, 200), (0.55, 500)]),
        "n": OrderBook.from_levels("n", [(0.44, 100)], [(0.50, 300), (0.60, 500)]),
    }
    found = scan_binary_complement(market, books, max_capital=10_000, min_profit=0.0)
    # 200 shares clear at 0.45 + 0.50 = 0.95, so each locked pair earns five cents.
    assert found.shares == pytest.approx(200)
    assert found.profit == pytest.approx(10.0)
    assert found.capital_required == pytest.approx(190.0)
    assert found.roi == pytest.approx(10.0 / 190.0)


def test_sizing_stops_where_the_next_level_destroys_the_edge():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [], [(0.45, 200), (0.90, 10_000)]),
        "n": OrderBook.from_levels("n", [], [(0.50, 200), (0.90, 10_000)]),
    }
    found = scan_binary_complement(market, books, max_capital=1_000_000, min_profit=0.0)
    assert found.shares == pytest.approx(200)


def test_capital_cap_shrinks_the_basket_without_breaching_the_budget():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [], [(0.45, 500)]),
        "n": OrderBook.from_levels("n", [], [(0.50, 500)]),
    }
    found = scan_binary_complement(market, books, max_capital=95.0, min_profit=0.0)
    assert found.capital_required <= 95.0 + 1e-9
    assert found.shares == pytest.approx(100, rel=1e-3)


def test_fees_can_erase_an_apparent_basket_edge():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [], [(0.49, 500)]),
        "n": OrderBook.from_levels("n", [], [(0.50, 500)]),
    }
    assert scan_binary_complement(market, books, fee_bps=0, max_capital=1e6, min_profit=0.0)
    assert scan_binary_complement(market, books, fee_bps=300, max_capital=1e6, min_profit=0.0) is None


def test_the_capital_budget_binds_on_baskets_that_sell():
    # Net cash flow is negative for a mint-and-sell basket, so a budget tested
    # against that quantity passes at full book depth: a $1,000 limit returned a
    # $100,000 basket. The budget applies to what must be funded up front.
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [(0.60, 100_000)], [(0.62, 10)]),
        "n": OrderBook.from_levels("n", [(0.45, 100_000)], [(0.47, 10)]),
    }
    found = scan_binary_mint_and_sell(market, books, max_capital=1_000, min_profit=0.0)
    assert found.capital_required <= 1_000 + 1e-6
    assert found.shares == pytest.approx(1_000)
    assert found.profit == pytest.approx(50.0)


def test_no_basket_is_reported_when_the_pair_costs_more_than_a_dollar():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [(0.50, 100)], [(0.52, 200)]),
        "n": OrderBook.from_levels("n", [(0.47, 100)], [(0.53, 200)]),
    }
    assert scan_binary_complement(market, books, max_capital=1e6, min_profit=0.0) is None


def test_mint_and_sell_is_capitalized_at_the_mint_not_the_net_cash():
    market = _market()
    books = {
        "y": OrderBook.from_levels("y", [(0.60, 150), (0.50, 400)], [(0.62, 100)]),
        "n": OrderBook.from_levels("n", [(0.45, 150), (0.40, 400)], [(0.47, 100)]),
    }
    found = scan_binary_mint_and_sell(market, books, max_capital=10_000, min_profit=0.0)
    assert found.shares == pytest.approx(150)
    assert found.profit == pytest.approx(7.5)
    # Selling legs return cash, but the complete set is funded in full first.
    assert found.capital_required == pytest.approx(150.0)
    assert found.requires_complete_set_mint


def test_group_baskets_price_against_their_guaranteed_settlement_value():
    markets = [_market(i, f"y{i}", f"n{i}") for i in (1, 2, 3)]
    group = MarketGroup("e1", "event", tuple(markets), neg_risk=True)
    books = {}
    for i in (1, 2, 3):
        books[f"y{i}"] = OrderBook.from_levels(f"y{i}", [(0.28, 100)], [(0.30, 100)])
        books[f"n{i}"] = OrderBook.from_levels(f"n{i}", [(0.68, 100)], [(0.60, 100)])

    dutch = scan_group_dutch_book(group, books, max_capital=10_000, min_profit=0.0)
    assert dutch.guaranteed_payout == pytest.approx(100.0)
    assert dutch.profit == pytest.approx(10.0)

    basket = scan_neg_risk_no_basket(group, books, max_capital=10_000, min_profit=0.0)
    # Exactly one of three outcomes resolves YES, so two NO shares pay out.
    assert basket.guaranteed_payout == pytest.approx(200.0)
    assert basket.profit == pytest.approx(20.0)


def test_unverified_groups_are_refused_because_the_payout_is_not_guaranteed():
    markets = [_market(i, f"y{i}", f"n{i}", neg_risk=False) for i in (1, 2, 3)]
    group = MarketGroup("e2", "event", tuple(markets), neg_risk=False)
    books = {}
    for i in (1, 2, 3):
        books[f"y{i}"] = OrderBook.from_levels(f"y{i}", [(0.28, 100)], [(0.30, 100)])
        books[f"n{i}"] = OrderBook.from_levels(f"n{i}", [(0.68, 100)], [(0.60, 100)])
    assert scan_group_dutch_book(group, books, max_capital=1e6, min_profit=0.0) is None
    permitted = scan_group_dutch_book(
        group, books, max_capital=1e6, min_profit=0.0, require_verified=False
    )
    assert permitted is not None and not permitted.verified_exhaustive
    assert permitted.notes


@pytest.mark.parametrize("seed", [7, 11, 99, 2024])
def test_scanner_matches_the_planted_ground_truth_exactly(seed):
    fixture = simulate_arbitrage_snapshot(seed=seed)
    found = scan_markets(
        fixture.markets, fixture.books, fixture.groups, max_capital=5_000, min_profit=0.5
    )
    detected: dict[str, set[str]] = {}
    for opportunity in found:
        key = (
            opportunity.event_slug
            if opportunity.kind.startswith(("group", "neg"))
            else opportunity.legs[0].market_id
        )
        detected.setdefault(opportunity.kind, set()).add(key)
    for kind, planted in fixture.planted.items():
        # Both directions matter: a missed basket is lost money and a phantom
        # one is money lost on a trade that was never an arbitrage.
        assert len(detected.get(kind, set())) == len(planted), kind


def test_empty_scan_produces_a_usable_frame():
    frame = opportunities_frame([])
    assert frame.empty
    assert "profit" in frame.columns
