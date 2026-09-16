import numpy as np
import pytest

from quant_system.polymarket.kelly import (
    allocate_exposure,
    drawdown_scaled_bankroll,
    expected_log_growth,
    kelly_fraction,
    size_position,
)
from quant_system.polymarket.orderbook import OrderBook
from quant_system.polymarket.pricing import (
    breakeven_probability,
    buy_cost_per_share,
    edge,
    roi_if_correct,
    settlement_value,
    shrink_probability,
    taker_fee_per_share,
)


def test_fee_is_symmetric_around_the_midpoint():
    assert taker_fee_per_share(0.3, 200) == pytest.approx(taker_fee_per_share(0.7, 200))
    assert taker_fee_per_share(0.5, 200) > taker_fee_per_share(0.05, 200)


def test_breakeven_sits_above_the_quote_once_fees_are_charged():
    assert breakeven_probability(0.40, 0) == pytest.approx(0.40)
    assert breakeven_probability(0.40, 200) > 0.40


def test_edge_and_roi_are_consistent_with_a_one_dollar_payoff():
    assert edge(0.55, 0.40) == pytest.approx(0.15)
    assert roi_if_correct(0.40) == pytest.approx(1.5)
    assert buy_cost_per_share(0.40, 100) == pytest.approx(0.404)


def test_settlement_pays_exactly_one_side():
    assert settlement_value("yes", 1) == 1.0
    assert settlement_value("no", 1) == 0.0
    assert settlement_value("yes", 0) + settlement_value("no", 0) == 1.0
    with pytest.raises(ValueError):
        settlement_value("maybe", 1)


def test_shrinkage_interpolates_between_forecast_and_price():
    assert shrink_probability(0.8, 0.5, 0.0) == pytest.approx(0.8)
    assert shrink_probability(0.8, 0.5, 1.0) == pytest.approx(0.5)
    assert shrink_probability(0.8, 0.5, 0.5) == pytest.approx(0.65)


def test_kelly_fraction_maximizes_expected_log_growth():
    probability, cost = 0.60, 0.50
    optimal = kelly_fraction(probability, cost)
    grid = np.linspace(0.001, 0.95, 5000)
    numeric = grid[np.argmax([expected_log_growth(f, probability, cost) for f in grid])]
    assert optimal == pytest.approx(numeric, abs=5e-3)


def test_kelly_refuses_a_bet_without_edge():
    assert kelly_fraction(0.40, 0.50) == 0.0
    assert kelly_fraction(0.50, 0.50) == 0.0


def test_sizing_reports_the_constraint_that_bound():
    blocked = size_position(10_000, 0.52, 0.50, min_edge=0.05)
    assert blocked.notional == 0.0 and blocked.binding_constraint == "min_edge"
    capped = size_position(10_000, 0.70, 0.50, kelly_multiplier=1.0, max_fraction=0.02)
    assert capped.applied_fraction == pytest.approx(0.02)
    assert capped.binding_constraint == "max_fraction"


def test_exposure_allocation_respects_the_aggregate_budget():
    scaled = allocate_exposure({"a": 0.1, "b": 0.2, "c": 0.3}, max_total_exposure=0.3)
    assert sum(scaled.values()) == pytest.approx(0.3)
    assert scaled["c"] > scaled["b"] > scaled["a"]
    untouched = allocate_exposure({"a": 0.05}, max_total_exposure=0.3)
    assert untouched == {"a": 0.05}


def test_bankroll_tapers_with_drawdown():
    assert drawdown_scaled_bankroll(10_000, 10_000) == 10_000
    assert drawdown_scaled_bankroll(7_500, 10_000, 0.25, 0.25) == pytest.approx(1_875.0)
    assert drawdown_scaled_bankroll(9_000, 10_000) < 9_000


def test_book_orders_levels_best_first_regardless_of_input_order():
    book = OrderBook.from_levels("t", [(0.30, 10), (0.40, 10)], [(0.60, 10), (0.50, 10)])
    assert book.best_bid == 0.40
    assert book.best_ask == 0.50
    assert book.mid == pytest.approx(0.45)


def test_walking_the_book_prices_depth_not_the_top_quote():
    book = OrderBook.from_levels("t", [], [(0.40, 100), (0.50, 100)])
    fill = book.walk("buy", 150)
    assert fill.shares == pytest.approx(150)
    assert fill.average_price == pytest.approx((100 * 0.40 + 50 * 0.50) / 150)
    assert fill.average_price > book.best_ask


def test_a_thin_book_returns_a_partial_fill_rather_than_pretending():
    book = OrderBook.from_levels("t", [], [(0.40, 50)])
    fill = book.walk("buy", 500)
    assert fill.shares == pytest.approx(50)
    assert not fill.complete


def test_limit_price_stops_the_walk_before_the_edge_is_destroyed():
    book = OrderBook.from_levels("t", [], [(0.40, 100), (0.55, 900)])
    fill = book.walk("buy", 1000, limit_price=0.45)
    assert fill.shares == pytest.approx(100)
    assert book.executable_shares("buy", 0.45) == pytest.approx(100)


def test_effective_price_includes_fees_on_the_correct_side():
    book = OrderBook.from_levels("t", [(0.40, 100)], [(0.50, 100)])
    buy = book.walk("buy", 10, fee_bps=200)
    sell = book.walk("sell", 10, fee_bps=200)
    assert buy.effective_price > buy.average_price
    assert sell.effective_price < sell.average_price
    assert buy.cost > 0 and sell.cost < 0


def test_imbalance_is_signed_and_bounded():
    assert OrderBook.from_levels("t", [(0.4, 100)], [(0.5, 0.0001)]).imbalance() > 0.9
    assert OrderBook.from_levels("t", [], []).imbalance() == 0.0
