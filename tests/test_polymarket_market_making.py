import numpy as np
import pandas as pd
import pytest

from quant_system.polymarket.market_making import (
    QuoteConfig,
    break_even_uninformed_rate,
    make_quotes,
    simulate_market_making,
    sweep_uninformed_fill_rate,
)
from quant_system.polymarket.simulation import PanelSpec, simulate_panel


def _panel(n_markets=200, seed=11):
    return simulate_panel(
        PanelSpec(n_markets=n_markets, observations_per_market=12, seed=seed)
    )


def test_quotes_straddle_fair_value_when_flat():
    config = QuoteConfig(half_spread=0.02)
    quote = make_quotes(0.50, 0.0, config)
    assert quote.bid == pytest.approx(0.48)
    assert quote.ask == pytest.approx(0.52)
    assert quote.mid == pytest.approx(0.50)


def test_quotes_lean_away_from_existing_inventory():
    config = QuoteConfig(half_spread=0.02, max_inventory=400, inventory_skew=1.0)
    long_book = make_quotes(0.50, 200.0, config)
    short_book = make_quotes(0.50, -200.0, config)
    # Holding a long position should make the maker keener to sell.
    assert long_book.mid < 0.50 < short_book.mid


def test_quote_size_goes_to_zero_on_the_side_that_breaches_the_cap():
    config = QuoteConfig(max_inventory=400, quote_size=100)
    assert make_quotes(0.5, 400.0, config).bid_size == 0.0
    assert make_quotes(0.5, 400.0, config).ask_size > 0.0
    assert make_quotes(0.5, -400.0, config).ask_size == 0.0
    assert make_quotes(0.5, -400.0, config).bid_size > 0.0


def test_quotes_widen_into_resolution():
    config = QuoteConfig(half_spread=0.02, widen_within_days=2.0, widen_multiple=2.0)
    far = make_quotes(0.50, 0.0, config, days_to_resolution=30.0)
    near = make_quotes(0.50, 0.0, config, days_to_resolution=0.5)
    assert near.spread == pytest.approx(2 * far.spread)


def test_quotes_stay_inside_the_unit_interval():
    config = QuoteConfig(half_spread=0.10)
    for fair in (0.01, 0.5, 0.99):
        quote = make_quotes(fair, 0.0, config)
        assert 0.0 < quote.bid < quote.ask < 1.0


def test_pure_informed_flow_loses_money():
    # With no benign flow every fill is adverse by construction, so a maker that
    # showed a profit here would mean the fill model is not modelling adverse
    # selection at all.
    result = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.0, seed=3), bankroll=100_000
    )
    assert result.summary["total_pnl"] < 0
    assert result.summary["informed_fill_share"] == pytest.approx(1.0)


def test_benign_flow_is_what_makes_a_maker_profitable():
    without = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.0, seed=3), bankroll=100_000
    )
    with_flow = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.9, seed=3), bankroll=100_000
    )
    assert with_flow.summary["total_pnl"] > without.summary["total_pnl"]
    assert with_flow.summary["spread_capture_ratio"] > without.summary["spread_capture_ratio"]


def test_informed_fills_are_marked_against_the_maker_immediately():
    result = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.3, seed=3), bankroll=100_000
    )
    fills = result.fills
    assert fills.loc[fills["informed"], "immediate_pnl"].mean() < 0
    assert fills.loc[~fills["informed"], "immediate_pnl"].mean() > 0


def test_inventory_never_breaches_its_cap_in_any_flow_regime():
    config = QuoteConfig(max_inventory=300, quote_size=100, seed=3)
    for rate in (0.0, 0.5, 1.0):
        result = simulate_market_making(
            _panel(),
            QuoteConfig(**{**config.__dict__, "uninformed_fill_rate": rate}),
            bankroll=100_000,
        )
        assert result.fills["inventory_after"].abs().max() <= 300 + 1e-9


def test_concurrency_cap_bounds_the_capital_committed():
    narrow = simulate_market_making(
        _panel(), QuoteConfig(max_concurrent_markets=5, seed=3), bankroll=100_000
    )
    wide = simulate_market_making(
        _panel(), QuoteConfig(max_concurrent_markets=200, seed=3), bankroll=100_000
    )
    assert narrow.summary["peak_committed_capital"] < wide.summary["peak_committed_capital"]
    assert narrow.summary["n_markets"] < wide.summary["n_markets"]


def test_a_fill_crossing_flat_is_collateralized_on_the_part_that_opens():
    from quant_system.polymarket.market_making import _affordable_size

    # Selling 400 against +50 closes 50 and opens a 350-share naked short, which
    # needs settlement collateral. Charging the whole fill at the pre-trade
    # sign's rate financed it out of nothing.
    assert _affordable_size("sell", 400, 0.50, 50.0, 1.0) < 60
    assert _affordable_size("buy", 400, 0.50, -50.0, 1.0) < 60
    # Closing alone releases collateral, so it is free even at zero cash.
    assert _affordable_size("sell", 50, 0.50, 50.0, 0.0) == pytest.approx(50.0)
    assert _affordable_size("buy", 50, 0.50, -50.0, 0.0) == pytest.approx(50.0)
    # With nothing to close and no cash, nothing is affordable.
    assert _affordable_size("sell", 400, 0.50, 0.0, 0.0) == 0.0
    assert _affordable_size("sell", 400, 0.50, 50.0, 1e9) == pytest.approx(400.0)


def test_widened_quotes_are_measured_at_the_width_actually_quoted():
    # Both maker headline diagnostics are ratios against quoted spread, so
    # recording the configured half-spread inside the widening window inflated
    # the capture ratio by the widening multiple.
    config = QuoteConfig(
        half_spread=0.02, widen_within_days=1e9, widen_multiple=3.0, seed=3
    )
    result = simulate_market_making(_panel(), config, bankroll=100_000)
    per_share = result.summary["quoted_spread_value"] / result.fills["size"].sum()
    assert per_share > 0.05  # 0.06 before clipping at the price bounds, not 0.02


def test_an_undercapitalized_book_declines_fills_rather_than_borrowing():
    result = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.5, seed=3), bankroll=50.0
    )
    assert result.summary["declined_fills"] > 0
    assert result.summary["capital_constrained"] == 1.0


def test_settlement_pays_the_final_inventory_at_zero_or_one():
    result = simulate_market_making(
        _panel(n_markets=40), QuoteConfig(uninformed_fill_rate=0.5, seed=3), bankroll=100_000
    )
    markets = result.markets
    expected = markets["final_inventory"] * markets["label"]
    assert np.allclose(markets["settlement_value"], expected)


def test_pnl_decomposes_into_quoted_spread_and_adverse_selection():
    result = simulate_market_making(
        _panel(), QuoteConfig(uninformed_fill_rate=0.4, seed=3), bankroll=100_000
    )
    markets = result.markets
    recombined = markets["quoted_spread_value"] + markets["adverse_selection_pnl"]
    assert np.allclose(recombined, markets["pnl"])


def test_a_forecast_edge_reduces_the_adverse_selection_a_maker_pays():
    panel = _panel()
    # Quote around the truth rather than the price: the maker now leans away
    # from the moves that would otherwise run it over.
    informed_fair = panel["true_probability"].astype(float)
    blind = simulate_market_making(
        panel, QuoteConfig(uninformed_fill_rate=0.0, seed=3), bankroll=100_000
    )
    sighted = simulate_market_making(
        panel, QuoteConfig(uninformed_fill_rate=0.0, seed=3), informed_fair, bankroll=100_000
    )
    assert sighted.summary["adverse_selection_pnl"] > blind.summary["adverse_selection_pnl"]


def test_sweep_is_monotone_and_locates_the_break_even_point():
    sweep = sweep_uninformed_fill_rate(
        _panel(), np.arange(0.0, 1.01, 0.2), QuoteConfig(seed=3), bankroll=100_000
    )
    assert sweep["total_pnl"].is_monotonic_increasing
    break_even = break_even_uninformed_rate(sweep)
    assert 0.0 < break_even < 1.0


def test_break_even_is_undefined_when_the_sweep_never_changes_sign():
    never = pd.DataFrame({"uninformed_fill_rate": [0.0, 0.5], "total_pnl": [-5.0, -1.0]})
    assert not np.isfinite(break_even_uninformed_rate(never))
    assert not np.isfinite(break_even_uninformed_rate(pd.DataFrame()))


def test_a_panel_without_the_required_columns_is_rejected():
    with pytest.raises(ValueError):
        simulate_market_making(pd.DataFrame({"market_id": ["a"], "price": [0.5]}))
