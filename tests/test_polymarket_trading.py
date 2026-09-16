from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quant_system.polymarket.backtest import (
    TradeConfig,
    run_backtest,
    theoretical_break_even_edge,
)
from quant_system.polymarket.execution import (
    OrderPlan,
    PaperBroker,
    RiskLimits,
    arbitrage_to_orders,
    build_order_plan,
    deduplicate_by_event,
)
from quant_system.polymarket.arbitrage import scan_markets
from quant_system.polymarket.simulation import simulate_arbitrage_snapshot

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _panel(n_markets=6, label=1, price=0.40):
    rows = []
    for market in range(n_markets):
        rows.append(
            {
                "market_id": f"m{market}",
                "question": f"Market {market}",
                "timestamp": pd.Timestamp(NOW) + pd.Timedelta(days=market),
                "price": price,
                "yes_ask": price + 0.01,
                "no_ask": (1 - price) + 0.01,
                "end_date": pd.Timestamp(NOW) + pd.Timedelta(days=30 + market),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def _config(**overrides):
    base = {
        "bankroll": 10_000.0,
        "fee_bps": 0.0,
        "default_spread": 0.02,
        "extra_slippage": 0.0,
        "kelly_multiplier": 0.25,
        "max_fraction": 0.05,
        "max_total_exposure": 0.20,
        "min_edge": 0.03,
        "shrinkage": 0.0,
        "min_notional": 5.0,
    }
    base.update(overrides)
    return TradeConfig(**base)


def test_a_forecast_equal_to_the_price_places_no_trade():
    panel = _panel()
    probability = panel["price"].copy()
    result = run_backtest(panel, probability, _config())
    assert result.trades.empty
    assert result.summary["n_trades"] == 0


def test_a_correct_forecast_settles_at_a_dollar_per_share():
    panel = _panel(n_markets=1, label=1, price=0.40)
    result = run_backtest(panel, pd.Series([0.9], index=panel.index), _config())
    trade = result.trades.iloc[0]
    assert trade["side"] == "yes"
    assert trade["payoff"] == 1.0
    assert trade["profit"] == pytest.approx(trade["shares"] - trade["capital"])
    assert result.final_bankroll > 10_000


def test_a_wrong_forecast_loses_exactly_the_capital_committed():
    panel = _panel(n_markets=1, label=0, price=0.40)
    result = run_backtest(panel, pd.Series([0.9], index=panel.index), _config())
    trade = result.trades.iloc[0]
    assert trade["payoff"] == 0.0
    assert trade["profit"] == pytest.approx(-trade["capital"])


def test_the_cheaper_side_is_taken_when_the_forecast_points_down():
    panel = _panel(n_markets=1, label=0, price=0.60)
    result = run_backtest(panel, pd.Series([0.1], index=panel.index), _config())
    assert result.trades.iloc[0]["side"] == "no"
    assert result.trades.iloc[0]["profit"] > 0


def test_entries_are_charged_the_spread_rather_than_filled_at_the_mid():
    panel = _panel(n_markets=1, price=0.40).drop(columns=["yes_ask", "no_ask"])
    result = run_backtest(panel, pd.Series([0.9], index=panel.index), _config(default_spread=0.04))
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(0.42)


def test_aggregate_exposure_is_capped_across_simultaneous_positions():
    panel = _panel(n_markets=12, label=1)
    probability = pd.Series(0.9, index=panel.index)
    result = run_backtest(panel, probability, _config(max_total_exposure=0.10, max_fraction=0.05))
    # Nothing settles before the last entry, so total capital is the peak.
    assert result.trades["capital"].sum() <= 0.10 * 10_000 + 1e-6


def test_settled_capital_is_released_for_later_trades():
    # Two markets, both wanting the entire exposure budget. The first settles
    # before the second is seen, so the budget must free up; if committed
    # capital were never released, only one trade could ever be placed.
    panel = pd.DataFrame(
        [
            {
                "market_id": "early", "question": "early",
                "timestamp": pd.Timestamp(NOW), "price": 0.40,
                "yes_ask": 0.41, "no_ask": 0.61,
                "end_date": pd.Timestamp(NOW) + pd.Timedelta(days=5), "label": 1,
            },
            {
                "market_id": "late", "question": "late",
                "timestamp": pd.Timestamp(NOW) + pd.Timedelta(days=10), "price": 0.40,
                "yes_ask": 0.41, "no_ask": 0.61,
                "end_date": pd.Timestamp(NOW) + pd.Timedelta(days=30), "label": 1,
            },
        ]
    )
    settings = _config(max_total_exposure=0.05, max_fraction=0.05)
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), settings)
    assert set(result.trades["market_id"]) == {"early", "late"}
    # Each trade may use the whole budget because they never overlap. The budget
    # is a share of the bankroll *at the time of entry*, which the first trade's
    # profit has already grown by the time the second is sized.
    bankroll_before = result.trades["bankroll_after"] - result.trades["profit"]
    assert (result.trades["capital"] <= 0.05 * bankroll_before + 1e-6).all()
    assert result.trades["capital"].iloc[1] > result.trades["capital"].iloc[0]


def test_capital_is_locked_while_positions_overlap():
    # Same budget, but now the first market settles long after the second is
    # seen, so the two compete for one allocation.
    panel = _panel(n_markets=2, label=1)
    panel["end_date"] = pd.Timestamp(NOW) + pd.Timedelta(days=90)
    settings = _config(max_total_exposure=0.05, max_fraction=0.05)
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), settings)
    assert result.trades["capital"].sum() <= 0.05 * 10_000 + 1e-6


def test_one_trade_per_market_prevents_pyramiding_into_the_same_outcome():
    panel = pd.concat([_panel(n_markets=1), _panel(n_markets=1)], ignore_index=True)
    panel.loc[1, "timestamp"] = panel.loc[0, "timestamp"] + pd.Timedelta(days=1)
    result = run_backtest(
        panel, pd.Series(0.9, index=panel.index), _config(one_trade_per_market=True)
    )
    assert len(result.trades) == 1


def test_shrinkage_can_pull_a_forecast_below_the_edge_gate():
    panel = _panel(n_markets=1, price=0.40)
    probability = pd.Series([0.50], index=panel.index)
    assert not run_backtest(panel, probability, _config(shrinkage=0.0)).trades.empty
    assert run_backtest(panel, probability, _config(shrinkage=0.9)).trades.empty


def test_break_even_edge_reflects_spread_and_fees():
    assert theoretical_break_even_edge(0.5, 0.02, 0) == pytest.approx(0.01)
    assert theoretical_break_even_edge(0.5, 0.02, 200) > 0.01


def test_summary_reports_bankroll_and_forecast_diagnostics():
    panel = _panel(n_markets=30, label=1)
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), _config())
    assert result.summary["n_trades"] > 0
    assert np.isfinite(result.summary["roi_on_capital"])
    assert "edge_slope" in result.summary
    assert len(result.equity) == len(result.trades) + 1


def _two_step_panel(prices, label=1, days=(0, 5), horizon=60):
    """One market observed twice, so an exit rule has a later price to act on."""
    return pd.DataFrame(
        [
            {
                "market_id": "m",
                "question": "q",
                "timestamp": pd.Timestamp(NOW) + pd.Timedelta(days=day),
                "price": price,
                "yes_ask": price + 0.01,
                "no_ask": (1 - price) + 0.01,
                "end_date": pd.Timestamp(NOW) + pd.Timedelta(days=horizon),
                "label": label,
            }
            for day, price in zip(days, prices)
        ]
    )


def test_exit_rules_are_off_by_default():
    panel = _two_step_panel([0.40, 0.88])
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), _config())
    assert set(result.trades["exit_reason"]) == {"settlement"}
    assert result.trades.iloc[0]["holding_days"] == pytest.approx(60.0)


def test_a_converged_position_is_closed_before_settlement():
    # Bought at 0.40 against a 0.90 forecast; once the price reaches 0.88 there
    # is almost no edge left to wait for.
    panel = _two_step_panel([0.40, 0.88])
    settings = _config(exit_edge_fraction=0.25)
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), settings)
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "convergence"
    assert trade["holding_days"] == pytest.approx(5.0)
    assert trade["profit"] > 0


def test_a_position_still_carrying_edge_is_held():
    panel = _two_step_panel([0.40, 0.45])
    settings = _config(exit_edge_fraction=0.25)
    result = run_backtest(panel, pd.Series(0.9, index=panel.index), settings)
    assert result.trades.iloc[0]["exit_reason"] == "settlement"


def test_exiting_pays_the_spread_a_second_time():
    panel = _two_step_panel([0.40, 0.88])
    wide = run_backtest(
        panel, pd.Series(0.9, index=panel.index),
        _config(exit_edge_fraction=0.25, default_spread=0.10),
    )
    narrow = run_backtest(
        panel, pd.Series(0.9, index=panel.index),
        _config(exit_edge_fraction=0.25, default_spread=0.02),
    )
    assert wide.trades.iloc[0]["payoff"] < narrow.trades.iloc[0]["payoff"]


def test_a_stop_cuts_a_losing_position_before_settlement():
    panel = _two_step_panel([0.40, 0.15], label=0)
    held = run_backtest(panel, pd.Series(0.9, index=panel.index), _config())
    stopped = run_backtest(
        panel, pd.Series(0.9, index=panel.index), _config(stop_loss_move=0.15)
    )
    assert held.trades.iloc[0]["exit_reason"] == "settlement"
    assert stopped.trades.iloc[0]["exit_reason"] == "stop_loss"
    # Settlement at zero loses the whole stake; the stop recovers part of it.
    assert stopped.trades.iloc[0]["profit"] > held.trades.iloc[0]["profit"]


def test_a_stop_can_cut_a_position_that_would_have_won():
    panel = _two_step_panel([0.40, 0.15], label=1)
    stopped = run_backtest(
        panel, pd.Series(0.9, index=panel.index), _config(stop_loss_move=0.15)
    )
    held = run_backtest(panel, pd.Series(0.9, index=panel.index), _config())
    assert stopped.trades.iloc[0]["profit"] < held.trades.iloc[0]["profit"]


def test_an_exit_releases_capital_for_another_market():
    early = _two_step_panel([0.40, 0.88]).assign(market_id="early")
    late = pd.DataFrame(
        [
            {
                "market_id": "late", "question": "q",
                "timestamp": pd.Timestamp(NOW) + pd.Timedelta(days=10), "price": 0.40,
                "yes_ask": 0.41, "no_ask": 0.61,
                "end_date": pd.Timestamp(NOW) + pd.Timedelta(days=60), "label": 1,
            }
        ]
    )
    panel = pd.concat([early, late], ignore_index=True)
    probability = pd.Series(0.9, index=panel.index)
    settings = _config(max_total_exposure=0.05, max_fraction=0.05)
    held = run_backtest(panel, probability, settings)
    exited = run_backtest(panel, probability, replace(settings, exit_edge_fraction=0.25))
    # Holding "early" to day 60 blocks the whole budget; exiting on day 5 frees it.
    assert set(held.trades["market_id"]) == {"early"}
    assert set(exited.trades["market_id"]) == {"early", "late"}


def test_capital_years_charge_for_the_time_money_is_committed():
    panel = _two_step_panel([0.40, 0.88])
    held = run_backtest(panel, pd.Series(0.9, index=panel.index), _config())
    exited = run_backtest(
        panel, pd.Series(0.9, index=panel.index), _config(exit_edge_fraction=0.25)
    )
    assert exited.summary["capital_years"] < held.summary["capital_years"]
    # A shorter hold gives up edge per trade but earns it far faster.
    assert exited.summary["roi_on_capital"] < held.summary["roi_on_capital"]
    assert exited.summary["roi_per_capital_year"] > held.summary["roi_per_capital_year"]


def test_exit_configuration_is_validated():
    with pytest.raises(ValueError):
        _config(exit_edge_fraction=1.5)
    with pytest.raises(ValueError):
        _config(stop_loss_move=0.0)


def test_resolution_risk_tightens_the_gate():
    panel = _panel(n_markets=40, label=1, price=0.40)
    probability = pd.Series(0.50, index=panel.index)
    safe = run_backtest(panel, probability, _config(min_edge=0.03))
    risky = run_backtest(panel, probability, _config(min_edge=0.03, resolution_risk=0.15))
    assert len(risky.trades) < len(safe.trades)


def test_a_failed_resolution_pays_the_recovery_value_not_the_outcome():
    panel = _panel(n_markets=60, label=1, price=0.40)
    result = run_backtest(
        panel, pd.Series(0.95, index=panel.index), _config(resolution_risk=0.5)
    )
    payoffs = set(result.trades["payoff"].round(6))
    assert 0.0 in payoffs  # some markets failed to resolve on merit
    assert 1.0 in payoffs  # the rest paid the true outcome


def test_which_markets_fail_does_not_depend_on_unrelated_settings():
    panel = _panel(n_markets=60, label=1, price=0.40)
    probability = pd.Series(0.95, index=panel.index)
    first = run_backtest(panel, probability, _config(resolution_risk=0.3))
    second = run_backtest(panel, probability, _config(resolution_risk=0.3, bankroll=50_000.0))
    failed = lambda run: set(run.trades.loc[run.trades["payoff"] == 0.0, "market_id"])  # noqa: E731
    smaller, larger = sorted((failed(first), failed(second)), key=len)
    # Changing the bankroll changes which markets are traded, but never which
    # ones the resolver botched.
    assert smaller <= larger


def test_resolution_settings_are_validated():
    with pytest.raises(ValueError):
        _config(resolution_risk=1.5)
    with pytest.raises(ValueError):
        _config(resolution_recovery=-0.1)


def _live_markets():
    fixture = simulate_arbitrage_snapshot(seed=7)
    markets = [replace(m, end_date=NOW + timedelta(days=30)) for m in fixture.markets]
    return markets, fixture


def test_order_plan_never_fills_worse_than_the_limit_that_justified_it():
    markets, fixture = _live_markets()
    forecasts = {
        m.market_id: min(fixture.books[m.yes_outcome.token_id].best_ask + 0.10, 0.97)
        for m in markets
    }
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    frame = plan.to_frame()
    assert not frame.empty
    assert (frame["effective_price"] <= frame["limit_price"] + 1e-9).all()
    assert (frame["edge"] >= RiskLimits().min_edge - 1e-9).all()


def test_order_plan_respects_the_aggregate_exposure_budget():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    limits = RiskLimits(max_total_exposure=0.15, max_orders=50)
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, limits, as_of=NOW)
    assert plan.total_notional <= 0.15 * 10_000 + 1e-6


def test_markets_without_a_forecast_are_skipped_with_a_reason():
    markets, fixture = _live_markets()
    plan = build_order_plan(markets, fixture.books, {}, 10_000, RiskLimits(), as_of=NOW)
    assert plan.orders == []
    assert all(entry["reason"] == "no forecast" for entry in plan.skipped)


def test_resolution_and_liquidity_gates_reject_before_sizing():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    far = build_order_plan(
        markets, fixture.books, forecasts, 10_000,
        RiskLimits(max_days_to_resolution=1.0), as_of=NOW,
    )
    assert far.orders == []
    assert any("too far out" in entry["reason"] for entry in far.skipped)

    illiquid = build_order_plan(
        markets, fixture.books, forecasts, 10_000,
        RiskLimits(min_book_depth_usd=1e12), as_of=NOW,
    )
    assert illiquid.orders == []
    assert any("min_book_depth_usd" in entry["reason"] for entry in illiquid.skipped)


def test_one_order_per_event_prevents_the_same_bet_under_several_names():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(
        markets, fixture.books, forecasts, 10_000,
        RiskLimits(one_order_per_event=True, max_orders=50), as_of=NOW,
    )
    events = plan.to_frame()["event_id"]
    assert events.is_unique


def test_deduplication_keeps_the_strongest_order_per_event():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(
        markets, fixture.books, forecasts, 10_000,
        RiskLimits(one_order_per_event=False, max_orders=100), as_of=NOW,
    )
    deduplicated = deduplicate_by_event(plan.orders)
    assert len(deduplicated) <= len(plan.orders)
    assert len({order.event_id for order in deduplicated}) == len(deduplicated)


def test_plan_serializes_with_its_rejections(tmp_path):
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    written = plan.to_json(tmp_path / "plan.json")
    assert written.exists()
    assert "skipped" in written.read_text()


def test_paper_broker_conserves_cash_and_settles_positions():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    broker = PaperBroker(bankroll=10_000)
    broker.submit(plan)
    committed = broker.state()["committed_capital"]
    assert committed == pytest.approx(plan.total_notional)
    assert broker.cash == pytest.approx(10_000 - committed)

    first = plan.orders[0]
    profit = broker.settle(first.market_id, 1 if first.side == "yes" else 0)
    assert profit > 0
    assert broker.state()["open_positions"] == len(plan.orders) - 1


def test_paper_broker_refuses_orders_it_cannot_fund():
    plan = OrderPlan(bankroll=10.0)
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    full = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    plan.orders = full.orders
    broker = PaperBroker(bankroll=1.0)
    ledger = broker.submit(plan)
    assert (ledger["status"] == "rejected_insufficient_cash").any()


def test_repeated_planning_cannot_ratchet_past_the_aggregate_cap():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    limits = RiskLimits(max_total_exposure=0.20, max_orders=50)
    broker = PaperBroker(bankroll=10_000)

    for _ in range(4):
        state = broker.state()
        plan = build_order_plan(
            markets, fixture.books, forecasts, broker.bankroll, limits,
            as_of=NOW, committed_capital=state["committed_capital"],
            available_cash=state["cash"],
        )
        broker.submit(plan)
    # Planning against leftover cash each run would have compounded well past
    # the limit; the cap belongs to the account.
    assert broker.state()["committed_capital"] <= 0.20 * 10_000 + 1e-6


def test_a_full_book_produces_an_empty_plan_with_a_reason():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    limits = RiskLimits(max_total_exposure=0.20)
    plan = build_order_plan(
        markets, fixture.books, forecasts, 10_000, limits, as_of=NOW,
        committed_capital=2_000, available_cash=8_000,
    )
    assert plan.orders == []
    assert any("exposure budget" in entry["reason"] for entry in plan.skipped)


def test_a_plan_is_capped_by_cash_as_well_as_by_exposure():
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(
        markets, fixture.books, forecasts, 10_000,
        RiskLimits(max_total_exposure=0.20, max_orders=50),
        as_of=NOW, available_cash=300.0,
    )
    assert plan.total_notional <= 300.0 + 1e-6


def test_paper_account_survives_a_save_and_reload(tmp_path):
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    broker = PaperBroker(bankroll=10_000)
    broker.submit(plan)
    path = broker.save(tmp_path / "account.json")

    restored = PaperBroker.load(path)
    assert restored.state() == broker.state()
    assert len(restored.fills) == len(broker.fills)
    assert sorted(restored.open_market_ids()) == sorted(broker.open_market_ids())


def test_loading_a_missing_account_opens_a_fresh_one(tmp_path):
    broker = PaperBroker.load(tmp_path / "absent.json", bankroll=777.0)
    assert broker.bankroll == 777.0
    assert broker.open_market_ids() == []


def test_a_restored_account_settles_the_positions_it_was_carrying(tmp_path):
    markets, fixture = _live_markets()
    forecasts = {m.market_id: 0.97 for m in markets}
    plan = build_order_plan(markets, fixture.books, forecasts, 10_000, RiskLimits(), as_of=NOW)
    broker = PaperBroker(bankroll=10_000)
    broker.submit(plan)
    broker.save(tmp_path / "account.json")

    restored = PaperBroker.load(tmp_path / "account.json")
    first = plan.orders[0]
    profit = restored.settle(first.market_id, 1 if first.side == "yes" else 0)
    assert profit > 0
    assert first.market_id not in restored.open_market_ids()


def test_arbitrage_baskets_expand_into_their_legs():
    fixture = simulate_arbitrage_snapshot(seed=7)
    found = scan_markets(
        fixture.markets, fixture.books, fixture.groups, max_capital=5_000, min_profit=1.0
    )
    legs = arbitrage_to_orders(found[0])
    assert len(legs) == len(found[0].legs)
    assert all(leg.rationale for leg in legs)
