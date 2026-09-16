"""Operational loop: acquire, forecast, plan, paper-execute, persist.

The research scripts answer whether an edge exists. This one is what you would
actually run. It is safe to invoke repeatedly -- on a schedule, or by hand --
because the paper account is persisted between runs, so positions, cash, and the
fill history survive.

Each run does the same five things:

1. **Settle.** Any market the account holds that has since resolved is closed at
   its actual outcome, releasing the capital behind it.
2. **Acquire.** Open markets, their books, and enough price history that the
   live feature vector has the same shape as the one the model was trained on.
   Feature parity is not a detail: a model fitted with momentum columns cannot
   score a snapshot that lacks them, and silently filling them with zeros would
   score every market as if its price had never moved.
3. **Forecast.** Fit the market-anchored model on markets that have *already
   settled*, then score the open ones. Training data is always strictly in the
   past, exactly as in the research study.
4. **Scan.** Report structural arbitrage separately, because its edge does not
   depend on the forecast being any good.
5. **Plan and execute.** Build a risk-gated order plan and run it through the
   paper broker.

Nothing is ever signed or submitted to the venue. ``--source live`` reads public
data; the orders go to a local ledger. Wiring a real broker is a deliberate,
separate step behind the ``ExecutionAdapter`` protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_system.config import load_config
from quant_system.polymarket.arbitrage import opportunities_frame, scan_markets
from quant_system.polymarket.client import PolymarketClient, SnapshotStore, token_ids
from quant_system.polymarket.execution import PaperBroker, RiskLimits, build_order_plan
from quant_system.polymarket.features import build_snapshot_features, design_matrix
from quant_system.polymarket.markets import Market, group_by_event, parse_markets
from quant_system.polymarket.model import market_anchored_model
from quant_system.polymarket.orderbook import OrderBook
from quant_system.polymarket.simulation import PanelSpec, simulate_panel

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Acquisition:
    """Everything one run needs, including the clock it should be judged by.

    ``as_of`` travels with the data rather than being read from the wall clock.
    A replayed snapshot, a simulated world, and a live fetch all have their own
    notion of "now", and using the wrong one makes the resolution-window gates
    reject markets that were tradable when the data was captured.
    """

    markets: list[Market]
    books: dict[str, OrderBook]
    resolved: pd.DataFrame
    live_history: pd.DataFrame
    as_of: datetime


def acquire_simulation(config: dict) -> Acquisition:
    """Build a self-contained live-like world: settled history plus open markets.

    The panel is split by time rather than at random: the markets that resolved
    before the cut become training data and the ones still running become the
    tradable universe, which is the same split the live path produces naturally.
    """
    from quant_system.polymarket.markets import Outcome

    spec = config["simulation"]
    panel = simulate_panel(
        PanelSpec(
            n_markets=int(spec["n_markets"]),
            observations_per_market=int(spec["observations_per_market"]),
            horizon_days=int(spec["horizon_days"]),
            mispricing_sigma=float(spec["mispricing_sigma"]),
            signal_strength=float(spec["signal_strength"]),
            mispricing_decay=float(spec["mispricing_decay"]),
            spread=float(spec["spread"]),
            seed=int(spec["seed"]),
        )
    )
    cut = panel["timestamp"].quantile(0.75)
    resolved = panel[panel["end_date"] <= cut]
    live_ids = panel.loc[panel["end_date"] > cut, "market_id"].unique()
    live_history = panel[panel["market_id"].isin(live_ids) & (panel["timestamp"] <= cut)]

    markets: list[Market] = []
    books: dict[str, OrderBook] = {}
    for market_id, group in live_history.groupby("market_id"):
        last = group.sort_values("timestamp").iloc[-1]
        yes_token, no_token = f"{market_id}-Y", f"{market_id}-N"
        markets.append(
            Market(
                market_id=str(market_id),
                question=str(last["question"]),
                slug=str(market_id).lower(),
                condition_id=f"0x{market_id}",
                outcomes=(Outcome("Yes", yes_token), Outcome("No", no_token)),
                end_date=last["end_date"].to_pydatetime(),
                event_id=str(market_id),
                event_slug=str(market_id).lower(),
                volume=float(last["volume"]),
                liquidity=float(last["liquidity"]),
            )
        )
        depth = float(last["ask_depth"])
        books[yes_token] = OrderBook.from_levels(
            yes_token,
            [(float(last["best_bid"]), depth)],
            [(float(last["best_ask"]), depth)],
        )
        books[no_token] = OrderBook.from_levels(
            no_token,
            [(1.0 - float(last["best_ask"]), depth)],
            [(1.0 - float(last["best_bid"]), depth)],
        )
    return Acquisition(markets, books, resolved, live_history, cut.to_pydatetime())


def acquire_live(
    config: dict,
    client: PolymarketClient,
    max_markets: int,
) -> Acquisition:
    """Fetch open markets with books, plus the settled panel used for training."""
    data_cfg = config["data"]
    panel_path = ROOT / data_cfg["panel_csv"]
    if not panel_path.exists():
        raise FileNotFoundError(
            f"No training panel at {panel_path}. Run scripts/fetch_polymarket_data.py "
            "first: the model can only learn from markets that have already settled."
        )
    resolved = pd.read_csv(panel_path, parse_dates=["timestamp", "end_date"])

    markets = [
        market
        for market in client.fetch_markets(max_markets=max_markets, active=True, closed=False)
        if market.is_binary and market.volume >= float(data_cfg["min_volume"])
    ]
    books = client.order_books(token_ids(markets))

    rows: list[dict] = []
    points = int(data_cfg["history_points_per_market"])
    for market in markets:
        yes = market.yes_outcome
        if yes is None:
            continue
        try:
            history = client.price_history(yes.token_id, interval=str(data_cfg["history_interval"]))
        except Exception as error:  # noqa: BLE001 - one bad market must not stop the run
            print(f"  no history for {market.market_id}: {error}")
            continue
        step = max(len(history) // points, 1)
        for point in history[::step][-points:]:
            rows.append(
                {
                    "market_id": market.market_id,
                    "question": market.question,
                    "timestamp": pd.to_datetime(point["t"], unit="s", utc=True),
                    "price": float(point["p"]),
                    "end_date": market.end_date,
                    "volume": market.volume,
                    "liquidity": market.liquidity,
                    "label": 0,  # unknown; never used, the live rows are only scored
                }
            )
    return Acquisition(
        markets, books, resolved, pd.DataFrame(rows), datetime.now(timezone.utc)
    )


def acquire_snapshot(config: dict) -> Acquisition:
    """Replay the most recent stored snapshot instead of calling the venue."""
    store = SnapshotStore(ROOT / config["data"]["snapshot_dir"])
    markets = [market for market in parse_markets(store.load("markets")) if market.is_binary]
    raw_books = store.load("books")
    books = {
        token: OrderBook.from_levels(token, payload.get("bids", []), payload.get("asks", []))
        for token, payload in raw_books.items()
    }
    panel_path = ROOT / config["data"]["panel_csv"]
    if not panel_path.exists():
        raise FileNotFoundError(f"No training panel at {panel_path}")
    resolved = pd.read_csv(panel_path, parse_dates=["timestamp", "end_date"])
    live_history = resolved.iloc[0:0].copy()
    captured = store.captured_at("markets") or datetime.now(timezone.utc)
    return Acquisition(markets, books, resolved, live_history, captured)


def forecast_live_markets(
    config: dict,
    resolved: pd.DataFrame,
    live_history: pd.DataFrame,
) -> dict[str, float]:
    """Fit on settled markets and score the latest observation of each open one.

    Training and scoring frames are built by the same call, so a feature that
    cannot be computed live is never one the model was fitted on.
    """
    model_cfg = config["model"]
    passthrough = [
        column
        for column in model_cfg.get("passthrough_features", []) or []
        if column in resolved.columns and column in live_history.columns
    ]
    windows = tuple(model_cfg["momentum_windows"])
    volatility_window = int(model_cfg["volatility_window"])

    train_frame = build_snapshot_features(resolved, windows, volatility_window,
                                          passthrough_columns=passthrough)
    X, y, _ = design_matrix(train_frame)
    if len(X) < 100 or y.nunique() < 2:
        print(f"  refusing to forecast: only {len(X)} usable settled observations")
        return {}

    model = market_anchored_model(alpha=model_cfg.get("anchor_alpha", "auto"))
    model.alpha_grid = tuple(model_cfg["anchor_alpha_grid"])
    counts = train_frame.loc[X.index].groupby("market_id")["price"].transform("size")
    model.fit(X, y, sample_weight=(1.0 / counts.astype(float)).to_numpy())

    live_frame = build_snapshot_features(live_history, windows, volatility_window,
                                         passthrough_columns=passthrough)
    live_X, _, live_meta = design_matrix(live_frame, dropna=True)
    if live_X.empty:
        print("  no live market had enough history for a complete feature vector")
        return {}
    live_X = live_X[X.columns]
    probability = model.predict_proba(live_X)[:, 1]

    latest = (
        pd.DataFrame({"market_id": live_meta["market_id"], "probability": probability})
        .groupby("market_id")
        .last()["probability"]
    )
    return {str(k): float(v) for k, v in latest.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "polymarket.yaml"))
    parser.add_argument("--source", choices=("simulation", "snapshot", "live"), default="simulation")
    parser.add_argument("--state", default=str(ROOT / "reports" / "paper_account.json"))
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="build and print the plan without recording it in the paper account",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    risk_cfg, data_cfg = config["risk"], config["data"]
    fee_bps = float(config["frictions"]["fee_bps"])
    max_markets = args.max_markets or int(data_cfg["max_markets"])

    broker = PaperBroker.load(args.state, bankroll=float(risk_cfg["bankroll"]))
    broker.fee_bps = fee_bps
    print(f"Paper account: {broker.state()}")

    print(f"Acquiring markets from: {args.source}")
    if args.source == "live":
        acquired = acquire_live(config, PolymarketClient(), max_markets)
    elif args.source == "snapshot":
        acquired = acquire_snapshot(config)
    else:
        acquired = acquire_simulation(config)
    markets, books = acquired.markets, acquired.books
    resolved, live_history, now = acquired.resolved, acquired.live_history, acquired.as_of
    print(
        f"  {len(markets)} open markets, {len(books)} books, "
        f"{resolved['market_id'].nunique()} settled markets for training"
    )
    print(f"  evaluating as of {now.isoformat()}")

    # Settle anything the account holds whose market has since resolved. A
    # position left open past resolution overstates both committed capital and
    # the account's risk.
    settleable = {m.market_id: m.yes_label for m in markets if m.is_resolved}
    realized = 0.0
    for market_id in broker.open_market_ids():
        label = settleable.get(market_id)
        if label is not None:
            realized += broker.settle(market_id, int(label))
    if realized:
        print(f"  settled matured positions for {realized:+,.2f}")

    print("Scanning for structural arbitrage ...")
    arbitrage_cfg = config["arbitrage"]
    opportunities = scan_markets(
        markets,
        books,
        group_by_event(markets),
        fee_bps=fee_bps,
        max_capital=float(arbitrage_cfg["max_capital_per_basket"]),
        min_profit=float(arbitrage_cfg["min_profit"]),
        min_shares=float(arbitrage_cfg["min_shares"]),
        include_mint_and_sell=bool(arbitrage_cfg["include_mint_and_sell"]),
        require_verified_groups=bool(arbitrage_cfg["require_verified_groups"]),
    )
    if opportunities:
        frame = opportunities_frame(opportunities)
        print(
            f"  {len(frame)} baskets, {frame['profit'].sum():+,.2f} locked profit on "
            f"{frame['capital_required'].sum():,.2f} capital"
        )
        print("  these do not depend on the forecast and are reported, not auto-executed")
    else:
        print("  none: no basket cleared its cost after fees and depth")

    print("Forecasting open markets ...")
    probabilities = forecast_live_markets(config, resolved, live_history)
    print(f"  {len(probabilities)} markets scored")

    limits = RiskLimits(
        kelly_multiplier=float(risk_cfg["kelly_multiplier"]),
        max_fraction_per_market=float(risk_cfg["max_fraction_per_market"]),
        max_total_exposure=float(risk_cfg["max_total_exposure"]),
        min_edge=float(risk_cfg["min_edge"]),
        shrinkage=float(risk_cfg["shrinkage"]),
        min_notional=float(risk_cfg["min_notional"]),
        max_spread=float(risk_cfg["max_spread"]),
        min_book_depth_usd=float(risk_cfg["min_book_depth_usd"]),
        min_days_to_resolution=float(risk_cfg["min_days_to_resolution"]),
        max_days_to_resolution=float(risk_cfg["max_days_to_resolution"]),
        max_orders=int(risk_cfg["max_orders"]),
        one_order_per_event=bool(risk_cfg["one_order_per_event"]),
    )
    # The aggregate exposure cap belongs to the account, not to this run: pass
    # what is already committed so repeated invocations cannot ratchet past it.
    state = broker.state()
    plan = build_order_plan(
        markets,
        books,
        probabilities,
        broker.bankroll,
        limits,
        fee_bps,
        as_of=now,
        committed_capital=state["committed_capital"],
        available_cash=state["cash"],
    )

    report_dir = ROOT / "reports"
    plan_path = plan.to_json(report_dir / "polymarket_order_plan.json")
    if plan.orders:
        columns = ["market_id", "side", "shares", "limit_price", "notional", "edge"]
        print(plan.to_frame()[columns].round(4).to_string(index=False))
        print(
            f"  {len(plan.orders)} orders, {plan.total_notional:,.2f} notional, "
            f"{plan.total_expected_profit:+,.2f} expected profit"
        )
    else:
        print("  no market cleared every gate; nothing to trade")
    print(f"  plan written to {plan_path.relative_to(ROOT)} with {len(plan.skipped)} rejections")

    if args.plan_only:
        print("\nPlan-only: the paper account was not touched.")
        return

    broker.submit(plan)
    state_path = broker.save(args.state)
    print(f"\nPaper account after execution: {broker.state()}")
    print(f"  state saved to {state_path}")
    print("\nNothing was transmitted to the venue. This is a local ledger.")


if __name__ == "__main__":
    main()
