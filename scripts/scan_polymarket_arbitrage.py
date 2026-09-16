"""Scan for structural (model-free) mispricings on Polymarket.

Reads either a stored snapshot or live books, prices every candidate basket
through the order book after fees, and writes a ranked report. Nothing is
submitted: the output is a plan.

The scan answers one question -- "is the venue currently quoting a basket below
its guaranteed settlement value, in size, after costs?" -- and a run that finds
nothing is the expected outcome on a liquid venue. An empty report is a working
scanner, not a broken one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_system.config import load_config
from quant_system.polymarket.arbitrage import opportunities_frame, scan_markets
from quant_system.polymarket.client import PolymarketClient, SnapshotStore, token_ids
from quant_system.polymarket.execution import arbitrage_to_orders
from quant_system.polymarket.markets import group_by_event, parse_markets
from quant_system.polymarket.orderbook import OrderBook
from quant_system.polymarket.simulation import simulate_arbitrage_snapshot

ROOT = Path(__file__).resolve().parents[1]


def load_snapshot(store: SnapshotStore):
    """Rebuild markets, books, and event groups from the latest stored snapshot."""
    markets = parse_markets(store.load("markets"))
    raw_books = store.load("books")
    books = {
        token: OrderBook.from_levels(token, payload.get("bids", []), payload.get("asks", []))
        for token, payload in raw_books.items()
    }
    return markets, books, group_by_event(markets)


def load_live(client: PolymarketClient, max_markets: int, min_volume: float):
    """Fetch tradable markets and their books directly from the venue."""
    markets = [
        market
        for market in client.fetch_markets(max_markets=max_markets, active=True, closed=False)
        if market.volume >= min_volume
    ]
    books = client.order_books(token_ids(markets))
    return markets, books, group_by_event(markets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "polymarket.yaml"))
    parser.add_argument(
        "--source",
        choices=("snapshot", "live", "simulation"),
        default="snapshot",
        help="snapshot replays stored books; simulation uses synthetic ones for a self-test",
    )
    parser.add_argument("--output", default=str(ROOT / "reports" / "polymarket_arbitrage.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    arbitrage_cfg, data_cfg = config["arbitrage"], config["data"]
    fee_bps = float(config["frictions"]["fee_bps"])

    if args.source == "simulation":
        fixture = simulate_arbitrage_snapshot()
        markets, books, groups = fixture.markets, fixture.books, fixture.groups
        planted = sum(len(value) for value in fixture.planted.values())
        print(f"Synthetic self-test: {planted} arbitrages planted across {len(markets)} markets")
    elif args.source == "live":
        markets, books, groups = load_live(
            PolymarketClient(),
            int(data_cfg["max_markets"]),
            float(data_cfg["min_volume"]),
        )
    else:
        markets, books, groups = load_snapshot(SnapshotStore(ROOT / data_cfg["snapshot_dir"]))

    print(f"Scanning {len(markets)} markets, {len(books)} books, {len(groups)} event groups ...")
    opportunities = scan_markets(
        markets,
        books,
        groups,
        fee_bps=fee_bps,
        max_capital=float(arbitrage_cfg["max_capital_per_basket"]),
        min_profit=float(arbitrage_cfg["min_profit"]),
        min_shares=float(arbitrage_cfg["min_shares"]),
        include_mint_and_sell=bool(arbitrage_cfg["include_mint_and_sell"]),
        require_verified_groups=bool(arbitrage_cfg["require_verified_groups"]),
    )

    frame = opportunities_frame(opportunities)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if frame.empty:
        print("No basket cleared its cost after fees and depth. Nothing to trade.")
    else:
        columns = ["kind", "shares", "capital_required", "profit", "roi", "verified_exhaustive"]
        print(frame[columns].head(20).to_string(index=False))
        print(
            f"\n{len(frame)} opportunities | total profit "
            f"{frame['profit'].sum():.2f} on {frame['capital_required'].sum():.2f} capital"
        )
        legs = arbitrage_to_orders(opportunities[0])
        print(f"\nBest basket expands to {len(legs)} legs:")
        for leg in legs:
            print(
                f"  {leg.action:4s} {leg.shares:10.2f} {leg.side:>3s} @ "
                f"{leg.effective_price:.4f}  {leg.question[:60]}"
            )
        print(
            "\nLegs are not atomic on-venue: a book that moves between fills leaves "
            "an outright position, so treat these sizes as an upper bound."
        )
    print(f"\nWrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
