"""Fetch Polymarket snapshots and build a resolved-market training panel.

Two artifacts are produced:

``markets_*.json`` / ``books_*.json``
    A point-in-time snapshot of tradable markets and their order books. The
    arbitrage scanner reads these so a scan is reproducible after the fact --
    a book that has since moved cannot be re-fetched.

``resolved_panel.csv``
    The supervised dataset. For every *settled* binary market the CLOB price
    history is sampled and each point is labelled with the outcome the market
    actually settled to. This is the only honest way to learn a forecast: the
    label has to come from a market that is already over.

Network access is required. When the venue is unreachable the research study
falls back to synthetic markets and says so in its report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_system.config import load_config
from quant_system.polymarket.client import PolymarketClient, SnapshotStore, token_ids
from quant_system.polymarket.markets import Market

ROOT = Path(__file__).resolve().parents[1]


def build_resolved_panel(
    client: PolymarketClient,
    markets: list[Market],
    interval: str = "max",
    points_per_market: int = 12,
    require_label: bool = True,
) -> pd.DataFrame:
    """Sample price history and, for settled markets, label it with the outcome.

    ``require_label=False`` reuses the same sampling for *open* markets, whose
    history a snapshot replay needs; the label column is then a placeholder that
    is scored against, never trained on.
    """
    rows: list[dict] = []
    for market in markets:
        label = market.yes_label
        yes = market.yes_outcome
        if yes is None or (require_label and label is None):
            continue
        try:
            history = client.price_history(yes.token_id, interval=interval)
        except Exception as error:  # noqa: BLE001 - one bad market must not stop the pull
            print(f"  skipped {market.market_id}: {error}")
            continue
        if len(history) < 2:
            continue

        step = max(len(history) // points_per_market, 1)
        for point in history[::step][:points_per_market]:
            rows.append(
                {
                    "market_id": market.market_id,
                    "question": market.question,
                    "timestamp": pd.to_datetime(point["t"], unit="s", utc=True),
                    "price": float(point["p"]),
                    "end_date": market.end_date,
                    "volume": market.volume,
                    "liquidity": market.liquidity,
                    "yes_token_id": yes.token_id,
                    "label": int(label) if label is not None else 0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "polymarket.yaml"))
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--skip-books", action="store_true", help="snapshot markets only")
    parser.add_argument("--skip-resolved", action="store_true", help="skip the training panel")
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="skip price history for open markets, which a snapshot replay needs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    max_markets = args.max_markets or int(data_cfg["max_markets"])
    store = SnapshotStore(ROOT / data_cfg["snapshot_dir"])
    client = PolymarketClient()

    print(f"Fetching up to {max_markets} active markets ...")
    active_raw = list(client.iter_markets(max_markets=max_markets, active=True, closed=False))
    store.save("markets", active_raw)
    print(f"  {len(active_raw)} market records saved")

    if not args.skip_books:
        from quant_system.polymarket.markets import parse_markets

        tokens = token_ids(parse_markets(active_raw))
        print(f"Fetching {len(tokens)} order books ...")
        books = client.order_books(tokens)
        store.save(
            "books",
            {
                token: {
                    "bids": [{"price": lv.price, "size": lv.size} for lv in book.bids],
                    "asks": [{"price": lv.price, "size": lv.size} for lv in book.asks],
                }
                for token, book in books.items()
            },
        )
        print(f"  {len(books)} books saved")

    if not args.skip_history:
        # A snapshot without price history cannot be replayed: the model is
        # fitted on momentum columns that a single point in time cannot supply.
        from quant_system.polymarket.markets import parse_markets as _parse

        open_markets = [m for m in _parse(active_raw) if m.is_binary]
        print(f"Fetching price history for {len(open_markets)} open markets ...")
        history = build_resolved_panel(
            client,
            open_markets,
            interval=str(data_cfg["history_interval"]),
            points_per_market=int(data_cfg["history_points_per_market"]),
            require_label=False,
        )
        store.save("history", history.assign(
            timestamp=history["timestamp"].astype(str),
            end_date=history["end_date"].astype(str),
        ).to_dict(orient="records") if not history.empty else [])
        print(f"  {len(history)} history rows saved")

    if not args.skip_resolved:
        print(f"Fetching up to {max_markets} resolved markets ...")
        resolved = client.fetch_markets(max_markets=max_markets, active=False, closed=True)
        settled = [market for market in resolved if market.is_binary and market.yes_label is not None]
        print(f"  {len(settled)} settled binary markets; sampling price history ...")
        panel = build_resolved_panel(
            client,
            settled,
            interval=str(data_cfg["history_interval"]),
            points_per_market=int(data_cfg["history_points_per_market"]),
        )
        target = ROOT / data_cfg["panel_csv"]
        if panel.empty:
            # Overwriting a good panel with an empty file turns one bad fetch
            # into a broken study: load_panel would see the file, read zero
            # columns, and raise instead of falling back to simulation.
            print(f"  no settled history retrieved; leaving {target} untouched")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            panel.to_csv(target, index=False)
            print(
                f"  wrote {len(panel)} rows across "
                f"{panel['market_id'].nunique()} markets -> {target}"
            )


if __name__ == "__main__":
    main()
