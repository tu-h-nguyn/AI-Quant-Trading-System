"""Two-sided quoting, inventory control, and maker-side backtesting.

Every strategy elsewhere in this package is a taker: it crosses the spread and
pays for immediacy. A maker is paid for it instead, which on a venue quoting one
to four cents against a one-dollar payoff is frequently larger than any forecast
edge available. That makes market making the other half of the profitability
question, and it fails for a different reason than forecasting does.

**The reason it fails is adverse selection.** A resting quote is not filled at
random. It is filled precisely when someone wants the other side, which is
disproportionately when they know something -- so a maker buys just before the
price falls and sells just before it rises. A fill simulation that fills a quote
whenever the price touches it, and then marks the position at the touched price,
reports a maker profit that does not exist.

:func:`simulate_market_making` therefore splits flow into two regimes:

* **Informed flow.** When the price moves through a quote, that quote is filled
  and the resulting position is immediately marked at the new price. The loss is
  automatic and is the honest cost of providing liquidity.
* **Uninformed flow.** A configurable share of periods produce a fill that is
  *not* associated with a price move. This is where a maker's profit comes from,
  and ``uninformed_fill_rate`` is an assumption about the venue rather than
  something this package can measure. Set it to zero and market making loses
  money by construction, which is the correct null.

The reported P&L is decomposed into the spread that was quoted and everything
else, so the two forces are visible separately rather than netted into one
number that cannot be diagnosed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .pricing import MAX_PRICE, MIN_PRICE, taker_fee_per_share

REQUIRED_PANEL_COLUMNS = ("market_id", "timestamp", "price", "label")


@dataclass(frozen=True)
class QuoteConfig:
    """Quoting geometry, inventory limits, and the assumed flow mix."""

    half_spread: float = 0.02
    quote_size: float = 100.0
    max_inventory: float = 400.0
    inventory_skew: float = 1.0
    # A maker quotes a chosen subset, not the whole venue. Capping concurrency
    # bounds the capital the book can tie up at any one moment.
    max_concurrent_markets: int = 50
    maker_fee_bps: float = 0.0
    # Share of quoting periods that produce a fill unrelated to a price move.
    # This is the entire source of maker profit and is an assumption, not a
    # measurement; zero makes market making unprofitable by construction.
    uninformed_fill_rate: float = 0.15
    # Quotes are widened as resolution approaches, where flow is most informed.
    widen_within_days: float = 2.0
    widen_multiple: float = 2.0
    min_quote_price: float = 0.02
    max_quote_price: float = 0.98
    seed: int = 42

    def __post_init__(self) -> None:
        if self.half_spread <= 0:
            raise ValueError("half_spread must be positive")
        if self.quote_size <= 0 or self.max_inventory <= 0:
            raise ValueError("quote_size and max_inventory must be positive")
        if not 0.0 <= self.uninformed_fill_rate <= 1.0:
            raise ValueError("uninformed_fill_rate must lie in [0, 1]")
        if self.inventory_skew < 0:
            raise ValueError("inventory_skew must be non-negative")
        if self.widen_multiple < 1.0:
            raise ValueError("widen_multiple must be at least 1")
        if self.max_concurrent_markets < 1:
            raise ValueError("max_concurrent_markets must be at least 1")


@dataclass(frozen=True)
class Quote:
    """A two-sided quote, already skewed for inventory and clipped to the unit interval."""

    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class MarketMakingResult:
    """Fill ledger, per-market outcomes, and the P&L decomposition."""

    fills: pd.DataFrame
    markets: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def make_quotes(
    fair_value: float,
    inventory: float,
    config: QuoteConfig,
    days_to_resolution: float | None = None,
) -> Quote:
    """Quote both sides around ``fair_value``, leaning against inventory.

    The quote centre is shifted away from the side that would deepen an existing
    position, so a maker that has accumulated a long position quotes lower and
    is more likely to be hit on its offer. Size is cut to zero on the side that
    would breach ``max_inventory``, which is what keeps a one-directional flow
    from turning a market-making book into a directional bet.
    """
    fair = float(np.clip(fair_value, MIN_PRICE, MAX_PRICE))
    half_spread = config.half_spread
    if days_to_resolution is not None and days_to_resolution <= config.widen_within_days:
        # Flow concentrates and informs as a market resolves; quoting the same
        # width into it is how a maker donates the spread back.
        half_spread *= config.widen_multiple

    skew = config.inventory_skew * (inventory / config.max_inventory) * half_spread
    centre = fair - skew

    bid = float(np.clip(centre - half_spread, config.min_quote_price, config.max_quote_price))
    ask = float(np.clip(centre + half_spread, config.min_quote_price, config.max_quote_price))
    if ask <= bid:
        ask = min(bid + 1e-3, config.max_quote_price)

    # Room left before the inventory cap binds, on each side.
    bid_size = min(config.quote_size, max(config.max_inventory - inventory, 0.0))
    ask_size = min(config.quote_size, max(config.max_inventory + inventory, 0.0))
    return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)


def simulate_market_making(
    panel: pd.DataFrame,
    config: QuoteConfig | None = None,
    fair_values: pd.Series | None = None,
    bankroll: float = 10_000.0,
) -> MarketMakingResult:
    """Quote every market in ``panel`` chronologically and settle its inventory.

    The loop runs in calendar order across the whole panel, not market by
    market, because a maker's markets overlap: capital committed to one book is
    unavailable to another until that market settles. Processing each market to
    completion in turn would let the same dollar quote a hundred markets in
    sequence and report a capital efficiency no maker has.

    ``fair_values`` supplies the maker's own estimate of each market's
    probability; without it the maker quotes around the market price and has no
    informational advantage, which isolates the spread-versus-adverse-selection
    trade-off. Providing a model forecast lets the maker skew toward its edge.

    Capital is enforced, not assumed: a short YES position is collateralized at
    one dollar per share because that is what settlement can demand, and a fill
    the cash balance cannot fund is declined and counted rather than silently
    financed.
    """
    settings = config or QuoteConfig()
    missing = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")

    frame = panel.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["market_id", "timestamp"], kind="mergesort")
    frame["fair_value"] = (
        pd.Series(fair_values).reindex(frame.index).astype(float)
        if fair_values is not None
        else frame["price"].astype(float)
    )
    # The price one observation ahead is what an informed fill is marked at.
    frame["next_price"] = frame.groupby("market_id", sort=False)["price"].shift(-1)
    if "end_date" in frame.columns:
        frame["settle_at"] = pd.to_datetime(frame["end_date"], utc=True)
    else:
        frame["settle_at"] = frame.groupby("market_id", sort=False)["timestamp"].transform("max")
    frame = frame.sort_values(["timestamp", "market_id"], kind="mergesort")
    final_time = frame["timestamp"].max()

    rng = np.random.default_rng(settings.seed)
    cash = float(bankroll)
    peak_committed = 0.0
    declined = 0
    books: dict[str, _Book] = {}
    settled: dict[str, dict] = {}
    fills: list[dict] = []

    for row in frame.itertuples(index=False):
        cash = _settle_matured(books, settled, row.timestamp, cash)
        if not np.isfinite(row.next_price):
            continue  # the market's last observation quotes nothing it can mark

        book = books.get(row.market_id)
        if book is None:
            if len(books) >= settings.max_concurrent_markets:
                continue
            book = _Book(
                market_id=row.market_id,
                label=int(row.label),
                settle_at=row.settle_at if pd.notna(row.settle_at) else final_time,
            )
            books[row.market_id] = book

        fair = row.fair_value if np.isfinite(row.fair_value) else row.price
        days_left = max(
            (book.settle_at - row.timestamp).total_seconds() / 86_400.0, 0.0
        )
        quote = make_quotes(fair, book.inventory, settings, days_left)

        events: list[tuple[str, float, bool]] = []
        if row.next_price >= quote.ask and quote.ask_size > 0:
            events.append(("sell", quote.ask, True))
        if row.next_price <= quote.bid and quote.bid_size > 0:
            events.append(("buy", quote.bid, True))
        if settings.uninformed_fill_rate > 0:
            if rng.uniform() < settings.uninformed_fill_rate and quote.ask_size > 0:
                events.append(("sell", quote.ask, False))
            if rng.uniform() < settings.uninformed_fill_rate and quote.bid_size > 0:
                events.append(("buy", quote.bid, False))

        # A resting quote has finite size: informed and uninformed flow in the
        # same period compete for it rather than each taking the full amount,
        # and neither may carry inventory past its cap.
        remaining = {"buy": quote.bid_size, "sell": quote.ask_size}
        for side, price, informed in events:
            wanted = min(remaining[side], _inventory_room(side, book.inventory, settings))
            size = _affordable_size(side, wanted, price, book.inventory, cash)
            if size < wanted - 1e-9:
                declined += 1
            if size <= 0:
                continue
            remaining[side] -= size
            fee = size * taker_fee_per_share(price, settings.maker_fee_bps)
            if side == "buy":
                cash -= size * price + fee
                book.cash_flow -= size * price + fee
                book.inventory += size
            else:
                cash += size * price - fee
                book.cash_flow += size * price - fee
                book.inventory -= size
            # The quote's own half-spread, not the configured one: inside the
            # widening window they differ by widen_multiple, and both headline
            # maker diagnostics are ratios against this figure.
            book.quoted_spread_value += size * (quote.spread / 2.0)
            book.n_informed += int(informed)
            book.n_uninformed += int(not informed)
            fills.append(
                {
                    "market_id": row.market_id,
                    "timestamp": row.timestamp,
                    "side": side,
                    "price": price,
                    "size": size,
                    "informed": informed,
                    "inventory_after": book.inventory,
                    "mark_price": row.next_price,
                    # Immediate mark-to-market: negative on an informed fill by
                    # construction, which is what adverse selection is.
                    "immediate_pnl": (row.next_price - price)
                    * (size if side == "buy" else -size),
                    "fee": fee,
                }
            )
        committed = sum(_collateral(item.inventory, 1.0) for item in books.values())
        peak_committed = max(peak_committed, committed)

    cash = _settle_matured(books, settled, pd.Timestamp.max.tz_localize("UTC"), cash)

    fills_frame = pd.DataFrame(fills) if fills else _empty_fills()
    markets_frame = (
        pd.DataFrame(list(settled.values())) if settled else _empty_markets()
    )
    summary = summarize_market_making(fills_frame, markets_frame, bankroll, cash)
    summary["peak_committed_capital"] = peak_committed
    summary["declined_fills"] = float(declined)
    summary["capital_constrained"] = float(declined > 0)
    return MarketMakingResult(fills=fills_frame, markets=markets_frame, summary=summary)


@dataclass
class _Book:
    """Per-market inventory and running diagnostics while the market is live."""

    market_id: str
    label: int
    settle_at: pd.Timestamp
    inventory: float = 0.0
    quoted_spread_value: float = 0.0
    cash_flow: float = 0.0
    n_informed: int = 0
    n_uninformed: int = 0


def _settle_matured(
    books: dict[str, _Book],
    settled: dict[str, dict],
    now: pd.Timestamp,
    cash: float,
) -> float:
    """Settle every market matured at ``now``, releasing its collateral."""
    for market_id in [key for key, item in books.items() if item.settle_at <= now]:
        book = books.pop(market_id)
        settlement = book.inventory * float(book.label)
        cash += settlement
        settled[market_id] = {
            "market_id": market_id,
            "label": book.label,
            "final_inventory": book.inventory,
            "settlement_value": settlement,
            "pnl": book.cash_flow + settlement,
            "quoted_spread_value": book.quoted_spread_value,
            "adverse_selection_pnl": book.cash_flow + settlement - book.quoted_spread_value,
            "n_fills": book.n_informed + book.n_uninformed,
            "n_informed_fills": book.n_informed,
            "n_uninformed_fills": book.n_uninformed,
        }
    return cash


def _collateral(inventory: float, price: float) -> float:
    """Cash a position ties up: its cost when long, a dollar a share when short."""
    return max(inventory, 0.0) * price + max(-inventory, 0.0) * 1.0


def summarize_market_making(
    fills: pd.DataFrame,
    markets: pd.DataFrame,
    bankroll: float,
    final_cash: float,
) -> dict[str, float]:
    """Aggregate a maker run, keeping spread and adverse selection separate.

    ``spread_capture_ratio`` is the diagnostic that decides whether the book is
    worth running: it is realized P&L over the spread that was quoted. One means
    every quoted cent was kept, and anything at or below zero means the flow took
    back more than the spread paid.
    """
    if markets.empty:
        return {
            "n_markets": 0.0,
            "n_fills": 0.0,
            "total_pnl": 0.0,
            "return_on_bankroll": 0.0,
            "quoted_spread_value": 0.0,
            "adverse_selection_pnl": 0.0,
            "spread_capture_ratio": float("nan"),
            "informed_fill_share": float("nan"),
            "mean_immediate_pnl": float("nan"),
            "profitable_market_share": float("nan"),
            "mean_absolute_final_inventory": 0.0,
        }
    total_pnl = float(markets["pnl"].sum())
    quoted = float(markets["quoted_spread_value"].sum())
    n_fills = float(markets["n_fills"].sum())
    return {
        "n_markets": float(len(markets)),
        "n_fills": n_fills,
        "total_pnl": total_pnl,
        "return_on_bankroll": float((final_cash - bankroll) / bankroll),
        "quoted_spread_value": quoted,
        "adverse_selection_pnl": total_pnl - quoted,
        "spread_capture_ratio": total_pnl / quoted if quoted > 0 else float("nan"),
        "informed_fill_share": float(markets["n_informed_fills"].sum() / n_fills)
        if n_fills > 0
        else float("nan"),
        "mean_immediate_pnl": float(fills["immediate_pnl"].mean()) if not fills.empty else float("nan"),
        "profitable_market_share": float((markets["pnl"] > 0).mean()),
        "mean_absolute_final_inventory": float(markets["final_inventory"].abs().mean()),
    }


def _inventory_room(side: str, inventory: float, config: QuoteConfig) -> float:
    """Shares that may still be added on ``side`` before the cap binds."""
    if side == "buy":
        return max(config.max_inventory - inventory, 0.0)
    return max(config.max_inventory + inventory, 0.0)


def _affordable_size(
    side: str,
    requested: float,
    price: float,
    inventory: float,
    cash: float,
) -> float:
    """Shrink a fill to what the cash balance can actually collateralize.

    A fill is split at the point where it crosses flat, because the two halves
    cost different things. Closing an existing position releases collateral and
    is free to the cash balance; opening a new one costs the price for a long
    and a dollar less the price for a short, since a dollar is what settlement
    can demand of it.

    Charging the whole fill at the rate implied by the *pre-trade* sign is what
    an earlier version did, and it financed positions out of nothing: selling
    400 against an inventory of +50 was treated as entirely closing, so a
    350-share naked short appeared with one dollar of cash behind it.
    """
    if requested <= 0:
        return 0.0
    # Closing capacity is available even at zero cash: unwinding a position
    # releases collateral rather than consuming it.
    closing_capacity = max(-inventory, 0.0) if side == "buy" else max(inventory, 0.0)
    opening_rate = price if side == "buy" else 1.0 - price
    if opening_rate <= 0:
        return requested
    # Closing is free, so the budget only has to cover whatever opens beyond it.
    affordable = closing_capacity + max(cash, 0.0) / opening_rate
    return float(min(requested, affordable))


def _empty_fills() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market_id", "timestamp", "side", "price", "size", "informed",
            "inventory_after", "mark_price", "immediate_pnl", "fee",
        ]
    )


def _empty_markets() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market_id", "label", "final_inventory", "settlement_value", "pnl",
            "quoted_spread_value", "adverse_selection_pnl", "n_fills",
            "n_informed_fills", "n_uninformed_fills",
        ]
    )


def sweep_uninformed_fill_rate(
    panel: pd.DataFrame,
    rates: Sequence[float],
    config: QuoteConfig | None = None,
    fair_values: pd.Series | None = None,
    bankroll: float = 10_000.0,
) -> pd.DataFrame:
    """Re-run the maker across assumed flow mixes and tabulate the outcome.

    The single most consequential input to a market-making book is not the
    spread it quotes but the share of its fills that are uninformed, and that is
    a property of the venue that no backtest on price history can measure. The
    sweep replaces a false point estimate with the break-even condition: read off
    the rate at which P&L crosses zero and judge whether the real order flow
    plausibly clears it.
    """
    base = config or QuoteConfig()
    rows = []
    for rate in rates:
        result = simulate_market_making(
            panel, replace(base, uninformed_fill_rate=float(rate)), fair_values, bankroll
        )
        rows.append({"uninformed_fill_rate": float(rate), **result.summary})
    return pd.DataFrame(rows)


def break_even_uninformed_rate(sweep: pd.DataFrame) -> float:
    """Interpolate the flow mix at which a maker's P&L crosses zero.

    Returns ``nan`` when the sweep never changes sign, which means the answer
    lies outside the range examined rather than at its edge.
    """
    if sweep.empty or "total_pnl" not in sweep.columns:
        return float("nan")
    ordered = sweep.sort_values("uninformed_fill_rate")
    rates = ordered["uninformed_fill_rate"].to_numpy(dtype=float)
    pnl = ordered["total_pnl"].to_numpy(dtype=float)
    for index in range(len(pnl) - 1):
        low, high = pnl[index], pnl[index + 1]
        if low <= 0 <= high and high != low:
            weight = -low / (high - low)
            return float(rates[index] + weight * (rates[index + 1] - rates[index]))
    return float("nan")
