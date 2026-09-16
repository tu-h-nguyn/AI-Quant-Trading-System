"""Structural, model-free mispricing scanners.

Everything in this module is an accounting identity rather than a forecast. A
binary market's two outcomes must settle to exactly one dollar between them, and
a venue-guaranteed exhaustive outcome set must settle to exactly one dollar
across all its members. When the executable cost of assembling such a basket is
below its guaranteed settlement value, the profit does not depend on being right
about anything.

Three qualifications keep that from being a free lunch, and each is modelled:

* **Depth.** Baskets are sized by walking every leg's book jointly, so the
  reported size is what the snapshot could actually absorb, not top-of-book.
* **Fees.** Every leg is charged the configured taker fee before profit is
  computed.
* **Exhaustiveness.** A basket is only an identity when the outcome set really
  does partition the state space. Only Polymarket ``negRisk`` groups carry that
  guarantee, and opportunities from any other grouping are flagged unverified.

What is *not* modelled is execution risk: the legs are not atomic, so a book
that moves between the first and last fill turns a locked basket into an
outright position. Treat the sizes here as an upper bound.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .markets import Market, MarketGroup
from .orderbook import BUY, SELL, Fill, OrderBook

BINARY_COMPLEMENT = "binary_complement"
BINARY_MINT_AND_SELL = "binary_mint_and_sell"
GROUP_DUTCH_BOOK = "group_dutch_book"
NEG_RISK_NO_BASKET = "neg_risk_no_basket"

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ArbitrageLeg:
    """One executed side of a basket, priced through the book."""

    market_id: str
    question: str
    token_id: str
    outcome_name: str
    side: str
    shares: float
    average_price: float
    effective_price: float
    notional: float
    fees: float
    levels_consumed: int

    @classmethod
    def from_fill(
        cls,
        market: Market,
        token_id: str,
        outcome_name: str,
        fill: Fill,
    ) -> ArbitrageLeg:
        return cls(
            market_id=market.market_id,
            question=market.question,
            token_id=token_id,
            outcome_name=outcome_name,
            side=fill.side,
            shares=fill.shares,
            average_price=fill.average_price,
            effective_price=fill.effective_price,
            notional=fill.notional,
            fees=fill.fees,
            levels_consumed=fill.levels_consumed,
        )


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """A sized basket whose settlement value exceeds its executable cost."""

    kind: str
    description: str
    legs: tuple[ArbitrageLeg, ...]
    shares: float
    capital_required: float
    guaranteed_payout: float
    fees: float
    profit: float
    verified_exhaustive: bool
    event_slug: str = ""
    requires_complete_set_mint: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def roi(self) -> float:
        """Profit per dollar of capital committed until settlement."""
        return self.profit / self.capital_required if self.capital_required > 0 else 0.0

    @property
    def profit_per_share(self) -> float:
        return self.profit / self.shares if self.shares > 0 else 0.0

    def to_row(self) -> dict[str, Any]:
        """Flatten for CSV/report output."""
        return {
            "kind": self.kind,
            "description": self.description,
            "event_slug": self.event_slug,
            "n_legs": len(self.legs),
            "shares": self.shares,
            "capital_required": self.capital_required,
            "guaranteed_payout": self.guaranteed_payout,
            "fees": self.fees,
            "profit": self.profit,
            "profit_per_share": self.profit_per_share,
            "roi": self.roi,
            "verified_exhaustive": self.verified_exhaustive,
            "requires_complete_set_mint": self.requires_complete_set_mint,
            "notes": "; ".join(self.notes),
        }


@dataclass(frozen=True)
class _Leg:
    """Scanner-internal description of a basket leg before it is executed."""

    market: Market
    token_id: str
    outcome_name: str
    side: str


def scan_binary_complement(
    market: Market,
    books: Mapping[str, OrderBook],
    fee_bps: float = 0.0,
    max_capital: float = 1_000.0,
    min_profit: float = 0.0,
    min_shares: float = 1.0,
) -> ArbitrageOpportunity | None:
    """Buy both sides of a binary market when the pair costs under a dollar.

    One YES share plus one NO share settles at exactly one dollar whichever way
    the market resolves, so any executable pair price below one is locked profit.
    """
    yes, no = market.yes_outcome, market.no_outcome
    if yes is None or no is None:
        return None
    legs = (
        _Leg(market, yes.token_id, yes.name, BUY),
        _Leg(market, no.token_id, no.name, BUY),
    )
    return _build_opportunity(
        kind=BINARY_COMPLEMENT,
        description=f"Buy YES+NO: {market.question}",
        legs=legs,
        books=books,
        payout_per_unit=1.0,
        fixed_cost_per_unit=0.0,
        fee_bps=fee_bps,
        max_capital=max_capital,
        min_profit=min_profit,
        min_shares=min_shares,
        verified_exhaustive=True,
        event_slug=market.event_slug or market.slug,
    )


def scan_binary_mint_and_sell(
    market: Market,
    books: Mapping[str, OrderBook],
    fee_bps: float = 0.0,
    max_capital: float = 1_000.0,
    min_profit: float = 0.0,
    min_shares: float = 1.0,
) -> ArbitrageOpportunity | None:
    """Mint a complete set for a dollar and sell both sides above it.

    Splitting one USDC of collateral yields one YES and one NO share. When the
    two bids together clear a dollar after fees the profit is realized
    immediately and nothing is held to settlement -- but it requires the
    on-chain split, so it is only actionable for a wallet able to call it.
    """
    yes, no = market.yes_outcome, market.no_outcome
    if yes is None or no is None:
        return None
    legs = (
        _Leg(market, yes.token_id, yes.name, SELL),
        _Leg(market, no.token_id, no.name, SELL),
    )
    return _build_opportunity(
        kind=BINARY_MINT_AND_SELL,
        description=f"Mint set and sell YES+NO: {market.question}",
        legs=legs,
        books=books,
        payout_per_unit=0.0,
        fixed_cost_per_unit=1.0,
        fee_bps=fee_bps,
        max_capital=max_capital,
        min_profit=min_profit,
        min_shares=min_shares,
        verified_exhaustive=True,
        event_slug=market.event_slug or market.slug,
        requires_complete_set_mint=True,
        notes=("Requires the on-chain complete-set split; profit realizes immediately.",),
    )


def scan_group_dutch_book(
    group: MarketGroup,
    books: Mapping[str, OrderBook],
    fee_bps: float = 0.0,
    max_capital: float = 1_000.0,
    min_profit: float = 0.0,
    min_shares: float = 1.0,
    require_verified: bool = True,
) -> ArbitrageOpportunity | None:
    """Buy YES on every outcome of an exhaustive group for under a dollar.

    Exactly one member of a partition resolves YES, so the basket settles at one
    dollar. ``require_verified`` refuses groups the venue has not marked
    ``negRisk``, because an incomplete outcome list makes the identity false.
    """
    if group.size < 2 or (require_verified and not group.exhaustive):
        return None
    legs = []
    for market in group.markets:
        yes = market.yes_outcome
        if yes is None:
            return None
        legs.append(_Leg(market, yes.token_id, yes.name, BUY))
    notes = () if group.exhaustive else (
        "Outcome set is not venue-verified as exhaustive; the payout is not guaranteed.",
    )
    return _build_opportunity(
        kind=GROUP_DUTCH_BOOK,
        description=f"Buy YES across {group.size} outcomes of {group.event_slug or group.event_id}",
        legs=tuple(legs),
        books=books,
        payout_per_unit=1.0,
        fixed_cost_per_unit=0.0,
        fee_bps=fee_bps,
        max_capital=max_capital,
        min_profit=min_profit,
        min_shares=min_shares,
        verified_exhaustive=group.exhaustive,
        event_slug=group.event_slug,
        notes=notes,
    )


def scan_neg_risk_no_basket(
    group: MarketGroup,
    books: Mapping[str, OrderBook],
    fee_bps: float = 0.0,
    max_capital: float = 1_000.0,
    min_profit: float = 0.0,
    min_shares: float = 1.0,
    require_verified: bool = True,
) -> ArbitrageOpportunity | None:
    """Buy NO on every outcome of an exhaustive group of size ``n``.

    Exactly one outcome resolves YES, so ``n - 1`` of the NO shares pay a dollar
    each. The basket is therefore worth ``n - 1`` and is mispriced whenever the
    NO quotes sum below that.
    """
    if group.size < 2 or (require_verified and not group.exhaustive):
        return None
    legs = []
    for market in group.markets:
        no = market.no_outcome
        if no is None:
            return None
        legs.append(_Leg(market, no.token_id, no.name, BUY))
    notes = () if group.exhaustive else (
        "Outcome set is not venue-verified as exhaustive; the payout is not guaranteed.",
    )
    return _build_opportunity(
        kind=NEG_RISK_NO_BASKET,
        description=f"Buy NO across {group.size} outcomes of {group.event_slug or group.event_id}",
        legs=tuple(legs),
        books=books,
        payout_per_unit=float(group.size - 1),
        fixed_cost_per_unit=0.0,
        fee_bps=fee_bps,
        max_capital=max_capital,
        min_profit=min_profit,
        min_shares=min_shares,
        verified_exhaustive=group.exhaustive,
        event_slug=group.event_slug,
        notes=notes,
    )


def scan_markets(
    markets: Sequence[Market],
    books: Mapping[str, OrderBook],
    groups: Sequence[MarketGroup] = (),
    fee_bps: float = 0.0,
    max_capital: float = 1_000.0,
    min_profit: float = 1.0,
    min_shares: float = 1.0,
    include_mint_and_sell: bool = True,
    require_verified_groups: bool = True,
) -> list[ArbitrageOpportunity]:
    """Run every scanner and return opportunities ranked by profit."""
    found: list[ArbitrageOpportunity] = []
    for market in markets:
        if market.closed or not market.is_binary:
            continue
        candidates = [
            scan_binary_complement(market, books, fee_bps, max_capital, min_profit, min_shares)
        ]
        if include_mint_and_sell:
            candidates.append(
                scan_binary_mint_and_sell(market, books, fee_bps, max_capital, min_profit, min_shares)
            )
        found.extend(item for item in candidates if item is not None)

    for group in groups:
        for scanner in (scan_group_dutch_book, scan_neg_risk_no_basket):
            opportunity = scanner(
                group,
                books,
                fee_bps,
                max_capital,
                min_profit,
                min_shares,
                require_verified_groups,
            )
            if opportunity is not None:
                found.append(opportunity)

    return sorted(found, key=lambda item: item.profit, reverse=True)


def opportunities_frame(opportunities: Iterable[ArbitrageOpportunity]) -> pd.DataFrame:
    """Tabulate opportunities for CSV output and reporting."""
    rows = [item.to_row() for item in opportunities]
    if not rows:
        return pd.DataFrame(
            columns=[
                "kind", "description", "event_slug", "n_legs", "shares",
                "capital_required", "guaranteed_payout", "fees", "profit",
                "profit_per_share", "roi", "verified_exhaustive",
                "requires_complete_set_mint", "notes",
            ]
        )
    return pd.DataFrame(rows).sort_values("profit", ascending=False).reset_index(drop=True)


def _build_opportunity(
    kind: str,
    description: str,
    legs: tuple[_Leg, ...],
    books: Mapping[str, OrderBook],
    payout_per_unit: float,
    fixed_cost_per_unit: float,
    fee_bps: float,
    max_capital: float,
    min_profit: float,
    min_shares: float,
    verified_exhaustive: bool,
    event_slug: str = "",
    requires_complete_set_mint: bool = False,
    notes: tuple[str, ...] = (),
) -> ArbitrageOpportunity | None:
    """Size a basket against the books and return it only if it clears filters."""
    leg_books = [books.get(leg.token_id) for leg in legs]
    if any(book is None for book in leg_books):
        return None
    pairs = [(book, leg.side) for book, leg in zip(leg_books, legs) if book is not None]

    shares = _basket_capacity(
        pairs,
        payout_per_unit=payout_per_unit,
        fixed_cost_per_unit=fixed_cost_per_unit,
        fee_bps=fee_bps,
    )
    shares = _cap_by_capital(
        pairs,
        shares,
        fee_bps=fee_bps,
        fixed_cost_per_unit=fixed_cost_per_unit,
        max_capital=max_capital,
    )
    if shares < min_shares:
        return None

    fills = [book.walk(leg.side, shares, fee_bps) for book, leg in zip(leg_books, legs)]  # type: ignore[union-attr]
    if any(not fill.complete for fill in fills):
        return None

    cash_out = sum(fill.cost for fill in fills) + fixed_cost_per_unit * shares
    payout = payout_per_unit * shares
    profit = payout - cash_out
    if profit < min_profit:
        return None

    # Capital is the peak outlay, not the net one: a basket that sells legs is
    # still funded in full before those proceeds arrive, so netting the inflows
    # here would report an unbounded return on a self-funding trade.
    capital = fixed_cost_per_unit * shares + sum(
        fill.cost for fill in fills if fill.side == BUY
    )
    if capital <= _TOLERANCE:
        return None

    executed = tuple(
        ArbitrageLeg.from_fill(leg.market, leg.token_id, leg.outcome_name, fill)
        for leg, fill in zip(legs, fills)
    )
    return ArbitrageOpportunity(
        kind=kind,
        description=description,
        legs=executed,
        shares=shares,
        capital_required=capital,
        guaranteed_payout=payout,
        fees=float(sum(fill.fees for fill in fills)),
        profit=profit,
        verified_exhaustive=verified_exhaustive,
        event_slug=event_slug,
        requires_complete_set_mint=requires_complete_set_mint,
        notes=notes,
    )


def _basket_capacity(
    legs: Sequence[tuple[OrderBook, str]],
    payout_per_unit: float,
    fixed_cost_per_unit: float,
    fee_bps: float,
) -> float:
    """Largest basket size whose *marginal* unit is still profitable.

    Marginal cost is a non-decreasing step function of size -- each leg walks
    into progressively worse levels -- so the profit-maximizing size is the
    point where marginal cost first reaches the guaranteed payout. The
    breakpoints of that step function are the cumulative depths of every leg,
    which makes the search exact rather than a numerical approximation.
    """
    breakpoints = {0.0}
    for book, side in legs:
        cumulative = 0.0
        for level in (book.asks if side == BUY else book.bids):
            cumulative += level.size
            breakpoints.add(cumulative)
    ordered = sorted(breakpoints)

    capacity = 0.0
    for lower, upper in zip(ordered, ordered[1:]):
        if upper <= lower + _TOLERANCE:
            continue
        probe = (lower + upper) / 2.0
        marginal = fixed_cost_per_unit
        for book, side in legs:
            price = _marginal_price(book, side, probe, fee_bps)
            if price is None:
                return capacity
            marginal += price
        if marginal >= payout_per_unit - _TOLERANCE:
            break
        capacity = upper
    return capacity


def _cap_by_capital(
    legs: Sequence[tuple[OrderBook, str]],
    shares: float,
    fee_bps: float,
    fixed_cost_per_unit: float,
    max_capital: float,
) -> float:
    """Shrink the basket until its cash outlay fits the capital budget.

    Cost is monotone in size, so a bisection converges; the result is rounded
    down to stay inside the budget rather than straddling it.
    """
    if shares <= 0 or max_capital <= 0:
        return 0.0
    if _basket_cost(legs, shares, fee_bps, fixed_cost_per_unit) <= max_capital:
        return shares
    low, high = 0.0, shares
    for _ in range(60):
        mid = (low + high) / 2.0
        if _basket_cost(legs, mid, fee_bps, fixed_cost_per_unit) <= max_capital:
            low = mid
        else:
            high = mid
    return low


def _basket_cost(
    legs: Sequence[tuple[OrderBook, str]],
    shares: float,
    fee_bps: float,
    fixed_cost_per_unit: float,
) -> float:
    """Net cash out for ``shares`` baskets, including fees and any mint."""
    total = fixed_cost_per_unit * shares
    for book, side in legs:
        fill = book.walk(side, shares, fee_bps)
        if not fill.complete:
            return float("inf")
        total += fill.cost
    return total


def _marginal_price(
    book: OrderBook,
    side: str,
    depth: float,
    fee_bps: float,
) -> float | None:
    """Fee-inclusive cost of the share sitting at cumulative ``depth``.

    A buy contributes its ask plus fee; a sell contributes minus its bid net of
    fee, so both directions add into a single marginal-cost figure. ``None``
    means the book is exhausted at that depth.
    """
    from .pricing import taker_fee_per_share

    levels = book.asks if side == BUY else book.bids
    cumulative = 0.0
    for level in levels:
        cumulative += level.size
        if depth < cumulative - _TOLERANCE:
            fee = taker_fee_per_share(level.price, fee_bps)
            return level.price + fee if side == BUY else -(level.price - fee)
    return None
