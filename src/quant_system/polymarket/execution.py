"""Risk-gated order planning and a paper broker.

This module turns forecasts into a concrete, auditable list of orders and stops
there. Nothing here signs a transaction, holds a key, or talks to a venue: the
output is a plan a human or a separate, deliberately configured adapter can act
on. That boundary is intentional. Order signing needs custody of a funded
wallet and cannot be exercised by this repository's tests, so shipping it
untested next to code that decides *what* to trade would be the riskiest part
of the system and the least verified.

Sizing is limit-price driven rather than mid driven. For a forecast ``q`` and a
required edge ``m``, the worst acceptable all-in price is ``q - m``; the book is
then asked how many shares are available at or better than that limit, and the
order is the smaller of that depth and the Kelly stake. An order built this way
cannot be filled at a price that destroys the edge that justified it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .arbitrage import ArbitrageOpportunity
from .client import build_market_url
from .kelly import allocate_exposure, size_position
from .markets import Market
from .orderbook import BUY, OrderBook
from .pricing import MAX_PRICE, MIN_PRICE, taker_fee_per_share


@dataclass(frozen=True)
class RiskLimits:
    """Hard gates applied before any order reaches a plan."""

    kelly_multiplier: float = 0.25
    max_fraction_per_market: float = 0.02
    max_total_exposure: float = 0.20
    min_edge: float = 0.03
    shrinkage: float = 0.35
    min_notional: float = 5.0
    max_spread: float = 0.10
    min_book_depth_usd: float = 100.0
    min_days_to_resolution: float = 0.25
    max_days_to_resolution: float = 365.0
    max_orders: int = 25
    one_order_per_event: bool = True

    def __post_init__(self) -> None:
        if self.min_days_to_resolution < 0 or self.max_days_to_resolution <= 0:
            raise ValueError("resolution windows must be non-negative and positive")
        if self.min_days_to_resolution >= self.max_days_to_resolution:
            raise ValueError("min_days_to_resolution must be below max_days_to_resolution")


@dataclass(frozen=True)
class OrderIntent:
    """A single proposed taker order with the reasoning that produced it."""

    market_id: str
    question: str
    event_id: str
    token_id: str
    side: str
    action: str
    shares: float
    limit_price: float
    effective_price: float
    notional: float
    probability: float
    edge: float
    bankroll_fraction: float
    market_url: str = ""
    rationale: str = ""

    @property
    def expected_profit(self) -> float:
        return self.edge * self.shares

    def to_row(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "event_id": self.event_id,
            "token_id": self.token_id,
            "side": self.side,
            "action": self.action,
            "shares": self.shares,
            "limit_price": self.limit_price,
            "effective_price": self.effective_price,
            "notional": self.notional,
            "probability": self.probability,
            "edge": self.edge,
            "bankroll_fraction": self.bankroll_fraction,
            "expected_profit": self.expected_profit,
            "market_url": self.market_url,
            "rationale": self.rationale,
        }


@dataclass
class OrderPlan:
    """A dry-run set of orders plus the audit trail of what was rejected."""

    bankroll: float
    orders: list[OrderIntent] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    generated_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: tuple[str, ...] = ()

    @property
    def total_notional(self) -> float:
        return float(sum(order.notional for order in self.orders))

    @property
    def total_expected_profit(self) -> float:
        return float(sum(order.expected_profit for order in self.orders))

    def to_frame(self) -> pd.DataFrame:
        rows = [order.to_row() for order in self.orders]
        if not rows:
            return pd.DataFrame(columns=list(OrderIntent.__annotations__) + ["expected_profit"])
        return pd.DataFrame(rows).sort_values("expected_profit", ascending=False).reset_index(
            drop=True
        )

    def to_json(self, path: str | Path) -> Path:
        """Persist the plan, including rejections, for later audit."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "generated_utc": self.generated_utc,
                    "bankroll": self.bankroll,
                    "total_notional": self.total_notional,
                    "total_expected_profit": self.total_expected_profit,
                    "notes": list(self.notes),
                    "orders": [order.to_row() for order in self.orders],
                    "skipped": self.skipped,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return target


def build_order_plan(
    markets: Sequence[Market],
    books: Mapping[str, OrderBook],
    probabilities: Mapping[str, float],
    bankroll: float,
    limits: RiskLimits | None = None,
    fee_bps: float = 0.0,
    as_of: datetime | None = None,
) -> OrderPlan:
    """Turn per-market forecasts into a risk-gated, exposure-capped order plan.

    ``probabilities`` maps ``market_id`` to the forecast probability that the
    market's YES outcome occurs. Markets absent from the mapping are skipped
    rather than traded at the market's own price.
    """
    rules = limits or RiskLimits()
    now = as_of or datetime.now(timezone.utc)
    plan = OrderPlan(bankroll=float(bankroll))
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")

    candidates: list[OrderIntent] = []
    for market in markets:
        probability = probabilities.get(market.market_id)
        reason = _rejection_reason(market, books, probability, rules, now)
        if reason:
            plan.skipped.append({"market_id": market.market_id, "reason": reason})
            continue

        best = _best_side(market, books, float(probability), rules, bankroll, fee_bps)
        if best is None:
            plan.skipped.append({"market_id": market.market_id, "reason": "no side cleared min_edge"})
            continue
        candidates.append(best)

    if rules.one_order_per_event:
        candidates = deduplicate_by_event(candidates)

    candidates.sort(key=lambda order: order.expected_profit, reverse=True)
    candidates = candidates[: rules.max_orders]

    scaled = allocate_exposure(
        {order.market_id: order.bankroll_fraction for order in candidates},
        max_total_exposure=rules.max_total_exposure,
    )
    for order in candidates:
        fraction = scaled[order.market_id]
        notional = fraction * float(bankroll)
        if notional < rules.min_notional:
            plan.skipped.append(
                {"market_id": order.market_id, "reason": "below min_notional after exposure cap"}
            )
            continue
        plan.orders.append(_rescale(order, notional, fraction))

    plan.notes = (
        "Dry run: no order is transmitted. Prices are snapshot prices and can move.",
        f"Fee model: {fee_bps:.1f} bps of min(price, 1 - price) per share.",
        "Limit prices already embed the required edge; do not raise them when submitting.",
    )
    return plan


def deduplicate_by_event(orders: Iterable[OrderIntent]) -> list[OrderIntent]:
    """Keep only the strongest order per event.

    Markets under one event describe the same underlying uncertainty, so
    independent Kelly stakes across them are a single concentrated bet wearing
    several names. Where an event id is unavailable the market stands alone.
    """
    best: dict[str, OrderIntent] = {}
    standalone: list[OrderIntent] = []
    for order in orders:
        if not order.event_id:
            standalone.append(order)
            continue
        current = best.get(order.event_id)
        if current is None or order.expected_profit > current.expected_profit:
            best[order.event_id] = order
    return list(best.values()) + standalone


def arbitrage_to_orders(opportunity: ArbitrageOpportunity) -> list[OrderIntent]:
    """Expand a basket opportunity into its individual legs.

    The legs are not atomic on-venue, so they carry the basket's identity in
    ``rationale`` to make partial execution visible in the audit trail.
    """
    orders: list[OrderIntent] = []
    for leg in opportunity.legs:
        orders.append(
            OrderIntent(
                market_id=leg.market_id,
                question=leg.question,
                event_id=opportunity.event_slug,
                token_id=leg.token_id,
                side=leg.outcome_name.lower(),
                action=leg.side,
                shares=leg.shares,
                limit_price=float(min(max(leg.average_price, MIN_PRICE), MAX_PRICE)),
                effective_price=leg.effective_price,
                notional=leg.notional,
                probability=float("nan"),
                edge=opportunity.profit_per_share / max(len(opportunity.legs), 1),
                bankroll_fraction=float("nan"),
                rationale=(
                    f"{opportunity.kind} leg {leg.outcome_name}; basket profit "
                    f"{opportunity.profit:.2f} on {opportunity.capital_required:.2f} capital"
                ),
            )
        )
    return orders


class ExecutionAdapter(Protocol):
    """Interface a live broker would implement; only paper trading ships here."""

    def submit(self, plan: OrderPlan) -> pd.DataFrame: ...

    def settle(self, market_id: str, outcome: int) -> float: ...


@dataclass
class PaperBroker:
    """A cash-and-position ledger that executes plans without a venue.

    Positions are held at cost and settled at 0 or 1, so the ledger reproduces
    the same accounting the backtester uses and a paper run can be compared
    against a simulated one directly.
    """

    bankroll: float = 10_000.0
    fee_bps: float = 0.0
    fills: list[dict[str, Any]] = field(default_factory=list)
    positions: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    @property
    def cash(self) -> float:
        return self.bankroll - sum(item["capital"] for item in self.positions.values())

    def submit(self, plan: OrderPlan) -> pd.DataFrame:
        """Record every order in a plan that the available cash can fund."""
        for order in plan.orders:
            cost = order.notional
            if cost > self.cash + 1e-9:
                self.fills.append({**order.to_row(), "status": "rejected_insufficient_cash"})
                continue
            key = (order.market_id, order.side)
            position = self.positions.setdefault(key, {"shares": 0.0, "capital": 0.0})
            position["shares"] += order.shares
            position["capital"] += cost
            self.fills.append(
                {
                    **order.to_row(),
                    "status": "filled",
                    "filled_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        return self.ledger()

    def settle(self, market_id: str, outcome: int) -> float:
        """Settle every open side of ``market_id`` and return realized profit."""
        if int(outcome) not in {0, 1}:
            raise ValueError("outcome must be 0 or 1")
        realized = 0.0
        for (held_market, side) in [key for key in self.positions if key[0] == market_id]:
            position = self.positions.pop((held_market, side))
            payoff = outcome if side == "yes" else 1 - outcome
            proceeds = position["shares"] * float(payoff)
            profit = proceeds - position["capital"]
            realized += profit
            self.bankroll += profit
            self.fills.append(
                {
                    "market_id": market_id,
                    "side": side,
                    "status": "settled",
                    "shares": position["shares"],
                    "capital": position["capital"],
                    "payoff": float(payoff),
                    "profit": profit,
                    "bankroll_after": self.bankroll,
                }
            )
        return realized

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self.fills) if self.fills else pd.DataFrame(columns=["status"])

    def state(self) -> dict[str, float]:
        return {
            "bankroll": float(self.bankroll),
            "cash": float(self.cash),
            "open_positions": float(len(self.positions)),
            "committed_capital": float(
                sum(item["capital"] for item in self.positions.values())
            ),
        }


def _rejection_reason(
    market: Market,
    books: Mapping[str, OrderBook],
    probability: float | None,
    rules: RiskLimits,
    now: datetime,
) -> str:
    """First gate the market fails, or an empty string when it passes."""
    if probability is None:
        return "no forecast"
    if market.closed or not market.active:
        return "market is not tradable"
    if not market.is_binary:
        return "not a binary market"
    yes, no = market.yes_outcome, market.no_outcome
    if yes is None or no is None:
        return "missing outcome tokens"
    yes_book, no_book = books.get(yes.token_id), books.get(no.token_id)
    if yes_book is None or no_book is None:
        return "missing order book"
    if yes_book.best_ask is None or no_book.best_ask is None:
        return "one-sided book"

    spread = yes_book.spread
    if spread is not None and spread > rules.max_spread:
        return f"spread {spread:.3f} above max_spread"
    depth = yes_book.notional_depth("sell") + no_book.notional_depth("sell")
    if depth < rules.min_book_depth_usd:
        return f"ask depth {depth:.0f} below min_book_depth_usd"

    days = market.days_to_resolution(now)
    if days is not None:
        if days < rules.min_days_to_resolution:
            return "resolves too soon"
        if days > rules.max_days_to_resolution:
            return "resolves too far out"
    return ""


def _best_side(
    market: Market,
    books: Mapping[str, OrderBook],
    probability: float,
    rules: RiskLimits,
    bankroll: float,
    fee_bps: float,
) -> OrderIntent | None:
    """Price both sides through the book and keep the better one, if any."""
    yes, no = market.yes_outcome, market.no_outcome
    best: OrderIntent | None = None
    sides = (
        ("yes", yes, probability),
        ("no", no, 1.0 - probability),
    )
    for side, outcome, side_probability in sides:
        if outcome is None:
            continue
        book = books.get(outcome.token_id)
        if book is None or book.best_ask is None:
            continue

        reference = book.mid if book.mid is not None else book.best_ask
        sized = size_position(
            bankroll=bankroll,
            probability=side_probability,
            effective_price=book.best_ask,
            kelly_multiplier=rules.kelly_multiplier,
            max_fraction=rules.max_fraction_per_market,
            min_edge=rules.min_edge,
            shrinkage=rules.shrinkage,
            market_price=reference,
            fee_bps=fee_bps,
            include_fee=True,
        )
        if sized.notional <= 0:
            continue

        # The limit is the worst price that still leaves the required edge,
        # net of the fee charged at that price.
        raw_limit = sized.probability - rules.min_edge
        limit = raw_limit - taker_fee_per_share(
            min(max(raw_limit, MIN_PRICE), MAX_PRICE), fee_bps
        )
        limit = float(min(max(limit, MIN_PRICE), MAX_PRICE))
        available = book.executable_shares(BUY, limit)
        shares = min(sized.shares, available)
        if shares <= 0:
            continue
        fill = book.walk(BUY, shares, fee_bps, limit_price=limit)
        if fill.shares <= 0:
            continue

        effective = fill.effective_price
        edge = sized.probability - effective
        if edge < rules.min_edge:
            continue
        notional = fill.shares * effective
        if notional < rules.min_notional:
            continue

        intent = OrderIntent(
            market_id=market.market_id,
            question=market.question,
            event_id=market.event_id,
            token_id=outcome.token_id,
            side=side,
            action=BUY,
            shares=fill.shares,
            limit_price=limit,
            effective_price=effective,
            notional=notional,
            probability=sized.probability,
            edge=edge,
            bankroll_fraction=notional / bankroll,
            market_url=build_market_url(market.slug) if market.slug else "",
            rationale=(
                f"forecast {sized.probability:.3f} vs all-in {effective:.3f}; "
                f"Kelly {sized.full_kelly:.3f} x {rules.kelly_multiplier:.2f}, "
                f"bound by {sized.binding_constraint}"
            ),
        )
        if best is None or intent.expected_profit > best.expected_profit:
            best = intent
    return best


def _rescale(order: OrderIntent, notional: float, fraction: float) -> OrderIntent:
    """Re-size an order after the aggregate exposure cap is applied."""
    shares = notional / order.effective_price if order.effective_price > 0 else 0.0
    return OrderIntent(
        market_id=order.market_id,
        question=order.question,
        event_id=order.event_id,
        token_id=order.token_id,
        side=order.side,
        action=order.action,
        shares=shares,
        limit_price=order.limit_price,
        effective_price=order.effective_price,
        notional=notional,
        probability=order.probability,
        edge=order.edge,
        bankroll_fraction=fraction,
        market_url=order.market_url,
        rationale=order.rationale,
    )
