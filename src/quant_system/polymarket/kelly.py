"""Bankroll sizing for binary contracts under edge uncertainty.

For a contract costing ``c`` all-in that pays one dollar, a forecast ``q``
implies net odds ``b = (1 - c) / c`` and the growth-optimal stake is the
classical Kelly fraction, which simplifies to::

    f* = (q - c) / (1 - c)

Full Kelly is not a sensible live setting. It is optimal only when ``q`` is
known exactly, and in prediction markets ``q`` is an estimate whose error is
correlated with the very quotes that look mispriced. Every entry point here
therefore composes three defences: shrinkage of the forecast toward the market
price, a fractional-Kelly multiplier, and hard caps on per-market and aggregate
exposure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .pricing import (
    buy_cost_per_share,
    shrink_probability,
    validate_price,
    validate_probability,
)


@dataclass(frozen=True)
class PositionSize:
    """A sized order together with the inputs that justified it."""

    bankroll: float
    probability: float
    effective_price: float
    edge: float
    full_kelly: float
    applied_fraction: float
    notional: float
    shares: float
    binding_constraint: str

    @property
    def expected_profit(self) -> float:
        """Expected dollar profit at the stated forecast."""
        return self.edge * self.shares


def kelly_fraction(probability: float, cost: float) -> float:
    """Growth-optimal bankroll fraction for a one-dollar binary payoff.

    ``cost`` is the all-in price per share including fees and slippage, not the
    quoted price. Returns 0 when the bet has no edge; the complementary side
    should be sized separately rather than by taking a negative stake here.
    """
    q = validate_probability(probability)
    c = validate_price(cost, name="cost")
    if q <= c:
        return 0.0
    return (q - c) / (1.0 - c)


def expected_log_growth(fraction: float, probability: float, cost: float) -> float:
    """Expected log wealth multiple from staking ``fraction`` of bankroll.

    Exposed mainly so the optimality of :func:`kelly_fraction` is testable
    rather than asserted.
    """
    f = float(fraction)
    q = validate_probability(probability)
    c = validate_price(cost, name="cost")
    if not 0.0 <= f < 1.0:
        raise ValueError("fraction must lie in [0, 1)")
    win = 1.0 - f + f / c
    lose = 1.0 - f
    if win <= 0 or lose <= 0:
        return -math.inf
    return q * math.log(win) + (1.0 - q) * math.log(lose)


def size_position(
    bankroll: float,
    probability: float,
    effective_price: float,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.02,
    min_edge: float = 0.02,
    shrinkage: float = 0.0,
    market_price: float | None = None,
    fee_bps: float = 0.0,
    include_fee: bool = False,
) -> PositionSize:
    """Size one position, reporting which constraint actually bound.

    ``effective_price`` should already reflect the depth-weighted fill and, if
    ``include_fee`` is False, the fee as well; pass ``include_fee=True`` to have
    the configured fee added here instead. ``shrinkage`` pulls the forecast
    toward ``market_price`` (defaulting to the effective price) before any edge
    is computed, so an over-confident model cannot bypass the gate.
    """
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    if not 0.0 < kelly_multiplier <= 1.0:
        raise ValueError("kelly_multiplier must lie in (0, 1]")
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must lie in (0, 1]")
    if min_edge < 0:
        raise ValueError("min_edge must be non-negative")

    price = validate_price(effective_price, name="effective_price")
    cost = buy_cost_per_share(price, fee_bps) if include_fee else price
    reference = validate_price(market_price) if market_price is not None else price
    q = shrink_probability(probability, reference, shrinkage) if shrinkage > 0 else (
        validate_probability(probability)
    )
    edge = q - cost

    if edge < min_edge:
        return PositionSize(
            bankroll=float(bankroll),
            probability=q,
            effective_price=cost,
            edge=edge,
            full_kelly=0.0,
            applied_fraction=0.0,
            notional=0.0,
            shares=0.0,
            binding_constraint="min_edge",
        )

    full = kelly_fraction(q, cost)
    scaled = full * kelly_multiplier
    applied = min(scaled, max_fraction)
    constraint = "max_fraction" if applied < scaled - 1e-12 else "kelly"
    notional = applied * float(bankroll)
    return PositionSize(
        bankroll=float(bankroll),
        probability=q,
        effective_price=cost,
        edge=edge,
        full_kelly=full,
        applied_fraction=applied,
        notional=notional,
        shares=notional / cost if cost > 0 else 0.0,
        binding_constraint=constraint,
    )


def allocate_exposure(
    fractions: Mapping[str, float] | Sequence[tuple[str, float]],
    max_total_exposure: float = 0.20,
) -> dict[str, float]:
    """Scale simultaneous stakes down to an aggregate exposure budget.

    Kelly is derived one bet at a time. Summing independently sized positions
    across dozens of live markets silently levers the bankroll, so the requested
    fractions are rescaled proportionally whenever they breach the budget.
    Correlated markets are *not* detected here -- see
    :func:`quant_system.polymarket.execution.deduplicate_by_event`.
    """
    if max_total_exposure <= 0:
        raise ValueError("max_total_exposure must be positive")
    items = dict(fractions)
    if any(value < 0 for value in items.values()):
        raise ValueError("fractions must be non-negative")
    total = sum(items.values())
    if total <= max_total_exposure or total <= 0:
        return dict(items)
    scale = max_total_exposure / total
    return {key: value * scale for key, value in items.items()}


def drawdown_scaled_bankroll(
    bankroll: float,
    peak_bankroll: float,
    max_drawdown: float = 0.25,
    floor_fraction: float = 0.25,
) -> float:
    """Taper the bankroll used for sizing as drawdown deepens.

    A model whose edge was real should recover; a model whose edge was an
    artifact should not be allowed to keep compounding the mistake at full size.
    Sizing capital falls linearly to ``floor_fraction`` of the current bankroll
    as the drawdown approaches ``max_drawdown``.
    """
    if bankroll <= 0 or peak_bankroll <= 0:
        return 0.0
    if not 0.0 < max_drawdown < 1.0:
        raise ValueError("max_drawdown must lie in (0, 1)")
    if not 0.0 < floor_fraction <= 1.0:
        raise ValueError("floor_fraction must lie in (0, 1]")
    drawdown = max(0.0, 1.0 - bankroll / peak_bankroll)
    if drawdown <= 0:
        return float(bankroll)
    severity = min(drawdown / max_drawdown, 1.0)
    taper = 1.0 - (1.0 - floor_fraction) * severity
    return float(bankroll) * taper
