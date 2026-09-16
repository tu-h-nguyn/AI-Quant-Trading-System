"""Depth-aware execution simulation against a central limit order book.

Top-of-book quotes overstate what a strategy can actually transact. Prediction
markets are thin, so a nominal edge of two cents frequently disappears once an
order walks three levels of a book that holds a few hundred shares. Every
component that sizes a trade in this package prices it through
:meth:`OrderBook.walk`, never through the best quote alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .pricing import taker_fee_per_share

BUY = "buy"
SELL = "sell"


@dataclass(frozen=True)
class Level:
    """A single price level with its resting size, in shares."""

    price: float
    size: float

    def __post_init__(self) -> None:
        if not 0.0 < self.price < 1.0:
            raise ValueError(f"level price must lie in (0, 1), got {self.price}")
        if self.size < 0:
            raise ValueError(f"level size must be non-negative, got {self.size}")


@dataclass(frozen=True)
class Fill:
    """The outcome of walking an order book.

    ``shares`` is what was actually executed, which is below the request when
    the book is exhausted. ``average_price`` is the size-weighted execution
    price excluding fees; ``cost`` includes them.
    """

    side: str
    shares: float
    notional: float
    fees: float
    levels_consumed: int
    requested_shares: float

    @property
    def average_price(self) -> float:
        """Size-weighted execution price, or ``nan`` when nothing filled."""
        return self.notional / self.shares if self.shares > 0 else math.nan

    @property
    def cost(self) -> float:
        """Cash paid (buy) or, as a negative number, received (sell)."""
        return self.notional + self.fees if self.side == BUY else -(self.notional - self.fees)

    @property
    def effective_price(self) -> float:
        """Execution price including fees, the number that decides an edge."""
        if self.shares <= 0:
            return math.nan
        fee_per_share = self.fees / self.shares
        return self.average_price + fee_per_share if self.side == BUY else (
            self.average_price - fee_per_share
        )

    @property
    def complete(self) -> bool:
        """Whether the book satisfied the full requested size."""
        return self.shares >= self.requested_shares - 1e-9


@dataclass(frozen=True)
class OrderBook:
    """A normalized snapshot of one outcome token's book.

    ``bids`` are sorted best-first (descending price) and ``asks`` best-first
    (ascending price) regardless of the ordering used by the source payload.
    """

    token_id: str
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    timestamp: float | None = None

    @classmethod
    def from_levels(
        cls,
        token_id: str,
        bids: Iterable[Sequence[float] | Mapping[str, Any] | Level],
        asks: Iterable[Sequence[float] | Mapping[str, Any] | Level],
        timestamp: float | None = None,
    ) -> OrderBook:
        """Build a book from loosely typed level records.

        Accepts ``Level`` objects, ``(price, size)`` pairs, or the
        ``{"price": ..., "size": ...}`` mappings returned by the CLOB API.
        """
        return cls(
            token_id=str(token_id),
            bids=tuple(sorted(_coerce_levels(bids), key=lambda x: -x.price)),
            asks=tuple(sorted(_coerce_levels(asks), key=lambda x: x.price)),
            timestamp=timestamp,
        )

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        """Midpoint of the top of book, or ``None`` when a side is empty."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def depth(self, side: str, levels: int | None = None) -> float:
        """Total resting shares on ``side``, optionally over the top levels."""
        book = self._side(side)
        selected = book if levels is None else book[:levels]
        return float(sum(level.size for level in selected))

    def notional_depth(self, side: str, levels: int | None = None) -> float:
        """Total resting dollars on ``side``, optionally over the top levels."""
        book = self._side(side)
        selected = book if levels is None else book[:levels]
        return float(sum(level.size * level.price for level in selected))

    def imbalance(self, levels: int = 3) -> float:
        """Signed book imbalance in ``[-1, 1]``; positive means bid-heavy."""
        bid = self.depth(BUY, levels)
        ask = self.depth(SELL, levels)
        total = bid + ask
        return 0.0 if total <= 0 else (bid - ask) / total

    def walk(
        self,
        side: str,
        shares: float,
        fee_bps: float = 0.0,
        limit_price: float | None = None,
    ) -> Fill:
        """Execute ``shares`` against the book, consuming levels in order.

        A ``buy`` lifts the asks and a ``sell`` hits the bids. ``limit_price``
        stops the walk once the book is worse than the limit, which is how the
        caller expresses "only trade while the edge survives". The returned
        fill reports the size that was actually available, so callers must
        check :attr:`Fill.complete` rather than assuming the request was met.
        """
        normalized = _normalize_side(side)
        requested = float(shares)
        if requested < 0:
            raise ValueError("shares must be non-negative")
        book = self.asks if normalized == BUY else self.bids

        remaining = requested
        notional = 0.0
        fees = 0.0
        consumed = 0
        for level in book:
            if remaining <= 1e-12:
                break
            if limit_price is not None:
                if normalized == BUY and level.price > limit_price + 1e-12:
                    break
                if normalized == SELL and level.price < limit_price - 1e-12:
                    break
            taken = min(remaining, level.size)
            if taken <= 0:
                continue
            notional += taken * level.price
            fees += taken * taker_fee_per_share(level.price, fee_bps)
            remaining -= taken
            consumed += 1

        return Fill(
            side=normalized,
            shares=requested - remaining,
            notional=notional,
            fees=fees,
            levels_consumed=consumed,
            requested_shares=requested,
        )

    def executable_shares(self, side: str, limit_price: float) -> float:
        """Shares available at or better than ``limit_price``."""
        normalized = _normalize_side(side)
        book = self.asks if normalized == BUY else self.bids
        if normalized == BUY:
            return float(sum(lv.size for lv in book if lv.price <= limit_price + 1e-12))
        return float(sum(lv.size for lv in book if lv.price >= limit_price - 1e-12))

    def _side(self, side: str) -> tuple[Level, ...]:
        return self.bids if _normalize_side(side) == BUY else self.asks


def _normalize_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {BUY, SELL}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return normalized


def _coerce_levels(
    raw: Iterable[Sequence[float] | Mapping[str, Any] | Level],
) -> list[Level]:
    levels: list[Level] = []
    for item in raw:
        if isinstance(item, Level):
            levels.append(item)
        elif isinstance(item, Mapping):
            levels.append(Level(float(item["price"]), float(item["size"])))
        else:
            price, size = item
            levels.append(Level(float(price), float(size)))
    return [level for level in levels if level.size > 0]
