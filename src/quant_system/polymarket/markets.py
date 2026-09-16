"""Normalized market records parsed from Polymarket's public payloads.

The Gamma API returns several fields as JSON-encoded strings (``outcomes``,
``outcomePrices``, ``clobTokenIds``) and mixes numeric types between string and
float across endpoints and versions. Parsing is therefore defensive: anything
that cannot be interpreted becomes ``None`` rather than raising, so a single
malformed market never aborts a scan over thousands of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

YES = "yes"
NO = "no"


@dataclass(frozen=True)
class Outcome:
    """One tradable side of a market, identified by its CLOB token id."""

    name: str
    token_id: str
    price: float | None = None

    @property
    def is_yes(self) -> bool:
        return self.name.strip().lower() in {YES, "up", "true"}


@dataclass(frozen=True)
class Market:
    """A single resolvable market with two or more mutually exclusive outcomes."""

    market_id: str
    question: str
    slug: str = ""
    condition_id: str = ""
    outcomes: tuple[Outcome, ...] = ()
    end_date: datetime | None = None
    active: bool = True
    closed: bool = False
    neg_risk: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    event_id: str = ""
    event_slug: str = ""
    resolved_index: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def is_resolved(self) -> bool:
        return self.closed and self.resolved_index is not None

    @property
    def yes_outcome(self) -> Outcome | None:
        """The YES side of a binary market, by name and then by position."""
        if not self.is_binary:
            return None
        for outcome in self.outcomes:
            if outcome.is_yes:
                return outcome
        return self.outcomes[0]

    @property
    def no_outcome(self) -> Outcome | None:
        yes = self.yes_outcome
        if yes is None:
            return None
        return next((o for o in self.outcomes if o.token_id != yes.token_id), None)

    @property
    def yes_label(self) -> int | None:
        """Binary settlement label: 1 when the YES side resolved true."""
        if not self.is_resolved or not self.is_binary:
            return None
        yes = self.yes_outcome
        resolved = self.outcomes[self.resolved_index]  # type: ignore[index]
        return int(yes is not None and resolved.token_id == yes.token_id)

    def days_to_resolution(self, as_of: datetime) -> float | None:
        """Calendar days from ``as_of`` until resolution, floored at zero."""
        if self.end_date is None:
            return None
        reference = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        return max((self.end_date - reference).total_seconds() / 86_400.0, 0.0)

    def outcome_by_token(self, token_id: str) -> Outcome | None:
        return next((o for o in self.outcomes if o.token_id == str(token_id)), None)


@dataclass(frozen=True)
class MarketGroup:
    """Markets under one event that partition a single outcome space.

    Polymarket's ``negRisk`` events are the reliable case: the venue itself
    guarantees the member markets are mutually exclusive and exhaustive, which
    is what makes a basket price comparable to one dollar. Groups assembled by
    any other rule must be treated as unverified.
    """

    event_id: str
    event_slug: str
    markets: tuple[Market, ...]
    neg_risk: bool = False

    @property
    def size(self) -> int:
        return len(self.markets)

    @property
    def exhaustive(self) -> bool:
        """Whether the partition assumption is venue-guaranteed."""
        return self.neg_risk and self.size >= 2


def parse_market(payload: Mapping[str, Any]) -> Market | None:
    """Convert one Gamma ``/markets`` record into a :class:`Market`.

    Returns ``None`` when the record has no usable outcome/token structure.
    """
    names = _json_list(payload.get("outcomes"))
    token_ids = _json_list(payload.get("clobTokenIds"))
    prices = _json_list(payload.get("outcomePrices"))
    if not names or not token_ids or len(names) != len(token_ids):
        return None

    outcomes = tuple(
        Outcome(
            name=str(name),
            token_id=str(token_id),
            price=_float_or_none(prices[i]) if i < len(prices) else None,
        )
        for i, (name, token_id) in enumerate(zip(names, token_ids))
    )
    closed = bool(payload.get("closed", False))
    events = payload.get("events") or []
    event = events[0] if isinstance(events, list) and events else {}
    if not isinstance(event, Mapping):
        event = {}

    return Market(
        market_id=str(payload.get("id", "")),
        question=str(payload.get("question", "")),
        slug=str(payload.get("slug", "")),
        condition_id=str(payload.get("conditionId", "")),
        outcomes=outcomes,
        end_date=_parse_datetime(payload.get("endDate")),
        active=bool(payload.get("active", True)),
        closed=closed,
        neg_risk=bool(payload.get("negRisk", False) or event.get("negRisk", False)),
        volume=_float_or_none(payload.get("volumeNum", payload.get("volume"))) or 0.0,
        liquidity=_float_or_none(payload.get("liquidityNum", payload.get("liquidity"))) or 0.0,
        event_id=str(event.get("id", "")),
        event_slug=str(event.get("slug", "")),
        resolved_index=_resolved_index(outcomes, closed),
        raw=dict(payload),
    )


def parse_markets(payloads: Iterable[Mapping[str, Any]]) -> list[Market]:
    """Parse a batch of Gamma records, silently dropping unusable ones."""
    parsed = (parse_market(item) for item in payloads if isinstance(item, Mapping))
    return [market for market in parsed if market is not None]


def group_by_event(markets: Iterable[Market]) -> list[MarketGroup]:
    """Bucket markets into their parent events, largest group first."""
    buckets: dict[str, list[Market]] = {}
    for market in markets:
        if market.event_id:
            buckets.setdefault(market.event_id, []).append(market)
    groups = [
        MarketGroup(
            event_id=event_id,
            event_slug=members[0].event_slug,
            markets=tuple(members),
            neg_risk=all(m.neg_risk for m in members),
        )
        for event_id, members in buckets.items()
    ]
    return sorted(groups, key=lambda g: (-g.size, g.event_id))


def _resolved_index(outcomes: tuple[Outcome, ...], closed: bool) -> int | None:
    """Infer the winning outcome from settled prices on a closed market."""
    if not closed:
        return None
    priced = [(i, o.price) for i, o in enumerate(outcomes) if o.price is not None]
    winners = [i for i, price in priced if price is not None and price > 0.99]
    if len(winners) != 1 or len(priced) != len(outcomes):
        return None
    return winners[0]


def _json_list(value: Any) -> list[Any]:
    """Decode a field that may arrive as a list or a JSON-encoded string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        return list(decoded) if isinstance(decoded, (list, tuple)) else []
    return []


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
