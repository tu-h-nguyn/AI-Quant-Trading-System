"""Deterministic synthetic prediction markets.

These generators exist so the pipeline can be exercised and regression-tested
without network access, and so the arbitrage scanners can be checked against
baskets whose correct answer is known in advance.

They are explicitly **not** evidence about Polymarket. ``simulate_panel``
constructs a world in which an exploitable edge exists by assumption:
``signal_strength`` controls how much of each market's mispricing is legible
from an observable feature. A profitable backtest on this panel demonstrates
that sizing, frictions, settlement, and capital accounting are wired together
correctly. It says nothing about whether such a signal exists in the real
venue, which only the cached live-data path can address.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .markets import Market, MarketGroup, Outcome
from .orderbook import OrderBook


@dataclass(frozen=True)
class PanelSpec:
    """Parameters of the synthetic world."""

    n_markets: int = 400
    observations_per_market: int = 12
    horizon_days: int = 60
    mispricing_sigma: float = 0.35
    signal_strength: float = 0.60
    mispricing_decay: float = 0.5
    spread: float = 0.02
    min_depth: float = 200.0
    max_depth: float = 5_000.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_markets < 1 or self.observations_per_market < 1:
            raise ValueError("n_markets and observations_per_market must be positive")
        if not 0.0 <= self.signal_strength <= 1.0:
            raise ValueError("signal_strength must lie in [0, 1]")
        if not 0.0 <= self.mispricing_decay <= 1.0:
            raise ValueError("mispricing_decay must lie in [0, 1]")


def simulate_panel(spec: PanelSpec | None = None) -> pd.DataFrame:
    """Generate a long snapshot panel with a known, partly legible edge.

    Each market carries a latent probability ``q``. Its quoted price is ``q``
    perturbed in logit space, and ``signal_strength`` of that perturbation is
    reflected in an observable ``signal`` column that a model can learn from;
    the remainder is unlearnable noise. Mispricing shrinks as resolution
    approaches, mirroring the way real markets tighten near settlement.
    """
    settings = spec or PanelSpec()
    rng = np.random.default_rng(settings.seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")

    latent = rng.beta(2.0, 2.0, size=settings.n_markets)
    labels = (rng.uniform(size=settings.n_markets) < latent).astype(int)
    offsets = rng.integers(0, 120, size=settings.n_markets)

    rows: list[dict] = []
    for market in range(settings.n_markets):
        q = float(np.clip(latent[market], 0.02, 0.98))
        logit_q = float(np.log(q / (1.0 - q)))
        market_start = start + pd.Timedelta(days=int(offsets[market]))
        end_date = market_start + pd.Timedelta(days=settings.horizon_days)
        step = max(settings.horizon_days // settings.observations_per_market, 1)

        for observation in range(settings.observations_per_market):
            timestamp = market_start + pd.Timedelta(days=observation * step)
            progress = observation / max(settings.observations_per_market - 1, 1)
            decay = 1.0 - settings.mispricing_decay * progress

            distortion = rng.normal(0.0, settings.mispricing_sigma) * decay
            # The observable signal reveals `signal_strength` of the distortion;
            # a positive signal means the quote sits below the true probability.
            signal = -settings.signal_strength * distortion + rng.normal(
                0.0, max(1.0 - settings.signal_strength, 1e-6) * settings.mispricing_sigma
            )
            price = float(1.0 / (1.0 + np.exp(-(logit_q + distortion))))
            price = float(np.clip(price, 0.02, 0.98))

            half_spread = settings.spread / 2.0
            depth = float(rng.uniform(settings.min_depth, settings.max_depth))
            rows.append(
                {
                    "market_id": f"SIM-{market:05d}",
                    "question": f"Synthetic market {market}",
                    "timestamp": timestamp,
                    "price": price,
                    "best_bid": float(np.clip(price - half_spread, 0.01, 0.99)),
                    "best_ask": float(np.clip(price + half_spread, 0.01, 0.99)),
                    "yes_ask": float(np.clip(price + half_spread, 0.01, 0.99)),
                    "no_ask": float(np.clip(1.0 - price + half_spread, 0.01, 0.99)),
                    "bid_depth": depth,
                    "ask_depth": depth * float(rng.uniform(0.6, 1.6)),
                    "volume": float(rng.lognormal(9.0, 1.0)),
                    "liquidity": depth * price,
                    "end_date": end_date,
                    "signal": float(signal),
                    "true_probability": q,
                    "label": int(labels[market]),
                }
            )
    return pd.DataFrame(rows)

@dataclass(frozen=True)
class ArbitrageFixture:
    """A synthetic snapshot plus the ground truth about what it contains."""

    markets: list[Market]
    books: dict[str, OrderBook]
    groups: list[MarketGroup]
    planted: dict[str, tuple[str, ...]]

    def planted_ids(self, kind: str) -> tuple[str, ...]:
        return self.planted.get(kind, ())


def simulate_arbitrage_snapshot(
    n_binary: int = 20,
    n_groups: int = 4,
    group_size: int = 4,
    spread: float = 0.02,
    edge: float = 0.03,
    group_edge: float = 0.06,
    depth: float = 500.0,
    seed: int = 7,
) -> ArbitrageFixture:
    """Build a snapshot whose arbitrage content is known exactly.

    Quotes are generated from a fair value per token plus a symmetric spread,
    and the planted cases shift that fair value so exactly one identity breaks:

    ``binary_complement``    both asks shifted down, so YES+NO asks clear a dollar
    ``binary_mint_and_sell`` both fairs shifted up, so YES+NO bids clear a dollar
    ``group_dutch_book``     YES basket priced under a dollar
    ``neg_risk_no_basket``   NO basket priced under ``n - 1``

    Every other market is quoted consistently, so a scanner that reports a case
    outside :attr:`ArbitrageFixture.planted` has produced a false positive. The
    planted edges are sized to exceed the spread, otherwise the "arbitrage"
    would be an artifact of quoting rather than a real basket mispricing.
    """
    if 2 * edge <= spread:
        raise ValueError("binary edge must exceed half the spread to be genuine")
    if group_edge <= group_size * spread / 2.0:
        raise ValueError("group edge must exceed the accumulated basket spread")

    rng = np.random.default_rng(seed)
    markets: list[Market] = []
    books: dict[str, OrderBook] = {}
    groups: list[MarketGroup] = []
    planted: dict[str, list[str]] = {
        "binary_complement": [],
        "binary_mint_and_sell": [],
        "group_dutch_book": [],
        "neg_risk_no_basket": [],
    }

    for index in range(n_binary):
        market_id = f"BIN{index}"
        yes_token, no_token = f"{market_id}-Y", f"{market_id}-N"
        fair = float(rng.uniform(0.25, 0.75))
        if index % 5 == 0:
            shift, kind = -edge, "binary_complement"
        elif index % 5 == 1:
            shift, kind = edge, "binary_mint_and_sell"
        else:
            shift, kind = 0.0, ""
        if kind:
            planted[kind].append(market_id)
        books[yes_token] = _quote(yes_token, fair + shift, spread, depth, rng)
        books[no_token] = _quote(no_token, (1.0 - fair) + shift, spread, depth, rng)
        markets.append(
            Market(
                market_id=market_id,
                question=f"Binary market {index}",
                slug=f"binary-{index}",
                condition_id=f"0xbin{index}",
                outcomes=(Outcome("Yes", yes_token), Outcome("No", no_token)),
                event_id=f"BINEV{index}",
                event_slug=f"binary-event-{index}",
            )
        )

    for group_index in range(n_groups):
        event_id = f"GRPEV{group_index}"
        if group_index % 3 == 0:
            basket, kind = 1.0 - group_edge, "group_dutch_book"
        elif group_index % 3 == 1:
            basket, kind = 1.0 + group_edge, "neg_risk_no_basket"
        else:
            # A neutral basket must sit inside (1 - n*spread/2, 1 + n*spread/2):
            # outside that band the accumulated spread stops covering the
            # mispricing and the group becomes a genuine basket arbitrage.
            basket, kind = 1.0, ""
        if kind:
            planted[kind].append(event_id)

        weights = rng.dirichlet(np.ones(group_size) * 4.0)
        members: list[Market] = []
        for member_index in range(group_size):
            market_id = f"GRP{group_index}-{member_index}"
            yes_token, no_token = f"{market_id}-Y", f"{market_id}-N"
            fair = float(np.clip(weights[member_index] * basket, 0.03, 0.9))
            books[yes_token] = _quote(yes_token, fair, spread, depth, rng)
            books[no_token] = _quote(no_token, 1.0 - fair, spread, depth, rng)
            members.append(
                Market(
                    market_id=market_id,
                    question=f"Group {group_index} outcome {member_index}",
                    slug=f"group-{group_index}-{member_index}",
                    condition_id=f"0xgrp{group_index}{member_index}",
                    outcomes=(Outcome("Yes", yes_token), Outcome("No", no_token)),
                    neg_risk=True,
                    event_id=event_id,
                    event_slug=f"group-event-{group_index}",
                )
            )
        markets.extend(members)
        groups.append(
            MarketGroup(
                event_id=event_id,
                event_slug=f"group-event-{group_index}",
                markets=tuple(members),
                neg_risk=True,
            )
        )

    return ArbitrageFixture(
        markets=markets,
        books=books,
        groups=groups,
        planted={key: tuple(value) for key, value in planted.items()},
    )


def _quote(
    token_id: str,
    fair: float,
    spread: float,
    depth: float,
    rng: np.random.Generator,
) -> OrderBook:
    """Two levels either side of ``fair``, separated by ``spread``."""
    fair = float(np.clip(fair, 0.03, 0.95))
    bid = float(np.clip(fair - spread / 2.0, 0.01, 0.98))
    ask = float(np.clip(fair + spread / 2.0, 0.02, 0.99))
    size = float(depth * rng.uniform(0.7, 1.3))
    return OrderBook.from_levels(
        token_id,
        bids=[(bid, size), (max(bid - 0.03, 0.01), size * 2)],
        asks=[(ask, size), (min(ask + 0.03, 0.99), size * 2)],
    )
