"""Binary prediction-market contract arithmetic.

A Polymarket outcome share costs ``price`` dollars, settles at ``1`` if the
outcome occurs and ``0`` otherwise. Price is therefore a directly readable
implied probability, which makes every quantity below exact rather than
model-dependent -- the only modelling choice is the fee schedule.

Polymarket's documented taker fee is symmetric around the midpoint::

    fee_per_share = rate * min(price, 1 - price)

so it is largest for coin-flip markets and vanishes at the extremes. The rate
is configurable and defaults to zero, matching markets that charge no taker
fee; set it from configuration rather than assuming a venue-wide value.
"""

from __future__ import annotations

MIN_PRICE = 1e-6
MAX_PRICE = 1.0 - 1e-6


def validate_price(price: float, name: str = "price") -> float:
    """Return ``price`` as a float, rejecting values outside the unit interval."""
    value = float(price)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
    return value


def validate_probability(probability: float, name: str = "probability") -> float:
    """Return ``probability`` as a float, allowing the degenerate endpoints."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def taker_fee_per_share(price: float, fee_bps: float = 0.0) -> float:
    """Fee charged per share for a taker order at ``price``."""
    if fee_bps < 0:
        raise ValueError("fee_bps must be non-negative")
    value = validate_price(price)
    return float(fee_bps) / 10_000.0 * min(value, 1.0 - value)


def buy_cost_per_share(price: float, fee_bps: float = 0.0) -> float:
    """All-in cash outlay per share bought, including fees."""
    return validate_price(price) + taker_fee_per_share(price, fee_bps)


def sell_proceeds_per_share(price: float, fee_bps: float = 0.0) -> float:
    """Cash received per share sold, net of fees."""
    return validate_price(price) - taker_fee_per_share(price, fee_bps)


def breakeven_probability(price: float, fee_bps: float = 0.0) -> float:
    """Probability at which buying at ``price`` has zero expected value.

    With fees this sits above the quoted price: the quote understates the
    probability a buyer actually needs in order to break even.
    """
    return buy_cost_per_share(price, fee_bps)


def expected_value_per_share(
    probability: float,
    price: float,
    fee_bps: float = 0.0,
) -> float:
    """Expected profit per share bought, in dollars.

    ``probability`` is the forecast probability that *this* share settles at 1,
    so the caller passes ``1 - q`` when pricing the complementary side.
    """
    q = validate_probability(probability)
    return q - buy_cost_per_share(price, fee_bps)


def edge(probability: float, price: float, fee_bps: float = 0.0) -> float:
    """Alias for :func:`expected_value_per_share` in probability units.

    For a binary contract the dollar expected value per share and the
    probability edge coincide, because the payoff is exactly one dollar.
    """
    return expected_value_per_share(probability, price, fee_bps)


def roi_if_correct(price: float, fee_bps: float = 0.0) -> float:
    """Return on invested capital when the purchased side settles at 1."""
    cost = buy_cost_per_share(price, fee_bps)
    if cost <= 0:
        raise ValueError("all-in cost must be positive")
    return (1.0 - cost) / cost


def complement_price(price: float) -> float:
    """Price of the opposite side implied by no-arbitrage on a binary pair."""
    return 1.0 - validate_price(price)


def shrink_probability(
    probability: float,
    market_price: float,
    shrinkage: float,
) -> float:
    """Pull a forecast toward the market price before sizing.

    Quoted prices aggregate information the model does not see, and a stale
    quote that looks mispriced is usually adverse selection rather than edge.
    ``shrinkage`` of 0 trusts the model fully and 1 defers entirely to price;
    values around 0.3-0.5 are a conservative default for live sizing.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    q = validate_probability(probability)
    p = validate_price(market_price)
    return (1.0 - shrinkage) * q + shrinkage * p


def resolution_adjusted_probability(
    probability: float,
    resolution_risk: float,
    recovery: float = 0.0,
) -> float:
    """Discount a forecast for the chance the market does not settle on merit.

    A prediction market pays out on what the resolver decides, not on what
    happened. Questions are settled on technicalities, disputed, or voided, and
    the contract then pays something unrelated to the forecast. With probability
    ``resolution_risk`` the payoff is ``recovery`` regardless of the outcome::

        q_effective = (1 - risk) * q + risk * recovery

    ``recovery`` defaults to **zero**, which is the only setting that makes this
    a risk model rather than a second source of edge. A non-zero recovery pays
    out more than a cheap contract cost, so under ``recovery=0.5`` a five-cent
    longshot *gains* from resolution risk and a strategy optimized against it
    learns to buy lottery tickets on the venue failing. That is an artifact of
    assuming voids are independent of price, which they are not.

    Set a non-zero recovery only to model a venue that demonstrably splits
    voided markets, and read the result knowing it is optimistic.
    """
    if not 0.0 <= resolution_risk <= 1.0:
        raise ValueError("resolution_risk must lie in [0, 1]")
    q = validate_probability(probability)
    rec = validate_probability(recovery, name="recovery")
    return (1.0 - resolution_risk) * q + resolution_risk * rec


def max_tolerable_resolution_risk(
    probability: float,
    price: float,
    fee_bps: float = 0.0,
    recovery: float = 0.0,
) -> float:
    """Resolution risk at which a position's expected value reaches zero.

    Reads as: how unreliable can settlement be before this trade stops being
    worth taking. At a thin edge the answer is often a few percent, which is why
    a gate set just above the friction floor is not actually conservative.
    """
    cost = buy_cost_per_share(price, fee_bps)
    q = validate_probability(probability)
    rec = validate_probability(recovery, name="recovery")
    if q <= cost:
        return 0.0
    if abs(q - rec) < 1e-12:
        return 1.0
    risk = (q - cost) / (q - rec)
    return float(min(max(risk, 0.0), 1.0))


def settlement_value(side: str, outcome: int) -> float:
    """Dollar settlement of one share of ``side`` given the realized outcome.

    ``side`` is ``"yes"`` or ``"no"`` and ``outcome`` is 1 when the market's
    YES outcome occurred.
    """
    normalized = side.strip().lower()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
    realized = int(outcome)
    if realized not in {0, 1}:
        raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
    return float(realized) if normalized == "yes" else float(1 - realized)
