"""Event-driven backtesting for binary prediction-market positions.

A prediction-market backtest cannot borrow the equities convention of
compounding a daily return series. Positions are lumpy, illiquid, settle at 0
or 1 on their own schedule, and lock capital until they do. This engine
therefore simulates cash directly:

* capital committed to an open position is unavailable to later trades, so the
  simulation cannot spend the same dollar twice;
* entries are priced at the ask plus fees rather than at the mid, because a mid
  fill is not available to a taker;
* positions are held to settlement and pay exactly one dollar per winning share;
* the bankroll used for sizing tapers as drawdown deepens.

Every decision uses only the probability supplied for that observation, which
the caller is expected to have produced out of sample -- typically through
:func:`quant_system.evaluation.walk_forward.walk_forward_predict`.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .kelly import drawdown_scaled_bankroll, size_position
from .metrics import bankroll_summary, edge_realization, trade_summary
from .pricing import (
    MAX_PRICE,
    MIN_PRICE,
    resolution_adjusted_probability,
    taker_fee_per_share,
)

REQUIRED_PANEL_COLUMNS = ("market_id", "timestamp", "price", "label")


@dataclass(frozen=True)
class TradeConfig:
    """Frictions, sizing rules, and risk limits for a simulation run."""

    bankroll: float = 10_000.0
    fee_bps: float = 0.0
    default_spread: float = 0.01
    extra_slippage: float = 0.0
    kelly_multiplier: float = 0.25
    max_fraction: float = 0.02
    max_total_exposure: float = 0.20
    min_edge: float = 0.03
    shrinkage: float = 0.30
    max_drawdown: float = 0.25
    drawdown_floor: float = 0.25
    one_trade_per_market: bool = True
    min_notional: float = 5.0
    # Exit rules, disabled by default so hold-to-settlement remains the baseline.
    # Holding a converged position earns nothing while its capital is unavailable
    # to anything else, so exiting early trades a slice of the edge for velocity.
    exit_edge_fraction: float | None = None
    stop_loss_move: float | None = None
    # Probability that a market settles on something other than its merits --
    # a technicality, a dispute, a void. Applied twice: the forecast is
    # discounted for it before sizing, and the simulation realizes it at that
    # rate, so the assumption is both paid for and charged.
    resolution_risk: float = 0.0
    # Zero is the only value that keeps this a cost. A non-zero recovery pays a
    # cheap contract more than it cost, so the strategy would learn to buy
    # longshots betting on the venue failing.
    resolution_recovery: float = 0.0
    resolution_seed: int = 42

    def __post_init__(self) -> None:
        if self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        if self.default_spread < 0 or self.extra_slippage < 0:
            raise ValueError("default_spread and extra_slippage must be non-negative")
        if not 0.0 < self.max_total_exposure <= 1.0:
            raise ValueError("max_total_exposure must lie in (0, 1]")
        if self.exit_edge_fraction is not None and not 0.0 <= self.exit_edge_fraction < 1.0:
            raise ValueError("exit_edge_fraction must lie in [0, 1)")
        if self.stop_loss_move is not None and not 0.0 < self.stop_loss_move <= 1.0:
            raise ValueError("stop_loss_move must lie in (0, 1]")
        if not 0.0 <= self.resolution_risk <= 1.0:
            raise ValueError("resolution_risk must lie in [0, 1]")
        if not 0.0 <= self.resolution_recovery <= 1.0:
            raise ValueError("resolution_recovery must lie in [0, 1]")


@dataclass
class BacktestResult:
    """Settled trade ledger, bankroll path, and aggregate diagnostics."""

    trades: pd.DataFrame
    equity: pd.Series
    summary: dict[str, float] = field(default_factory=dict)

    @property
    def final_bankroll(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else float("nan")


@dataclass
class _OpenPosition:
    """A live position, carrying what the exit rules need to evaluate it."""

    market_id: str
    question: str
    side: str
    shares: float
    capital: float
    entry_price: float
    predicted_probability: float
    predicted_edge: float
    entry_time: pd.Timestamp
    settle_time: pd.Timestamp
    label: int
    entry_market_price: float = float("nan")


def run_backtest(
    panel: pd.DataFrame,
    probabilities: pd.Series,
    config: TradeConfig | None = None,
) -> BacktestResult:
    """Simulate trading a probability forecast against a snapshot panel.

    ``panel`` is the long snapshot frame produced by
    :func:`quant_system.polymarket.features.build_snapshot_features`, and
    ``probabilities`` is indexed by the same ``observation_id``. Rows without a
    forecast are simply not traded, which is how a walk-forward warm-up period
    is represented.
    """
    settings = config or TradeConfig()
    missing = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")

    frame = panel.loc[panel.index.intersection(probabilities.index)].copy()
    if frame.empty:
        return BacktestResult(_empty_trades(), pd.Series(dtype=float), _summary(_empty_trades(),
                                                                               pd.Series(dtype=float)))
    frame["probability"] = probabilities.reindex(frame.index).astype(float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp", "market_id"], kind="mergesort")

    end_dates = (
        pd.to_datetime(frame["end_date"], utc=True)
        if "end_date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    final_time = frame["timestamp"].max()

    bankroll = settings.bankroll
    peak = bankroll
    locked = 0.0
    open_positions: list[_OpenPosition] = []
    traded_markets: set[str] = set()
    settled: list[dict] = []
    equity_index: list[pd.Timestamp] = []
    equity_values: list[float] = []

    for observation_id, row in frame.iterrows():
        now = row["timestamp"]
        # Settle first: capital released by a matured position is available to
        # the trade being considered at this same timestamp.
        bankroll, locked = _settle_due(
            open_positions, now, bankroll, locked, settled, equity_index, equity_values,
            settings,
        )
        peak = max(peak, bankroll)

        market_id = str(row["market_id"])
        if settings.exit_edge_fraction is not None or settings.stop_loss_move is not None:
            bankroll, locked = _apply_exits(
                open_positions, market_id, row, now, settings,
                bankroll, locked, settled, equity_index, equity_values,
            )
            peak = max(peak, bankroll)

        if settings.one_trade_per_market and market_id in traded_markets:
            continue
        probability = row["probability"]
        if not np.isfinite(probability):
            continue

        market_price = float(np.clip(float(row["price"]), MIN_PRICE, MAX_PRICE))
        yes_ask, no_ask = _entry_prices(row, market_price, settings)

        sizing_bankroll = drawdown_scaled_bankroll(
            bankroll, peak, settings.max_drawdown, settings.drawdown_floor
        )
        available = min(
            max(bankroll - locked, 0.0),
            max(settings.max_total_exposure * bankroll - locked, 0.0),
        )
        if sizing_bankroll <= 0 or available < settings.min_notional:
            continue

        # Discount both sides for settlement that may not follow the merits,
        # before any of the sizing gates see the forecast.
        candidates = (
            (
                "yes",
                resolution_adjusted_probability(
                    probability, settings.resolution_risk, settings.resolution_recovery
                ),
                yes_ask,
                market_price,
            ),
            (
                "no",
                resolution_adjusted_probability(
                    1.0 - probability, settings.resolution_risk, settings.resolution_recovery
                ),
                no_ask,
                1.0 - market_price,
            ),
        )
        best = None
        for side, side_probability, side_price, side_market in candidates:
            sized = size_position(
                bankroll=sizing_bankroll,
                probability=side_probability,
                effective_price=side_price,
                kelly_multiplier=settings.kelly_multiplier,
                max_fraction=settings.max_fraction,
                min_edge=settings.min_edge,
                shrinkage=settings.shrinkage,
                market_price=side_market,
                fee_bps=settings.fee_bps,
                include_fee=True,
            )
            if sized.notional > 0 and (best is None or sized.edge > best[1].edge):
                best = (side, sized)
        if best is None:
            continue

        side, sized = best
        notional = min(sized.notional, available)
        if notional < settings.min_notional:
            continue
        shares = notional / sized.effective_price

        settle_time = end_dates.loc[observation_id]
        if pd.isna(settle_time) or settle_time < now:
            settle_time = final_time
        open_positions.append(
            _OpenPosition(
                market_id=market_id,
                question=str(row.get("question", "")),
                side=side,
                shares=shares,
                capital=notional,
                entry_price=sized.effective_price,
                predicted_probability=sized.probability,
                predicted_edge=sized.edge,
                entry_time=now,
                settle_time=settle_time,
                label=int(row["label"]),
                entry_market_price=market_price,
            )
        )
        locked += notional
        traded_markets.add(market_id)

    # Anything still open at the end of the sample settles on its known label.
    bankroll, locked = _settle_due(
        open_positions,
        pd.Timestamp.max.tz_localize("UTC"),
        bankroll,
        locked,
        settled,
        equity_index,
        equity_values,
        settings,
    )

    trades = pd.DataFrame(settled) if settled else _empty_trades()
    equity = pd.Series(equity_values, index=pd.Index(equity_index, name="settled_at"), dtype=float)
    if not equity.empty:
        equity = pd.concat([pd.Series([settings.bankroll], index=[frame["timestamp"].min()]), equity])
    return BacktestResult(trades=trades, equity=equity, summary=_summary(trades, equity))


def _entry_prices(
    row: pd.Series,
    market_price: float,
    settings: TradeConfig,
) -> tuple[float, float]:
    """Taker entry prices for both sides, from the book when it is available.

    Without quoted depth the ask is reconstructed as the mid plus half the
    configured spread, which is deliberately pessimistic relative to trading at
    the mid and keeps a book-less panel from reporting a friction-free edge.
    """
    half_spread = settings.default_spread / 2.0 + settings.extra_slippage
    yes_ask = row.get("yes_ask", np.nan)
    no_ask = row.get("no_ask", np.nan)
    if not np.isfinite(yes_ask):
        yes_ask = market_price + half_spread
    if not np.isfinite(no_ask):
        no_ask = (1.0 - market_price) + half_spread
    return (
        float(np.clip(yes_ask, MIN_PRICE, MAX_PRICE)),
        float(np.clip(no_ask, MIN_PRICE, MAX_PRICE)),
    )


def _apply_exits(
    open_positions: list[_OpenPosition],
    market_id: str,
    row: pd.Series,
    now: pd.Timestamp,
    settings: TradeConfig,
    bankroll: float,
    locked: float,
    settled: list[dict],
    equity_index: list[pd.Timestamp],
    equity_values: list[float],
) -> tuple[float, float]:
    """Close any position in ``market_id`` whose exit condition has triggered.

    Two rules, both optional:

    ``exit_edge_fraction``
        Close once the remaining edge has decayed to that fraction of the edge
        at entry. A position whose price has converged to the forecast has
        nothing left to earn, and the capital behind it is doing nothing.

    ``stop_loss_move``
        Close when the market has moved that far against the position. This is a
        risk control, not an edge rule -- in a prediction market a price moving
        against you usually means information arrived, but it will sometimes cut
        a position that was right early.

    Exits pay the spread a second time, so a rule that fires often can consume
    more in frictions than the freed capital earns.
    """
    held = [item for item in open_positions if item.market_id == market_id]
    if not held:
        return bankroll, locked

    market_price = float(np.clip(float(row["price"]), MIN_PRICE, MAX_PRICE))
    for position in held:
        if position.settle_time <= now:
            continue  # settlement takes precedence and is handled by _settle_due
        side_price = market_price if position.side == "yes" else 1.0 - market_price
        entry_side_price = (
            position.entry_market_price
            if position.side == "yes"
            else 1.0 - position.entry_market_price
        )
        exit_price = _exit_price(side_price, settings)

        reason = ""
        if settings.stop_loss_move is not None and np.isfinite(entry_side_price):
            if entry_side_price - side_price >= settings.stop_loss_move:
                reason = "stop_loss"
        if not reason and settings.exit_edge_fraction is not None:
            remaining = position.predicted_probability - exit_price
            if position.predicted_edge > 0 and remaining < (
                settings.exit_edge_fraction * position.predicted_edge
            ):
                reason = "convergence"
        if not reason:
            continue

        proceeds = position.shares * exit_price
        profit = proceeds - position.capital
        bankroll += profit
        locked = max(locked - position.capital, 0.0)
        open_positions.remove(position)
        settled.append(_trade_record(position, now, profit, exit_price, reason, bankroll))
        equity_index.append(now)
        equity_values.append(bankroll)
    return bankroll, locked


def _exit_price(side_price: float, settings: TradeConfig) -> float:
    """Price received when closing a position, after crossing the spread again."""
    half_spread = settings.default_spread / 2.0 + settings.extra_slippage
    gross = side_price - half_spread
    net = gross - taker_fee_per_share(
        float(np.clip(gross, MIN_PRICE, MAX_PRICE)), settings.fee_bps
    )
    return float(np.clip(net, 0.0, 1.0))


def _trade_record(
    position: _OpenPosition,
    closed_at: pd.Timestamp,
    profit: float,
    payoff: float,
    reason: str,
    bankroll: float,
) -> dict:
    """One row of the settled ledger, however the position was closed."""
    return {
        "market_id": position.market_id,
        "question": position.question,
        "side": position.side,
        "entry_time": position.entry_time,
        "settled_at": closed_at,
        "holding_days": (closed_at - position.entry_time).total_seconds() / 86_400.0,
        "shares": position.shares,
        "capital": position.capital,
        "entry_price": position.entry_price,
        "predicted_probability": position.predicted_probability,
        "predicted_edge": position.predicted_edge,
        "label": position.label,
        "payoff": float(payoff),
        "profit": profit,
        "profit_per_share": profit / position.shares if position.shares > 0 else np.nan,
        "exit_reason": reason,
        "bankroll_after": bankroll,
    }


def _realized_payoff(position: _OpenPosition, settings: TradeConfig) -> float:
    """Settlement a position actually receives, including a failed resolution.

    The draw is derived from the market id rather than from an iteration
    counter, so the same market fails in the same runs regardless of how many
    other positions were opened first. Without that, changing an unrelated
    parameter would reshuffle which markets void and the comparison between two
    configurations would be noise.
    """
    if settings.resolution_risk > 0 and _resolution_failed(
        position.market_id, settings.resolution_seed, settings.resolution_risk
    ):
        return float(settings.resolution_recovery)
    return float(position.label if position.side == "yes" else 1 - position.label)


def _resolution_failed(market_id: str, seed: int, risk: float) -> bool:
    """Stable per-market draw for whether settlement went off the merits."""
    stream = zlib.crc32(f"{seed}:{market_id}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(stream).uniform() < risk


def _settle_due(
    open_positions: list[_OpenPosition],
    now: pd.Timestamp,
    bankroll: float,
    locked: float,
    settled: list[dict],
    equity_index: list[pd.Timestamp],
    equity_values: list[float],
    settings: TradeConfig,
) -> tuple[float, float]:
    """Settle every position matured at ``now``, releasing its capital."""
    still_open = []
    for position in sorted(open_positions, key=lambda item: item.settle_time):
        if position.settle_time > now:
            still_open.append(position)
            continue
        payoff = _realized_payoff(position, settings)
        proceeds = position.shares * float(payoff)
        profit = proceeds - position.capital
        bankroll += profit
        locked = max(locked - position.capital, 0.0)
        settled.append(
            _trade_record(
                position, position.settle_time, profit, float(payoff), "settlement", bankroll
            )
        )
        equity_index.append(position.settle_time)
        equity_values.append(bankroll)
    open_positions[:] = still_open
    return bankroll, locked


def _summary(trades: pd.DataFrame, equity: pd.Series) -> dict[str, float]:
    summary = {**trade_summary(trades), **bankroll_summary(equity)}
    if not trades.empty:
        realization = edge_realization(trades["predicted_edge"], trades["profit_per_share"])
        summary.update({f"edge_{key}": value for key, value in realization.items()})
    return summary


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market_id", "question", "side", "entry_time", "settled_at", "holding_days",
            "shares", "capital", "entry_price", "predicted_probability", "predicted_edge",
            "label", "payoff", "profit", "profit_per_share", "exit_reason", "bankroll_after",
        ]
    )


def theoretical_break_even_edge(price: float, spread: float, fee_bps: float) -> float:
    """Minimum true edge a taker needs before frictions to break even.

    Useful as a sanity gate on ``TradeConfig.min_edge``: a threshold below this
    value guarantees losses on average no matter how good the forecast is.
    """
    clipped = float(np.clip(price, MIN_PRICE, MAX_PRICE))
    return spread / 2.0 + taker_fee_per_share(clipped, fee_bps)
