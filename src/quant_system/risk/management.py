from __future__ import annotations

import pandas as pd


def volatility_target(
    returns: pd.Series,
    target_vol: float = 0.10,
    window: int = 20,
    max_leverage: float = 1.5,
) -> pd.Series:
    """Scale a base position by trailing realized volatility, lagged one day."""
    if target_vol <= 0 or max_leverage <= 0:
        raise ValueError("target_vol and max_leverage must be positive")
    realized = returns.rolling(window).std() * (252.0**0.5)
    leverage = (target_vol / realized).replace([float("inf"), -float("inf")], pd.NA)
    leverage = leverage.clip(lower=0.0, upper=max_leverage).fillna(0.0)
    return leverage.shift(1).fillna(0.0).rename("volatility_scale")


def apply_drawdown_limit(
    equity: pd.Series,
    positions: pd.Series,
    max_drawdown: float = 0.20,
) -> pd.Series:
    """Flatten positions while the running drawdown breaches a configured limit."""
    if not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    guarded = positions.where(drawdown > -max_drawdown, 0.0)
    return guarded.rename(positions.name or "position")
