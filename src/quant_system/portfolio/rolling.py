from __future__ import annotations

import pandas as pd


def rolling_weight_schedule(
    returns: pd.DataFrame,
    optimizer,
    lookback: int = 252,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Build weights using only trailing data, then hold until next rebalance.

    The first eligible rebalance occurs after ``lookback`` observations. The
    returned schedule is intended to be shifted by the backtest engine before
    applying returns, so weights estimated at t affect returns from t+1.
    """
    if lookback < 2 or rebalance_every < 1:
        raise ValueError("lookback must be >= 2 and rebalance_every must be >= 1")
    returns = returns.sort_index().copy()
    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for i in range(lookback, len(returns), rebalance_every):
        window = returns.iloc[i - lookback:i].dropna(how="all")
        if len(window) < 2:
            continue
        valid = window.columns[window.notna().sum() >= min(lookback, 20)]
        if len(valid) == 0:
            continue
        w = optimizer(window[valid]).reindex(returns.columns).fillna(0.0)
        next_end = min(i + rebalance_every, len(returns))
        weights.iloc[i:next_end] = w.to_numpy()
    return weights.ffill().fillna(0.0)


def apply_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Apply lagged portfolio weights and charge costs on absolute weight turnover."""
    aligned_r, aligned_w = returns.align(weights, join="inner", axis=0)
    aligned_r, aligned_w = aligned_r.align(aligned_w, join="inner", axis=1)
    held = aligned_w.shift(1).fillna(0.0)
    gross = aligned_r.mul(held).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    costs = turnover * transaction_cost_bps / 10_000.0
    return gross - costs, turnover, held
