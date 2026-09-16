from __future__ import annotations

import pandas as pd


def inverse_volatility_weights(volatility: pd.Series | pd.DataFrame):
    """Inverse-volatility weights normalized to sum to one."""
    inv = 1.0 / volatility.replace(0, pd.NA)
    return inv.div(inv.sum(axis=1), axis=0).fillna(0) if isinstance(inv, pd.DataFrame) else inv.div(inv.sum()).fillna(0)


def cap_weights(weights: pd.Series, max_weight: float = 0.20) -> pd.Series:
    """Project non-negative weights onto a capped simplex.

    Unlike clip-then-renormalize, the iterative redistribution preserves the
    requested upper bound after normalization.
    """
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    if w.empty:
        return w
    if len(w) * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the number of assets")
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= w.sum()
    for _ in range(len(w) + 1):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        capacity = (max_weight - w[under]).clip(lower=0)
        if capacity.sum() <= 0:
            raise ValueError("unable to redistribute capped portfolio weights")
        w[under] += excess * capacity / capacity.sum()
    return w / w.sum()
