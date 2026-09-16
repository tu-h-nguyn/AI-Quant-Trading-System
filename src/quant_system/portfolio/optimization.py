from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_variance_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.5,
    min_weight: float = 0.0,
) -> pd.Series:
    """Long-only minimum-variance portfolio using historical covariance."""
    if returns.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns)
    cov = returns.cov().fillna(0.0).to_numpy() * 252.0
    n = len(returns.columns)
    x0 = np.repeat(1.0 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        raise ValueError(f"Portfolio optimization failed: {result.message}")
    return pd.Series(result.x, index=returns.columns, name="weight")


def risk_parity_weights(returns: pd.DataFrame, max_weight: float = 0.5) -> pd.Series:
    """Numerical equal-risk-contribution portfolio with long-only bounds."""
    cov = returns.cov().fillna(0.0).to_numpy()
    n = cov.shape[0]
    if n == 1:
        return pd.Series(1.0, index=returns.columns)
    x0 = np.repeat(1.0 / n, n)

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 1e6
        marginal = cov @ w
        contrib = w * marginal / np.sqrt(port_var)
        target = contrib.sum() / n
        return float(np.square(contrib - target).sum())

    result = minimize(
        objective,
        x0=x0,
        bounds=[(0.0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    if not result.success:
        raise ValueError(f"Risk parity optimization failed: {result.message}")
    return pd.Series(result.x, index=returns.columns, name="weight")
