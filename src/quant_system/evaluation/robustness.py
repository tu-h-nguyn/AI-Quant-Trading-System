from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    if window < 2:
        raise ValueError("window must be >= 2")
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return np.sqrt(252.0) * mean.div(std.replace(0, np.nan))


def parameter_sensitivity(results: pd.DataFrame, parameter: str, metric: str) -> pd.DataFrame:
    """Summarize a parameter sweep without selecting a winning configuration."""
    required = {parameter, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return (
        results.groupby(parameter)[metric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )


def bootstrap_mean(returns: pd.Series, n_bootstrap: int = 2000, seed: int = 42) -> pd.Series:
    """Bootstrap the mean return to quantify sampling uncertainty."""
    values = returns.dropna().to_numpy()
    if len(values) == 0:
        raise ValueError("returns must contain at least one finite observation")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    return pd.Series(samples.mean(axis=1), name="bootstrap_mean")
