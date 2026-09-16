from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import ZERO_DISPERSION


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    if window < 2:
        raise ValueError("window must be >= 2")
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return np.sqrt(252.0) * mean.div(std.replace(0, np.nan))


def parameter_sensitivity(results: pd.DataFrame, parameter: str, metric: str) -> pd.DataFrame:
    """Aggregate a parameter sweep; intentionally does not select a winner."""
    missing = {parameter, metric} - set(results.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return results.groupby(parameter)[metric].agg(["count", "mean", "std", "min", "max"]).reset_index()


def block_bootstrap_mean(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    block_size: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Moving-block bootstrap for a dependent return series."""
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("returns must contain at least one finite observation")
    if n_bootstrap < 1 or block_size < 1:
        raise ValueError("n_bootstrap and block_size must be positive")
    if block_size > len(values):
        block_size = len(values)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(values) - block_size + 1)
    n_blocks = int(np.ceil(len(values) / block_size))
    out = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([values[s : s + block_size] for s in chosen])[: len(values)]
        out[i] = sample.mean()
    return pd.Series(out, name="bootstrap_mean")


def bootstrap_mean(
    returns: pd.Series,
    n_bootstrap: int = 2000,
    block_size: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Backward-compatible alias for the moving-block bootstrap."""
    return block_bootstrap_mean(
        returns,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=seed,
    )


def percentile_interval(samples: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return (float(samples.quantile(alpha / 2)), float(samples.quantile(1 - alpha / 2)))


def subperiod_summary(returns: pd.Series, periods_per_year: int = 252) -> pd.DataFrame:
    """Return calendar-year diagnostics to expose parameter/regime instability."""
    rows = []
    clean = returns.dropna()
    for period, r in clean.groupby(clean.index.year):
        vol = r.std(ddof=1) * np.sqrt(periods_per_year)
        rows.append(
            {
                "period": int(period),
                "observations": int(len(r)),
                "return": float((1 + r).prod() - 1),
                "annualized_volatility": float(vol),
                # The shared threshold, imported rather than copied: two values
                # drifting apart is exactly what it was introduced to prevent.
                "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))
                if r.std(ddof=1) > ZERO_DISPERSION
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
