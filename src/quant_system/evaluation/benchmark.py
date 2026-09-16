from __future__ import annotations

import numpy as np
import pandas as pd


def benchmark_returns(returns: pd.DataFrame, benchmark: str) -> pd.Series:
    """Return a named asset as the benchmark series."""
    if benchmark not in returns.columns:
        raise KeyError(f"benchmark {benchmark!r} not found")
    return returns[benchmark].rename(f"{benchmark}_benchmark")


def relative_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Compute benchmark-relative diagnostics without ranking strategies."""
    s, b = strategy.align(benchmark, join="inner")
    active = s - b
    active_vol = active.std(ddof=1) * np.sqrt(252.0)
    tracking_error = float(active_vol) if pd.notna(active_vol) else float("nan")
    info_ratio = float(active.mean() * 252.0 / active_vol) if active_vol and active_vol > 0 else float("nan")
    return {
        "active_return": float(active.mean() * 252.0),
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "up_capture": float(s[s.index.isin(b[b > 0].index)].mean() / b[b > 0].mean())
        if (b > 0).any() else float("nan"),
        "down_capture": float(s[s.index.isin(b[b < 0].index)].mean() / b[b < 0].mean())
        if (b < 0).any() else float("nan"),
    }
