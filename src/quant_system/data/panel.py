from __future__ import annotations

import pandas as pd


def build_price_panel(frames: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Align multiple OHLCV frames into a single date x asset panel."""
    series = {}
    for symbol, frame in frames.items():
        if field not in frame.columns:
            raise KeyError(f"{symbol} is missing required field: {field}")
        series[symbol] = frame[field].rename(symbol)
    panel = pd.concat(series, axis=1).sort_index()
    return panel.dropna(how="all")


def returns_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns for all assets."""
    if price_panel.empty:
        return price_panel.copy()
    return price_panel.pct_change().replace([float("inf"), -float("inf")], pd.NA)
