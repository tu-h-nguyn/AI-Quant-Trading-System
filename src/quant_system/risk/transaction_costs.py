from __future__ import annotations

import pandas as pd


def estimate_cost(turnover: pd.Series, spread_bps: float = 2.0, commission_bps: float = 1.0) -> pd.Series:
    """Estimate proportional trading cost from turnover.

    Cost is expressed as a decimal return and therefore must be subtracted
    from gross portfolio returns.
    """
    if spread_bps < 0 or commission_bps < 0:
        raise ValueError("cost parameters must be non-negative")
    return turnover.abs() * (spread_bps + commission_bps) / 10_000.0
