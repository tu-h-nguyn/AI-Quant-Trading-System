import numpy as np
import pandas as pd
import pytest

from quant_system.evaluation.robustness import (
    bootstrap_mean,
    parameter_sensitivity,
    rolling_sharpe,
)
from quant_system.risk.transaction_costs import estimate_cost


def test_cost_is_zero_without_turnover():
    turnover = pd.Series([0.0, 0.0, 0.0])
    assert estimate_cost(turnover).sum() == 0.0


def test_cost_scales_linearly():
    turnover = pd.Series([1.0])
    assert estimate_cost(turnover, spread_bps=2, commission_bps=1).iloc[0] == pytest.approx(0.0003)


def test_rolling_sharpe_requires_minimum_window():
    with pytest.raises(ValueError):
        rolling_sharpe(pd.Series([0.01, 0.02]), window=1)


def test_sensitivity_summary():
    frame = pd.DataFrame({"window": [5, 5, 20], "sharpe": [0.4, 0.6, 0.8]})
    out = parameter_sensitivity(frame, "window", "sharpe")
    assert list(out["window"]) == [5, 20]
    assert out.loc[out.window == 5, "count"].iloc[0] == 2


def test_bootstrap_is_reproducible():
    returns = pd.Series(np.arange(10, dtype=float))
    a = bootstrap_mean(returns, n_bootstrap=100, seed=7)
    b = bootstrap_mean(returns, n_bootstrap=100, seed=7)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
