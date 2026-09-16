import numpy as np
import pandas as pd
import pytest

from quant_system.evaluation.robustness import block_bootstrap_mean, percentile_interval, subperiod_summary
from quant_system.evaluation.splits import make_time_series_splitter
from quant_system.portfolio.rolling import apply_weights
from quant_system.portfolio.weights import cap_weights


def test_time_series_splitter_is_chronological_and_has_gap():
    splitter = make_time_series_splitter(n_splits=3, gap=2)
    X = np.arange(30)
    for train, test in splitter.split(X):
        assert train.max() < test.min()
        assert test.min() - train.max() > 2


def test_capped_weights_preserve_sum_and_cap():
    weights = cap_weights(pd.Series([0.95, 0.03, 0.02]), max_weight=0.4)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights <= 0.4 + 1e-12).all()


def test_capped_weights_reject_infeasible_cap():
    with pytest.raises(ValueError):
        cap_weights(pd.Series([0.5, 0.5, 0.0]), max_weight=0.2)


def test_rolling_application_uses_lagged_weights_and_costs():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, 0.0, 0.10]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}, index=idx)
    net, turnover, held = apply_weights(returns, weights, transaction_cost_bps=100)
    assert held.loc[idx[1], "A"] == 1.0
    assert net.loc[idx[1]] < 0.10
    assert turnover.loc[idx[2]] > 0


def test_block_bootstrap_is_reproducible():
    returns = pd.Series(np.linspace(-0.01, 0.01, 100))
    a = block_bootstrap_mean(returns, n_bootstrap=50, block_size=5, seed=7)
    b = block_bootstrap_mean(returns, n_bootstrap=50, block_size=5, seed=7)
    pd.testing.assert_series_equal(a, b)


def test_percentile_interval_and_subperiod_summary():
    samples = pd.Series(np.arange(100))
    lo, hi = percentile_interval(samples)
    assert lo < hi
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    table = subperiod_summary(pd.Series(0.001, index=idx))
    assert {"period", "return", "sharpe"}.issubset(table.columns)
