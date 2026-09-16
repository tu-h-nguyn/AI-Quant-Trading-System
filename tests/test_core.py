import numpy as np
import pandas as pd

from quant_system.backtest.engine import backtest
from quant_system.backtest.metrics import max_drawdown
from quant_system.data.panel import build_price_panel, returns_panel
from quant_system.portfolio.optimization import min_variance_weights, risk_parity_weights
from quant_system.risk.management import volatility_target


def _prices(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    a = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    b = 80 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_panel_and_returns_are_aligned():
    px = _prices()
    panel = build_price_panel({"A": pd.DataFrame({"close": px.A}), "B": pd.DataFrame({"close": px.B})})
    ret = returns_panel(panel)
    assert panel.columns.tolist() == ["A", "B"]
    assert ret.shape == panel.shape


def test_backtest_shifts_signal():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=idx)
    signal = pd.Series([1.0, 1.0, 1.0], index=idx)
    result = backtest(df, signal, transaction_cost_bps=0)
    # The signal is lagged, so the first bar is flat however bullish the signal
    # is; from the second bar the position set on the first bar earns the move.
    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] > 0.0
    assert result.returns.iloc[2] > 0.0


def test_max_drawdown_is_non_positive():
    values = pd.Series([100, 120, 90, 110], dtype=float)
    assert max_drawdown(values.pct_change().fillna(0)) <= 0


def test_optimized_weights_sum_to_one_and_respect_cap():
    returns = returns_panel(_prices()).dropna()
    for weights in (min_variance_weights(returns, max_weight=0.8), risk_parity_weights(returns, max_weight=0.8)):
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= -1e-9).all()
        assert (weights <= 0.8 + 1e-9).all()


def test_volatility_target_is_lagged():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    returns = pd.Series(np.linspace(0.001, 0.02, len(idx)), index=idx)
    scale = volatility_target(returns, target_vol=0.1, window=5)
    assert scale.index.equals(idx)
    assert scale.iloc[0] == 0.0
