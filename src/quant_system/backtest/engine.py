import pandas as pd
from dataclasses import dataclass
from .metrics import equity_curve, summary

@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    positions: pd.Series
    metrics: dict

def backtest(df, signal, transaction_cost_bps=5.0, initial_capital=100000.0):
    x=df.join(signal.rename("signal"),how="inner")
    daily=x.close.pct_change().fillna(0)
    positions=x.signal.shift(1).fillna(0)
    gross=positions*daily
    costs=positions.diff().abs().fillna(0)*(transaction_cost_bps/10000)
    net=gross-costs
    return BacktestResult(net.rename("strategy_return"),equity_curve(net,initial_capital),positions.rename("position"),summary(net,positions))
