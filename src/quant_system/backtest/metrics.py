import numpy as np
import pandas as pd

def equity_curve(r, initial_capital=100000): return initial_capital*(1+r.fillna(0)).cumprod()
def cagr(r, periods=252):
    r=r.dropna();
    if not len(r): return np.nan
    return (1+r).prod()**(periods/len(r))-1
def volatility(r, periods=252): return r.std(ddof=1)*np.sqrt(periods)
def sharpe(r):
    r=r.dropna(); return np.nan if r.std(ddof=1)==0 else r.mean()/r.std(ddof=1)*np.sqrt(252)
def sortino(r):
    r=r.dropna(); d=r[r<0].std(ddof=1); return np.nan if d==0 or np.isnan(d) else r.mean()/d*np.sqrt(252)
def max_drawdown(r):
    curve=(1+r.fillna(0)).cumprod(); return float((curve/curve.cummax()-1).min())
def summary(r, position=None):
    out={"total_return":float((1+r.fillna(0)).prod()-1),"cagr":cagr(r),"volatility":volatility(r),"sharpe":sharpe(r),"sortino":sortino(r),"max_drawdown":max_drawdown(r)}
    if position is not None: out["turnover"]=float(position.diff().abs().fillna(0).sum())
    return out
