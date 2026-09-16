import numpy as np

# A dispersion below this is floating-point residue, not variation. Pandas
# returns 1.8e-18 for the standard deviation of ten identical values, so an
# `== 0` guard never fires and a flat return series divides by that residue:
# a constant 1% a day reported a Sharpe of 8.7e+16 before this threshold.
ZERO_DISPERSION = 1e-15


def equity_curve(r, initial_capital=100000):
    return initial_capital * (1 + r.fillna(0)).cumprod()


def cagr(r, periods=252):
    r = r.dropna()
    if not len(r):
        return np.nan
    return (1 + r).prod() ** (periods / len(r)) - 1


def volatility(r, periods=252):
    return r.std(ddof=1) * np.sqrt(periods)


def sharpe(r):
    r = r.dropna()
    s = r.std(ddof=1)
    return np.nan if not np.isfinite(s) or s <= ZERO_DISPERSION else r.mean() / s * np.sqrt(252)


def sortino(r):
    """Return over downside deviation, annualized.

    Downside deviation is the root-mean-square of the negative part taken over
    the *whole* series, which is the standard definition. Measuring the sample
    standard deviation of only the losing periods, as this did, drops the
    frequency of losses entirely -- two strategies with the same loss sizes on
    five days and on fifty days scored identically -- and is undefined when
    there is exactly one losing period, so a year with a single catastrophic day
    reported no Sortino at all.
    """
    r = r.dropna()
    if r.empty:
        return np.nan
    d = float(np.sqrt((r.clip(upper=0.0) ** 2).mean()))
    return np.nan if not np.isfinite(d) or d <= ZERO_DISPERSION else r.mean() / d * np.sqrt(252)


def max_drawdown(r):
    curve = (1 + r.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def summary(r, position=None):
    out = {
        "total_return": float((1 + r.fillna(0)).prod() - 1),
        "cagr": cagr(r),
        "volatility": volatility(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
    }
    if position is not None:
        out["turnover"] = float(position.diff().abs().fillna(0).sum())
    return out
