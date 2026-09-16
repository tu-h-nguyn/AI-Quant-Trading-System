import pandas as pd


def buy_and_hold(df):
    return pd.Series(1.0, index=df.index, name="signal")


def momentum(df, lookback=20):
    return (df.close.pct_change(lookback) > 0).astype(float).rename("signal")


def moving_average(df, fast=20, slow=50):
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    return (df.close.rolling(fast).mean() > df.close.rolling(slow).mean()).astype(float).rename("signal")
