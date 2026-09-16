import pandas as pd


def add_return_features(df, windows=(1, 5, 20)):
    out = df.copy()
    for w in windows:
        out[f"return_{w}d"] = out["close"].pct_change(w)
    return out


def add_volatility_features(df, windows=(5, 20)):
    out = df.copy()
    r = out["close"].pct_change()
    for w in windows:
        out[f"volatility_{w}d"] = r.rolling(w).std()
    return out


def add_technical_features(df, fast_window=20, slow_window=50):
    out = df.copy()
    fast = out.close.rolling(fast_window).mean()
    slow = out.close.rolling(slow_window).mean()
    out["ma_fast"] = fast
    out["ma_slow"] = slow
    out["ma_ratio"] = fast / slow - 1
    out["volume_change"] = out.volume.pct_change()
    return out


def build_feature_frame(df, return_windows, volatility_windows, fast_window, slow_window, horizon):
    x = add_return_features(df, return_windows)
    x = add_volatility_features(x, volatility_windows)
    x = add_technical_features(x, fast_window, slow_window)
    future = x.close.shift(-horizon) / x.close - 1
    y = (future > 0).astype(int)
    cols = [c for c in x.columns if c not in {"open", "high", "low", "close", "volume"}]
    X = x[cols].replace([float("inf"), -float("inf")], pd.NA)
    valid = X.notna().all(axis=1) & future.notna()
    return X.loc[valid], y.loc[valid], cols
