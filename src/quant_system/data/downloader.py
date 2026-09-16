import pandas as pd
import yfinance as yf
from pathlib import Path

def download_ohlcv(symbols, start, end=None, interval="1d"):
    result = {}
    for symbol in symbols:
        df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if df.empty: raise ValueError(f"No data returned for {symbol}")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        result[symbol] = df[["open","high","low","close","volume"]].dropna().sort_index()
    return result

def save_symbol_data(df, symbol, directory="data/raw"):
    path = Path(directory); path.mkdir(parents=True, exist_ok=True)
    out = path / f"{symbol.upper()}.csv"; df.to_csv(out, index_label="date"); return out
