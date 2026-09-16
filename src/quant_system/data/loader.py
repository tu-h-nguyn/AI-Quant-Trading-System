import pandas as pd
from pathlib import Path

def load_symbol_data(symbol, directory="data/raw"):
    path = Path(directory) / f"{symbol.upper()}.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
