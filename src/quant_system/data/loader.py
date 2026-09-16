from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_symbol_data(symbol: str, directory: str | Path = "data/raw") -> pd.DataFrame:
    path = Path(directory) / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()


# Backward-compatible alias used by research scripts.
load_symbol = load_symbol_data
