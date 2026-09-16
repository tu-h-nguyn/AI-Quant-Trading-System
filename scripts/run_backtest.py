"""Single-asset baseline backtest for the first symbol in the configuration.

The smallest entry point in the repository: it runs the non-ML rules against one
symbol and prints their cost-aware metrics, which is the comparison every other
study is ultimately measured against.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_system.backtest.engine import backtest  # noqa: E402
from quant_system.config import load_config  # noqa: E402
from quant_system.data.loader import load_symbol_data  # noqa: E402
from quant_system.strategies.baselines import momentum, moving_average  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    symbol = config["data"]["symbols"][0]
    df = load_symbol_data(symbol, ROOT / "data" / "raw")

    features = config["features"]
    strategies = {
        "Momentum": momentum(df, 20),
        "MovingAverage": moving_average(
            df, features["moving_average_fast"], features["moving_average_slow"]
        ),
    }
    for name, signal in strategies.items():
        result = backtest(
            df,
            signal,
            config["backtest"]["transaction_cost_bps"],
            config["backtest"]["initial_capital"],
        )
        print(name, result.metrics)


if __name__ == "__main__":
    main()
