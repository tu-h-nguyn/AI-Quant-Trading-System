from __future__ import annotations

from pathlib import Path

import yaml

from quant_system.data.loader import load_symbol_data
from quant_system.data.panel import build_price_panel, returns_panel
from quant_system.portfolio.optimization import min_variance_weights, risk_parity_weights


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    symbols = config["data"]["symbols"]
    frames = {
        symbol: load_symbol_data(symbol, ROOT / "data" / "raw")
        for symbol in symbols
    }
    prices = build_price_panel(frames)
    returns = returns_panel(prices).dropna(how="all")
    estimation_returns = returns.dropna()
    if estimation_returns.empty:
        raise ValueError("No overlapping return observations available for portfolio estimation")

    method = config["portfolio"]["method"]
    cap = float(config["portfolio"]["max_weight"])
    if method == "risk_parity":
        weights = risk_parity_weights(estimation_returns, max_weight=cap)
    else:
        weights = min_variance_weights(estimation_returns, max_weight=cap)

    portfolio_returns = returns[weights.index].fillna(0.0).mul(weights, axis=1).sum(axis=1)
    annualized = portfolio_returns.mean() * 252
    volatility = portfolio_returns.std() * (252**0.5)
    sharpe = annualized / volatility if volatility > 0 else float("nan")

    print("Portfolio weights")
    print(weights.round(4).to_string())
    print("\nResearch-period diagnostics")
    print(f"Annualized return (arithmetic): {annualized:.2%}")
    print(f"Annualized volatility:           {volatility:.2%}")
    print(f"Sharpe proxy:                    {sharpe:.2f}")


if __name__ == "__main__":
    main()
