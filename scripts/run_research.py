from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from quant_system.backtest.metrics import summary
from quant_system.data.loader import load_symbol_data
from quant_system.data.panel import build_price_panel, returns_panel
from quant_system.evaluation.benchmark import benchmark_returns, relative_metrics
from quant_system.evaluation.experiment import save_experiment
from quant_system.evaluation.robustness import percentile_interval, block_bootstrap_mean, subperiod_summary
from quant_system.portfolio.optimization import min_variance_weights, risk_parity_weights
from quant_system.portfolio.rolling import apply_weights, rolling_weight_schedule

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    symbols = config["data"]["symbols"]
    frames = {s: load_symbol_data(s, ROOT / "data" / "raw") for s in symbols}
    prices = build_price_panel(frames)
    returns = returns_panel(prices).dropna(how="all").dropna(how="all")

    method = config["portfolio"]["method"]
    cap = float(config["portfolio"]["max_weight"])
    lookback = int(config["research"]["portfolio_lookback"])
    rebalance = int(config["research"]["rebalance_every"])
    benchmark = config["research"]["benchmark"]

    optimizer = lambda x: (
        risk_parity_weights(x.dropna(), max_weight=cap)
        if method == "risk_parity"
        else min_variance_weights(x.dropna(), max_weight=cap)
    )
    weights = rolling_weight_schedule(returns, optimizer, lookback, rebalance)
    strategy_returns, turnover, _ = apply_weights(
        returns, weights, float(config["backtest"]["transaction_cost_bps"])
    )
    strategy_metrics = summary(strategy_returns)
    strategy_metrics["turnover"] = float(turnover.sum())

    bench = benchmark_returns(returns, benchmark).reindex(strategy_returns.index).fillna(0.0)
    strategy_metrics.update({f"benchmark_relative_{k}": v for k, v in relative_metrics(strategy_returns, bench).items()})

    bootstrap = block_bootstrap_mean(
        strategy_returns,
        n_bootstrap=int(config["research"]["bootstrap_samples"]),
        block_size=int(config["research"]["bootstrap_block"]),
        seed=int(config["model"]["random_state"]),
    )
    ci_low, ci_high = percentile_interval(bootstrap)
    strategy_metrics["mean_daily_return_bootstrap_ci_low"] = ci_low
    strategy_metrics["mean_daily_return_bootstrap_ci_high"] = ci_high

    record = save_experiment(
        ROOT / "reports" / "experiments",
        "v4_rolling_portfolio",
        config,
        strategy_metrics,
        {
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "observations": len(strategy_returns),
            "subperiods": subperiod_summary(strategy_returns).to_dict(orient="records"),
        },
    )
    print(f"Saved experiment: {record.relative_to(ROOT)}")
    print(pd.Series(strategy_metrics).round(6).to_string())


if __name__ == "__main__":
    main()
