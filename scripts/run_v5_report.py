from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from quant_system.backtest.engine import backtest
from quant_system.config import load_config
from quant_system.data.loader import load_symbol_data
from quant_system.evaluation.diagnostics import (
    calibration_metrics,
    calibration_table,
    model_feature_importance,
    threshold_analysis,
)
from quant_system.evaluation.experiment import save_experiment
from quant_system.evaluation.plots import (
    plot_calibration,
    plot_drawdown,
    plot_equity,
    plot_feature_importance,
    plot_rolling_sharpe,
    plot_turnover,
)
from quant_system.evaluation.reporting import build_markdown_report
from quant_system.evaluation.robustness import rolling_sharpe
from quant_system.features.core import build_feature_frame
from quant_system.models.core import probability_to_signal, train_logistic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_cfg, feature_cfg, model_cfg = config["data"], config["features"], config["model"]
    symbol = data_cfg["symbols"][0]
    df = load_symbol_data(symbol, ROOT / "data" / "raw")
    X, y, columns = build_feature_frame(
        df,
        feature_cfg["return_windows"],
        feature_cfg["volatility_windows"],
        feature_cfg["moving_average_fast"],
        feature_cfg["moving_average_slow"],
        model_cfg["horizon"],
    )
    model, probability, _, model_metrics = train_logistic(
        X, y, model_cfg["test_size"], model_cfg["random_state"]
    )
    split = int(len(X) * (1 - model_cfg["test_size"]))
    y_test = y.iloc[split:]
    future_return = df["close"].shift(-model_cfg["horizon"]) / df["close"] - 1
    future_return = future_return.reindex(probability.index)
    strategy = backtest(
        df.loc[probability.index],
        probability_to_signal(probability, model_cfg["threshold"]),
        config["backtest"]["transaction_cost_bps"],
        config["backtest"]["initial_capital"],
    )
    calib_table = calibration_table(y_test, probability)
    calib = calibration_metrics(y_test, probability)
    thresholds = threshold_analysis(y_test, probability, future_return)
    importance = model_feature_importance(model, columns)
    rolling = rolling_sharpe(
        strategy.returns,
        window=min(252, max(20, len(strategy.returns) // 4)),
    )

    artifact_dir = ROOT / "reports" / "figures"
    plot_paths = {
        "equity_curve": plot_equity(
            strategy.returns,
            df["close"].pct_change().reindex(strategy.returns.index),
            artifact_dir / "v5_equity_curve.png",
        ),
        "drawdown": plot_drawdown(strategy.returns, artifact_dir / "v5_drawdown.png"),
        "turnover": plot_turnover(
            strategy.positions.diff().abs().fillna(0), artifact_dir / "v5_turnover.png"
        ),
        "rolling_sharpe": plot_rolling_sharpe(
            rolling, artifact_dir / "v5_rolling_sharpe.png"
        ),
        "calibration": plot_calibration(
            calib_table, artifact_dir / "v5_calibration.png"
        ),
        "feature_importance": plot_feature_importance(
            importance, artifact_dir / "v5_feature_importance.png"
        ),
    }
    report = build_markdown_report(
        ROOT / "reports" / "v5_research_report.md",
        f"V5 Research Report — {symbol} Logistic Signal",
        [symbol],
        model_metrics,
        strategy.metrics,
        calib,
        {k: str(v.relative_to(ROOT)) for k, v in plot_paths.items()},
        [
            "Feature importance is descriptive and does not establish causality.",
            "Threshold analysis is exploratory; thresholds should be evaluated inside time-series validation rather than selected on the final test set.",
            "The benchmark is the underlying buy-and-hold return series for contextual comparison.",
        ],
    )
    record = save_experiment(
        ROOT / "reports" / "experiments",
        "v5_model_diagnostics",
        config,
        {**model_metrics, **strategy.metrics, **calib},
        {
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "symbol": symbol,
            "threshold_rows": thresholds.to_dict(orient="records"),
            "feature_importance": importance.to_dict(orient="records"),
            "report": str(report.relative_to(ROOT)),
        },
    )
    print(f"Saved report: {report.relative_to(ROOT)}")
    print(f"Saved experiment: {record.relative_to(ROOT)}")
    print(pd.Series({**model_metrics, **strategy.metrics, **calib}).round(6).to_string())


if __name__ == "__main__":
    main()
