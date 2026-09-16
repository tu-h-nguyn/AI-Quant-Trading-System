from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from quant_system.backtest.engine import backtest
from quant_system.backtest.metrics import summary
from quant_system.config import load_config
from quant_system.data.loader import load_symbol_data
from quant_system.evaluation.experiment import save_experiment
from quant_system.evaluation.robustness import block_bootstrap_mean, percentile_interval
from quant_system.evaluation.walk_forward import walk_forward_predict
from quant_system.features.core import build_feature_frame
from quant_system.models.core import probability_to_signal

ROOT = Path(__file__).resolve().parents[1]


def _models(random_state: int) -> dict[str, Pipeline]:
    return {
        "Logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _auc(y: pd.Series, p: pd.Series) -> float:
    from sklearn.metrics import roc_auc_score

    aligned_y, aligned_p = y.align(p, join="inner")
    return float(roc_auc_score(aligned_y, aligned_p)) if aligned_y.nunique() > 1 else float("nan")


def _run_signal_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    config: dict,
) -> tuple[dict, pd.Series, pd.Series]:
    result = backtest(
        df.loc[signal.index],
        signal,
        float(config["backtest"]["transaction_cost_bps"]),
        float(config["backtest"]["initial_capital"]),
    )
    return result.metrics, result.returns, result.positions


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    data_cfg, feature_cfg, model_cfg, research_cfg = (
        config["data"],
        config["features"],
        config["model"],
        config["research"],
    )

    rows: list[dict] = []
    return_series: dict[tuple[str, str], pd.Series] = {}

    for symbol in data_cfg["symbols"]:
        df = load_symbol_data(symbol, ROOT / "data" / "raw")
        X, y, _ = build_feature_frame(
            df,
            feature_cfg["return_windows"],
            feature_cfg["volatility_windows"],
            feature_cfg["moving_average_fast"],
            feature_cfg["moving_average_slow"],
            model_cfg["horizon"],
        )
        oos_start = max(
            int(research_cfg["walk_forward_min_train_size"]) + int(research_cfg["walk_forward_gap"]),
            1,
        )
        prediction_start = X.index[oos_start] if len(X) > oos_start else X.index[-1]
        oos_index = X.index[X.index >= prediction_start]

        baselines = {
            "Buy & Hold": pd.Series(1.0, index=oos_index, name="signal"),
            "Momentum": (X.loc[oos_index, "return_20d"] > 0).astype(float).rename("signal"),
            "MA Crossover": (X.loc[oos_index, "ma_ratio"] > 0).astype(float).rename("signal"),
        }
        for strategy, signal in baselines.items():
            metrics, returns, positions = _run_signal_backtest(df, signal, config)
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy,
                    "model_auc": float("nan"),
                    **metrics,
                }
            )
            return_series[(symbol, strategy)] = returns

        for strategy, model in _models(int(model_cfg["random_state"])).items():
            probability = walk_forward_predict(
                model,
                X,
                y,
                test_window=int(research_cfg["walk_forward_test_window"]),
                min_train_size=int(research_cfg["walk_forward_min_train_size"]),
                gap=int(research_cfg["walk_forward_gap"]),
            )
            signal = probability_to_signal(probability, float(model_cfg["threshold"]))
            metrics, returns, positions = _run_signal_backtest(df, signal, config)
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy,
                    "model_auc": _auc(y.reindex(probability.index), probability),
                    **metrics,
                }
            )
            return_series[(symbol, strategy)] = returns

    detail = pd.DataFrame(rows).sort_values(["strategy", "symbol"]).reset_index(drop=True)

    aggregate_rows: list[dict] = []
    for strategy in detail["strategy"].unique():
        series = [r for (symbol, name), r in return_series.items() if name == strategy]
        if not series:
            continue
        portfolio_returns = pd.concat(series, axis=1).mean(axis=1).sort_index().fillna(0.0)
        metrics = summary(portfolio_returns)
        bootstrap = block_bootstrap_mean(
            portfolio_returns,
            n_bootstrap=int(research_cfg["bootstrap_samples"]),
            block_size=int(research_cfg["bootstrap_block"]),
            seed=int(model_cfg["random_state"]),
        )
        ci_low, ci_high = percentile_interval(bootstrap)
        aggregate_rows.append(
            {
                "strategy": strategy,
                **metrics,
                "mean_daily_return_bootstrap_ci_low": float(ci_low),
                "mean_daily_return_bootstrap_ci_high": float(ci_high),
                "n_assets": len(series),
                "n_oos_observations": int(len(portfolio_returns)),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows)
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    detail_path = report_dir / "flagship_results.csv"
    aggregate_path = report_dir / "flagship_aggregate_results.csv"
    detail.to_csv(detail_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    lines = [
        "# Flagship AI Quant Research Study",
        "",
        "> Research question: Does a leakage-aware machine-learning signal add predictive and risk-adjusted value out of sample after transaction costs, relative to simple trading rules?",
        "",
        "## Experimental design",
        "",
        "- Universe: SPY, QQQ, IWM, TLT, GLD.",
        "- Frequency: daily observations.",
        "- Features: lagged returns, rolling volatility, moving-average structure, and volume change.",
        f"- Label horizon: {model_cfg['horizon']} trading days.",
        f"- OOS evaluation: walk-forward refits every {research_cfg['walk_forward_test_window']} days with a {research_cfg['walk_forward_gap']}-day embargo.",
        f"- Minimum training history: {research_cfg['walk_forward_min_train_size']} observations.",
        f"- Transaction cost: {config['backtest']['transaction_cost_bps']} bps per unit turnover.",
        "- Threshold is fixed ex ante at the configured value; it is not optimized on the OOS sample.",
        "- Reported aggregate portfolio is the equal-weight average of per-asset strategy returns.",
        "",
        "## Aggregate OOS results",
        "",
        "| Strategy | CAGR | Sharpe | Sortino | Max Drawdown | Turnover | OOS observations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['strategy']} | {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['sortino']:.3f} | {row['max_drawdown']:.2%} | {row['turnover']:.2f} | {row['n_oos_observations']} |"
        )

    lines += [
        "",
        "## Per-asset results",
        "",
        "| Symbol | Strategy | AUC | CAGR | Sharpe | Sortino | Max Drawdown | Turnover |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detail.to_dict(orient="records"):
        auc = "—" if pd.isna(row["model_auc"]) else f"{row['model_auc']:.3f}"
        lines.append(
            f"| {row['symbol']} | {row['strategy']} | {auc} | {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['sortino']:.3f} | {row['max_drawdown']:.2%} | {row['turnover']:.2f} |"
        )

    lines += [
        "",
        "## Statistical diagnostic",
        "",
        "The aggregate daily-return mean is bootstrapped with moving blocks to preserve short-range dependence. The interval is descriptive rather than a guarantee of future performance.",
        "",
        "## Reproducibility",
        "",
        "The exact configuration is stored in `configs/default.yaml`; the experiment records the Git commit and output CSVs are generated by `scripts/run_flagship_study.py`.",
        "",
        "## Caveats",
        "",
        "- This is historical backtesting, not evidence of future profitability.",
        "- Yahoo Finance data can be revised or differ from institutional feeds.",
        "- Equal-weight aggregation is intentionally simple and is not an optimized portfolio construction layer.",
        "- Multiple-model experimentation creates researcher degrees of freedom; future work should pre-register hypotheses before tuning.",
    ]
    report_path = report_dir / "flagship_research_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    record = save_experiment(
        report_dir / "experiments",
        "flagship_oos_study",
        config,
        {"n_rows": len(detail), "n_strategies": int(detail["strategy"].nunique())},
        {
            "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            "detail_csv": str(detail_path.relative_to(ROOT)),
            "aggregate_csv": str(aggregate_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    )
    print(report_path.relative_to(ROOT))
    print(detail.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(record.relative_to(ROOT))


if __name__ == "__main__":
    main()
