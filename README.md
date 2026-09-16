# AI Quant Trading System

End-to-end quantitative research platform for systematic trading experiments with machine-learning signals, portfolio construction, risk controls, transaction-cost-aware backtesting, and out-of-sample research.

> Research and educational software only. Not investment advice and not intended for live trading.

## Research question

Can machine-learning signals improve risk-adjusted out-of-sample performance over simple systematic baselines after trading frictions?

## Research architecture

`Market data -> features -> model/rule signals -> portfolio construction -> risk controls -> execution costs -> backtest -> walk-forward OOS -> robustness analysis -> experiment registry`

The system separates research components so models, portfolio methods, and assumptions can be changed without rewriting the backtest layer.

## Current research capabilities

- Multi-asset price-panel alignment and return calculation.
- Buy-and-hold, momentum, and moving-average baselines.
- Logistic Regression and XGBoost signal models.
- Chronological walk-forward evaluation with optional embargo/gap.
- Time-series parameter search using chronological folds.
- Rolling portfolio construction using trailing observations only.
- Long-only constrained minimum-variance and risk-parity optimization.
- Weight caps that remain valid after normalization.
- Volatility targeting, drawdown controls, transaction costs, and turnover.
- Benchmark-relative diagnostics including active return and tracking error.
- Rolling Sharpe, calendar-year subperiod analysis, parameter sensitivity, and moving-block bootstrap uncertainty intervals.
- JSON experiment registry with configuration, metrics, metadata, and reproducibility fields.
- Unit tests and GitHub Actions CI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_backtest.py
python scripts/train_model.py
python scripts/run_multi_asset.py
python scripts/run_research.py
pytest
```

The default research universe is `SPY, QQQ, IWM, TLT, GLD`. Edit `configs/default.yaml` to change symbols, dates, costs, model settings, portfolio constraints, and research windows.

## Methodological safeguards

- Features must be computed from information available at or before each timestamp.
- Training and evaluation are chronological; random shuffling is avoided for time-series experiments.
- Walk-forward predictions are generated from independently fitted estimators.
- An optional gap/embargo prevents training observations immediately adjacent to the test window from being used.
- Portfolio weights are estimated from trailing observations only and lagged before returns are applied.
- Transaction costs are charged on turnover.
- ML signals are compared with simple baselines and benchmark assets.
- Robustness analysis reports subperiod behavior and uncertainty rather than relying on one point estimate.

## Research structure

```text
configs/                       experiment configuration
data/                         local datasets (ignored)
scripts/                      reproducible CLI entry points
src/quant_system/
  data/                       download/load/multi-asset panels
  features/                   feature engineering
  strategies/                 rule-based signals
  models/                     ML training/prediction
  portfolio/                  optimization and rolling weights
  risk/                       volatility and risk controls
  backtest/                   simulation, costs, metrics
  evaluation/                 OOS, time-series splits, robustness, registry
 tests/                        research invariants and regression tests
 reports/experiments/          generated experiment records
 .github/workflows/            CI
```

## Research roadmap

### V4 — Research Engine

- Leakage-aware time-series hyperparameter search.
- Rolling/expanding walk-forward evaluation with embargo support.
- Rolling portfolio construction with lagged weights.
- Benchmark and active-risk diagnostics.
- Subperiod/regime stability checks.
- Moving-block bootstrap uncertainty intervals.
- Parameter sensitivity without cherry-picking a single configuration.
- Reproducible experiment registry.

### V5 — Research Product

- Research notebooks and publication-style report generation.
- Signal probability calibration and threshold analysis.
- Feature importance / model diagnostics.
- FastAPI inference service.
- Dashboard for signals, portfolio, drawdown, turnover, and diagnostics.
- Docker and production-style CI/CD.

## Reproducibility

Research artifacts should record the experiment configuration, timestamp, and source commit when `GIT_COMMIT` is available. Generated datasets and reports are kept out of version control unless explicitly selected for publication.

## Disclaimer

This repository is for research and education. Backtests are historical simulations and can contain model error, estimation error, data issues, and assumptions that differ from real execution. Nothing in this repository constitutes investment advice.
