# AI Quant Trading System

End-to-end quantitative research platform for systematic trading experiments with machine-learning signals, portfolio construction, risk controls, transaction-cost-aware backtesting, and out-of-sample research.

> Research and educational software only. Not investment advice and not intended for live trading.

## Research question

Can a systematic signal improve risk-adjusted out-of-sample performance after trading frictions, while remaining stable across time and modeling assumptions?

## Research architecture

`Market data -> features -> model/rule signals -> portfolio construction -> risk controls -> execution costs -> backtest -> walk-forward OOS -> robustness -> diagnostics -> report -> experiment registry`

The system separates research components so models, portfolio methods, validation schemes, and assumptions can be changed without rewriting the backtest layer.

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
- Probability calibration, threshold diagnostics, and model feature-importance analysis.
- Reproducible equity, drawdown, turnover, rolling-Sharpe, calibration, and feature-importance figures.
- Publication-style Markdown research report generation.
- JSON experiment registry with configuration, metrics, metadata, and reproducibility fields.
- Unit tests and GitHub Actions CI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_research.py
python scripts/run_v5_report.py
pytest
```

The default research universe is `SPY, QQQ, IWM, TLT, GLD`. Edit `configs/default.yaml` to change symbols, dates, costs, model settings, portfolio constraints, and research windows.

## V5 evidence layer

`python scripts/run_v5_report.py` produces a complete model-evaluation artifact for the first symbol in the configured universe:

- out-of-sample classification metrics;
- probability calibration table, Brier score, and ECE;
- threshold sensitivity table;
- native feature importance for the fitted model;
- strategy equity and drawdown curves;
- turnover and rolling-Sharpe diagnostics;
- a Markdown report under `reports/v5_research_report.md`;
- a machine-readable experiment record under `reports/experiments/`.

The diagnostics are deliberately descriptive. Thresholds and feature rankings are not treated as proof of predictive causality and should not be selected from the final test set without validation.

## Methodological safeguards

- Features must be computed from information available at or before each timestamp.
- Training and evaluation are chronological; random shuffling is avoided for time-series experiments.
- Walk-forward predictions are generated from independently fitted estimators.
- An optional gap/embargo prevents training observations immediately adjacent to the test window from being used.
- Portfolio weights are estimated from trailing observations only and lagged before returns are applied.
- Transaction costs are charged on turnover.
- ML signals are compared with simple baselines and benchmark assets.
- Robustness analysis reports subperiod behavior and uncertainty rather than relying on one point estimate.
- Research artifacts record the source commit when `GIT_COMMIT` is available.

## Research structure

```text
configs/                       experiment configuration
data/                          local datasets (ignored)
scripts/                       reproducible CLI entry points
src/quant_system/
  data/                        download/load/multi-asset panels
  features/                    feature engineering
  models/                      ML training/prediction
  portfolio/                   optimization and rolling weights
  risk/                        volatility and risk controls
  backtest/                    simulation, costs, metrics
  evaluation/                  OOS, splits, robustness, diagnostics, reports
 tests/                        research invariants and regression tests
 reports/experiments/          generated experiment records
 reports/figures/              generated figures
 .github/workflows/            CI
```

## Roadmap

### V4 — Research Engine

- Leakage-aware time-series validation.
- Rolling/expanding walk-forward evaluation.
- Rolling portfolio construction.
- Benchmark-relative and subperiod diagnostics.
- Block-bootstrap uncertainty analysis.
- Parameter sensitivity.
- Reproducible experiment registry.

### V5 — Evidence / Research Product

- Probability calibration and threshold diagnostics.
- Model feature-importance diagnostics.
- Reproducible research figures.
- Publication-style report generation.

### V6 — Research-to-Engineering

- FastAPI inference service.
- Dashboard for signals, portfolio, drawdown, turnover, and diagnostics.
- Dockerized workflows.
- Production-style CI/CD and scheduled research runs.

## Reproducibility

Research outputs should record configuration, timestamp, and source commit. Generated datasets and figures are excluded from version control unless explicitly selected for publication.

## Disclaimer

This repository is for research and education. Backtests are historical simulations and can contain model error, estimation error, data issues, and assumptions that differ from real execution. Nothing in this repository constitutes investment advice.
