# AI Quant Trading System

End-to-end quantitative research platform for systematic trading experiments with machine-learning signals, portfolio construction, risk controls, transaction-cost-aware backtesting, and walk-forward evaluation.

> Research and educational software only. Not investment advice and not intended for live trading.

## Research question

Can machine-learning signals improve risk-adjusted out-of-sample performance over simple systematic baselines after realistic trading frictions?

## Architecture

`Market data -> feature engineering -> ML/rule signals -> portfolio optimization -> risk controls -> execution model -> backtest -> OOS evaluation`

The project is deliberately split into replaceable research components so that a strategy, model, or portfolio method can be tested without rewriting the whole system.

## Current capabilities

- Multi-asset price-panel alignment and return calculation.
- Rule-based baselines: buy-and-hold, momentum, moving-average crossover.
- Leakage-aware chronological train/test evaluation.
- Logistic Regression and XGBoost signal models.
- Constrained minimum-variance and risk-parity portfolio weights.
- Transaction costs and position turnover in the backtest engine.
- Volatility targeting and drawdown guardrails.
- Expanding or rolling walk-forward refits with independent model instances.
- Unit tests for key research invariants and GitHub Actions CI.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_backtest.py
python scripts/train_model.py
pytest
```

The default configuration uses SPY daily data. Edit `configs/default.yaml` to change symbols, dates, costs, and model parameters.

## Methodological safeguards

- Signals are shifted before next-period returns are applied.
- Features use only information available at or before each timestamp.
- Train/test splits are chronological rather than randomly shuffled.
- Portfolio covariance estimates must be formed from historical observations only when used in research experiments.
- Transaction costs are charged on position changes.
- ML models are compared against simple baselines.
- Walk-forward evaluation is available before interpreting out-of-sample performance.

## Structure

```text
configs/                 experiment configuration
data/                    local datasets (ignored)
scripts/                 reproducible CLI entry points
src/quant_system/
  data/                  download/load/multi-asset panels
  features/              feature engineering
  strategies/            rule-based signals
  models/                ML training/prediction
  portfolio/             weights and optimization
  risk/                   volatility and risk controls
  backtest/              simulation, costs, metrics
  evaluation/            walk-forward OOS evaluation
tests/                   unit tests
reports/                 generated research outputs
.github/workflows/       CI
```

## Research roadmap

### V2 — Research engine
- Multi-asset experiments
- XGBoost signal model
- Portfolio optimization
- Volatility targeting
- Walk-forward OOS evaluation

### V3 — Robustness
- Slippage and spread assumptions
- Parameter sensitivity grids
- Subperiod and regime analysis
- Bootstrap confidence intervals
- Probability calibration and threshold analysis

### V4 — Research product
- Experiment registry and reproducible result artifacts
- Research notebooks and report generation
- FastAPI inference service
- Dashboard for signals, portfolio, drawdown, and diagnostics
- Docker and production-style CI/CD
