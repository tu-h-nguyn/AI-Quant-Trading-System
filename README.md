# AI Quant Trading System

End-to-end quantitative research platform for systematic trading experiments with machine-learning signals, transaction-cost-aware backtesting, portfolio construction, risk analysis, and walk-forward evaluation.

> Research and educational software only. Not investment advice and not intended for live trading.

## Research question

Can machine-learning signals improve risk-adjusted out-of-sample performance over simple systematic baselines after transaction costs?

## Pipeline

`Market data -> features -> signals -> portfolio -> risk controls -> backtest -> walk-forward evaluation`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_backtest.py
python scripts/train_model.py
pytest -q
```

The default configuration uses SPY daily data. Edit `configs/default.yaml` to change symbols, dates, costs, and model parameters.

## Methodological safeguards

- Signals are shifted before next-period returns are applied.
- Train/test splits are chronological rather than randomly shuffled.
- Transaction costs are charged on turnover.
- ML is evaluated against simple baselines.
- Walk-forward evaluation is included before interpreting out-of-sample results.

## Structure

```text
configs/                 experiment configuration
data/                    local datasets (ignored)
notebooks/               research notebooks
scripts/                 reproducible CLI entry points
src/quant_system/
  data/                  download/load
  features/              feature engineering
  strategies/            rule-based signals
  models/                ML training/prediction
  portfolio/             weights and constraints
  risk/                   risk metrics
  backtest/              simulation and costs
  evaluation/            walk-forward analysis
tests/                   unit tests
reports/                 generated figures
```

## Roadmap

1. Multi-asset portfolio construction
2. XGBoost/LightGBM models
3. Volatility targeting and risk parity
4. Slippage and execution assumptions
5. Experiment tracking
6. FastAPI service and dashboard
7. CI/CD
