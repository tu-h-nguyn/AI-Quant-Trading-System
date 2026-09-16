"""Forecast-quality and bankroll diagnostics for prediction-market trading.

Accuracy is the wrong yardstick here. A market quoted at 0.90 that resolves YES
90% of the time is perfectly efficient, and a model that "predicts" those
outcomes at 90% accuracy has no edge whatsoever. The question is always whether
a forecast is better *than the price*, which is what
:func:`brier_skill_score` measures against the market's own implied probability.

The second question is whether a measured forecast edge survives contact with
execution. :func:`edge_realization` regresses realized profit per share on the
edge that was predicted before the trade: a slope near one means the edge was
real, and a slope near zero means it was noise that costs money to act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import max_drawdown
from ..evaluation.diagnostics import calibration_metrics

_EPSILON = 1e-12


def _aligned(y_true: pd.Series, probability: pd.Series) -> tuple[pd.Series, pd.Series]:
    y, p = pd.Series(y_true).align(pd.Series(probability), join="inner")
    mask = y.notna() & p.notna()
    return y[mask].astype(float), p[mask].astype(float).clip(_EPSILON, 1 - _EPSILON)


def brier_score(y_true: pd.Series, probability: pd.Series) -> float:
    """Mean squared error of a probabilistic forecast; lower is better."""
    y, p = _aligned(y_true, probability)
    if y.empty:
        return float("nan")
    return float(((p - y) ** 2).mean())


def log_loss_score(y_true: pd.Series, probability: pd.Series) -> float:
    """Mean negative log-likelihood of a probabilistic forecast."""
    y, p = _aligned(y_true, probability)
    if y.empty:
        return float("nan")
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_skill_score(
    y_true: pd.Series,
    probability: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Fractional Brier improvement over a benchmark forecast.

    Pass the market's implied probability as ``benchmark``. Positive means the
    model carries information the price does not; zero or negative means it does
    not, which is the expected result for an efficient market and is the honest
    null this project starts from.
    """
    y, p = _aligned(y_true, probability)
    y_bench, b = _aligned(y_true, benchmark)
    index = y.index.intersection(y_bench.index)
    if len(index) == 0:
        return float("nan")
    model = float(((p[index] - y[index]) ** 2).mean())
    reference = float(((b[index] - y[index]) ** 2).mean())
    if reference <= _EPSILON:
        return float("nan")
    return 1.0 - model / reference


def forecast_report(
    y_true: pd.Series,
    probability: pd.Series,
    market_price: pd.Series,
) -> dict[str, float]:
    """Full forecast scorecard: absolute quality plus skill versus the price."""
    y, p = _aligned(y_true, probability)
    calibration = calibration_metrics(y, p) if not y.empty else {"brier_score": float("nan"),
                                                                 "ece": float("nan")}
    return {
        "n_observations": float(len(y)),
        "base_rate": float(y.mean()) if not y.empty else float("nan"),
        "brier_score": brier_score(y, p),
        "market_brier_score": brier_score(y_true, market_price),
        "brier_skill_vs_market": brier_skill_score(y_true, p, market_price),
        "log_loss": log_loss_score(y, p),
        "market_log_loss": log_loss_score(y_true, market_price),
        "calibration_error": float(calibration["ece"]),
    }


def edge_realization(
    predicted_edge: pd.Series,
    realized_profit_per_share: pd.Series,
) -> dict[str, float]:
    """Regress realized profit per share on the edge predicted beforehand.

    ``slope`` is the diagnostic that matters: one means a predicted cent of edge
    delivered a cent, and zero means the predictions carried no information
    about outcomes.

    Always read ``slope`` next to ``slope_std_error``. An edge gate clusters
    every accepted trade at nearly the same predicted edge, so the regressor has
    almost no spread while the binary payoff swings by a full dollar; the slope
    is then enormous and meaningless. The standard error makes that visible
    instead of leaving a spurious number to be quoted.
    """
    x, y = pd.Series(predicted_edge).align(pd.Series(realized_profit_per_share), join="inner")
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(float), y[mask].astype(float)
    if len(x) < 3 or float(x.var(ddof=1)) <= _EPSILON:
        return {
            "slope": float("nan"),
            "slope_std_error": float("nan"),
            "slope_t_stat": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
            "n_trades": float(len(x)),
            "mean_predicted_edge": float(x.mean()) if len(x) else float("nan"),
            "mean_realized_edge": float(y.mean()) if len(y) else float("nan"),
        }
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(((y - fitted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    spread = float(((x - x.mean()) ** 2).sum())
    std_error = (
        float(np.sqrt(residual / (len(x) - 2) / spread)) if spread > _EPSILON else float("nan")
    )
    return {
        "slope": float(slope),
        "slope_std_error": std_error,
        "slope_t_stat": float(slope) / std_error if std_error and np.isfinite(std_error) and
        std_error > _EPSILON else float("nan"),
        "intercept": float(intercept),
        "r_squared": 1.0 - residual / total if total > _EPSILON else float("nan"),
        "n_trades": float(len(x)),
        "mean_predicted_edge": float(x.mean()),
        "mean_realized_edge": float(y.mean()),
    }


def trade_summary(trades: pd.DataFrame) -> dict[str, float]:
    """Aggregate a settled trade ledger into economic diagnostics.

    ``per_trade_sharpe`` is deliberately *not* annualized. Prediction-market
    trades have heterogeneous holding periods and a lumpy binary payoff, so a
    252-day scaling would manufacture a number with no defensible frequency.
    """
    if trades.empty:
        return {
            "n_trades": 0.0,
            "total_profit": 0.0,
            "capital_deployed": 0.0,
            "roi_on_capital": float("nan"),
            "hit_rate": float("nan"),
            "mean_trade_return": float("nan"),
            "per_trade_sharpe": float("nan"),
            "mean_holding_days": float("nan"),
        }
    profit = trades["profit"].astype(float)
    capital = trades["capital"].astype(float)
    trade_return = profit / capital.replace(0.0, np.nan)
    deployed = float(capital.sum())
    holding = (
        trades["holding_days"].astype(float)
        if "holding_days" in trades.columns
        else pd.Series(dtype=float)
    )
    return {
        "n_trades": float(len(trades)),
        "total_profit": float(profit.sum()),
        "capital_deployed": deployed,
        "roi_on_capital": float(profit.sum() / deployed) if deployed > 0 else float("nan"),
        "hit_rate": float((profit > 0).mean()),
        "mean_trade_return": float(trade_return.mean()),
        "per_trade_sharpe": float(trade_return.mean() / trade_return.std(ddof=1))
        if trade_return.std(ddof=1) > _EPSILON
        else float("nan"),
        "mean_holding_days": float(holding.mean()) if not holding.empty else float("nan"),
    }


def bankroll_summary(equity: pd.Series) -> dict[str, float]:
    """Path diagnostics for a bankroll curve sampled at settlement events."""
    curve = pd.Series(equity).astype(float).dropna()
    if len(curve) < 2:
        return {
            "final_bankroll": float(curve.iloc[-1]) if len(curve) else float("nan"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "n_settlements": float(len(curve)),
        }
    steps = curve.pct_change().dropna()
    return {
        "final_bankroll": float(curve.iloc[-1]),
        "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1.0),
        "max_drawdown": float(max_drawdown(steps)),
        "n_settlements": float(len(curve)),
    }
