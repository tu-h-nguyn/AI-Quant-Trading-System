from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_equity(strategy: pd.Series, benchmark: pd.Series | None, path: str | Path, initial: float = 1.0) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(initial * (1 + strategy.fillna(0)).cumprod(), label="Strategy")
    if benchmark is not None:
        ax.plot(initial * (1 + benchmark.fillna(0)).cumprod(), label="Benchmark")
    ax.set_title("Equity Curve")
    ax.legend()
    return _save(fig, path)


def plot_drawdown(returns: pd.Series, path: str | Path) -> Path:
    wealth = (1 + returns.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.35)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    return _save(fig, path)


def plot_turnover(turnover: pd.Series, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turnover.index, turnover.values)
    ax.set_title("Portfolio Turnover")
    ax.set_ylabel("Absolute weight turnover")
    return _save(fig, path)


def plot_rolling_sharpe(rolling_sharpe: pd.Series, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Rolling Sharpe")
    return _save(fig, path)


def plot_calibration(table: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    clean = table.dropna(subset=["predicted_probability", "observed_rate"])
    ax.plot(clean["predicted_probability"], clean["observed_rate"], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Probability Calibration")
    ax.legend()
    return _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 12) -> Path:
    data = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["feature"], data["importance"])
    ax.set_title(f"Top {min(top_n, len(importance))} Feature Importance")
    return _save(fig, path)
