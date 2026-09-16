from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_table(y_true: pd.Series, probability: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Create equal-width probability calibration bins."""
    y, p = y_true.align(probability, join="inner")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = pd.cut(p.clip(0.0, 1.0), bins=edges, include_lowest=True)
    out = pd.DataFrame({"y": y.astype(float), "p": p.astype(float), "bin": labels}).groupby("bin", observed=False).agg(
        observations=("y", "size"), predicted_probability=("p", "mean"), observed_rate=("y", "mean")
    ).reset_index()
    return out


def calibration_metrics(y_true: pd.Series, probability: pd.Series) -> dict[str, float]:
    """Return Brier score and mean absolute calibration error."""
    y, p = y_true.align(probability, join="inner")
    table = calibration_table(y, p)
    weight = table["observations"] / table["observations"].sum()
    ece = float((weight * (table["predicted_probability"] - table["observed_rate"]).abs().fillna(0)).sum())
    return {"brier_score": float(brier_score_loss(y, p)), "ece": ece}


def threshold_analysis(
    y_true: pd.Series,
    probability: pd.Series,
    future_return: pd.Series | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate classification/trading diagnostics across decision thresholds."""
    y, p = y_true.align(probability, join="inner")
    r = future_return.reindex(y.index) if future_return is not None else None
    values = thresholds if thresholds is not None else np.arange(0.40, 0.71, 0.05)
    rows = []
    for threshold in values:
        signal = p >= float(threshold)
        row = {"threshold": float(threshold), "signal_rate": float(signal.mean())}
        if signal.any():
            row["precision"] = float(y[signal].mean())
            row["mean_future_return_when_active"] = float(r[signal].mean()) if r is not None else np.nan
        else:
            row["precision"] = np.nan
            row["mean_future_return_when_active"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def model_feature_importance(model, feature_names: list[str], X: pd.DataFrame | None = None, y=None) -> pd.DataFrame:
    """Extract native feature importance, or fall back to permutation importance."""
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).reshape(-1))
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
        values = np.asarray(result.importances_mean)
    else:
        raise ValueError("model has no native importance; provide X and y for permutation importance")
    if len(values) != len(feature_names):
        raise ValueError("feature_names length does not match model importance length")
    total = float(values.sum())
    normalized = values / total if total > 0 else values
    return pd.DataFrame({"feature": feature_names, "importance": values, "normalized_importance": normalized}).sort_values("importance", ascending=False).reset_index(drop=True)
