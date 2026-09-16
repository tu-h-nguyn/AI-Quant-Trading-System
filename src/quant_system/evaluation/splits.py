from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def make_time_series_splitter(n_splits: int = 5, gap: int = 0) -> TimeSeriesSplit:
    """Create a chronological splitter with an optional embargo/gap."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def parameter_grid_search(
    estimator,
    X,
    y,
    param_grid: dict,
    scoring,
    n_splits: int = 5,
    gap: int = 0,
):
    """Leakage-aware grid search over chronological folds.

    The estimator is cloned for every fold/parameter set. ``scoring`` receives
    ``(estimator, X_test, y_test)`` and must return a scalar score.
    """
    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid

    X = np.asarray(X) if not hasattr(X, "iloc") else X
    y = np.asarray(y) if not hasattr(y, "iloc") else y
    splitter = make_time_series_splitter(n_splits=n_splits, gap=gap)
    rows = []
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, test_idx in splitter.split(X):
            fitted = clone(estimator).set_params(**params)
            fitted.fit(X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx],
                       y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx])
            X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_test = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
            fold_scores.append(float(scoring(fitted, X_test, y_test)))
        rows.append({**params, "mean_score": float(np.mean(fold_scores)),
                     "std_score": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                     "n_folds": len(fold_scores)})
    return rows
