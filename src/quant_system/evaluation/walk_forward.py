from __future__ import annotations

from copy import deepcopy

import pandas as pd


def walk_forward_predict(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_window: int = 63,
    min_train_size: int = 252,
    expanding: bool = True,
    train_window: int = 504,
) -> pd.Series:
    """Generate strictly out-of-sample probabilities with periodic independent refits."""
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if test_window <= 0 or min_train_size <= 0:
        raise ValueError("test_window and min_train_size must be positive")

    predictions: list[pd.Series] = []
    for start in range(min_train_size, len(X), test_window):
        end = min(start + test_window, len(X))
        train_start = 0 if expanding else max(0, start - train_window)
        fitted = deepcopy(model)
        fitted.fit(X.iloc[train_start:start], y.iloc[train_start:start])
        p = fitted.predict_proba(X.iloc[start:end])[:, 1]
        predictions.append(pd.Series(p, index=X.index[start:end], name="probability"))

    return pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)
