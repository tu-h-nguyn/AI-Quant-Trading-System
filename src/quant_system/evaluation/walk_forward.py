from __future__ import annotations

from sklearn.base import clone
import pandas as pd


def walk_forward_predict(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_window: int = 63,
    min_train_size: int = 252,
    expanding: bool = True,
    train_window: int = 504,
    gap: int = 0,
) -> pd.Series:
    """Generate strictly OOS probabilities with chronological refits.

    ``gap`` creates an embargo between the end of the training sample and the
    start of the test sample, which is useful when labels span multiple bars.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if test_window <= 0 or min_train_size <= 0 or gap < 0:
        raise ValueError("test_window/min_train_size must be positive and gap non-negative")
    predictions: list[pd.Series] = []
    for start in range(min_train_size + gap, len(X), test_window):
        train_end = start - gap
        end = min(start + test_window, len(X))
        train_start = 0 if expanding else max(0, train_end - train_window)
        fitted = clone(model)
        fitted.fit(X.iloc[train_start:train_end], y.iloc[train_start:train_end])
        p = fitted.predict_proba(X.iloc[start:end])[:, 1]
        predictions.append(pd.Series(p, index=X.index[start:end], name="probability"))
    return pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)
