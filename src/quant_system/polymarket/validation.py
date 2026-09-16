"""Resolution-aware out-of-sample splitting for prediction markets.

Row-wise walk-forward is correct when every row carries its own label. A
prediction-market panel breaks that assumption twice over:

* every snapshot of one market shares a single label, so splitting by row puts
  the *same* outcome on both sides of the boundary and the model is scored on
  markets it has already memorized;
* a label is not knowable until the market settles, so training on a market
  that resolves next month is training on information the trader did not have.

Both are fixed by the same rule, which is also the live constraint: at each
decision point, train only on markets that had **already settled** by then.
That makes group leakage impossible by construction -- a settled market's
observations all precede its settlement, so it cannot appear in a later test
block -- without needing a separate grouping pass.

The price of the rule is a long cold start: nothing can be predicted until
enough markets have resolved. That cost is real and is reported rather than
engineered away.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone


def resolution_aware_folds(
    timestamps: pd.Series,
    resolution_times: pd.Series,
    test_window: int = 500,
    min_train_size: int = 1000,
    embargo_days: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_positions, test_positions)`` for each chronological block.

    ``timestamps`` and ``resolution_times`` must be aligned with the design
    matrix and ordered by observation time. Blocks whose training set is too
    small are skipped, so the caller sees no prediction rather than one made
    from an inadequate history.
    """
    if test_window <= 0 or min_train_size <= 0:
        raise ValueError("test_window and min_train_size must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")

    observed = pd.to_datetime(timestamps, utc=True)
    resolved = pd.to_datetime(resolution_times, utc=True)
    if not observed.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")

    embargo = pd.Timedelta(days=float(embargo_days))
    n = len(observed)
    for start in range(0, n, test_window):
        stop = min(start + test_window, n)
        cutoff = observed.iloc[start] - embargo
        # Compared through pandas rather than numpy so that tz-aware and
        # tz-naive panels both work without silently shifting by the offset.
        train = np.flatnonzero((resolved <= cutoff).to_numpy())
        if len(train) < min_train_size:
            continue
        yield train, np.arange(start, stop)


def resolution_aware_walk_forward(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    resolution_times: pd.Series,
    test_window: int = 500,
    min_train_size: int = 1000,
    embargo_days: float = 0.0,
    sample_weight: pd.Series | None = None,
) -> pd.Series:
    """Out-of-sample probabilities using only markets settled before each block.

    The estimator is cloned per fold, so no fitted state survives from one block
    to the next. Folds whose training sample is single-class are skipped, since
    a classifier fitted on one class produces a constant probability that would
    otherwise be scored as a forecast.

    ``sample_weight`` is sliced per fold and forwarded to estimators that accept
    it. Pass the reciprocal of each market's observation count to stop a market
    quoted a hundred times from counting as a hundred independent outcomes; the
    row count of a prediction-market panel overstates its information by exactly
    that factor.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if sample_weight is not None and len(sample_weight) != len(X):
        raise ValueError("sample_weight must have the same length as X")

    predictions: list[pd.Series] = []
    y_values = np.asarray(y)
    for train, test in resolution_aware_folds(
        timestamps, resolution_times, test_window, min_train_size, embargo_days
    ):
        if len(np.unique(y_values[train])) < 2:
            continue
        fitted = clone(model)
        if sample_weight is None:
            fitted.fit(X.iloc[train], y.iloc[train])
        else:
            fitted.fit(X.iloc[train], y.iloc[train], sample_weight=sample_weight.iloc[train])
        probability = fitted.predict_proba(X.iloc[test])[:, 1]
        predictions.append(
            pd.Series(probability, index=X.index[test], name="probability")
        )
    if not predictions:
        return pd.Series(dtype=float, name="probability")
    return pd.concat(predictions).sort_index()


def fold_diagnostics(
    timestamps: pd.Series,
    resolution_times: pd.Series,
    test_window: int = 500,
    min_train_size: int = 1000,
    embargo_days: float = 0.0,
) -> pd.DataFrame:
    """Per-fold training size and coverage, for reporting the cold start."""
    rows = []
    observed = pd.to_datetime(timestamps, utc=True)
    for train, test in resolution_aware_folds(
        timestamps, resolution_times, test_window, min_train_size, embargo_days
    ):
        rows.append(
            {
                "test_start": observed.iloc[test[0]],
                "test_end": observed.iloc[test[-1]],
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    return pd.DataFrame(rows)


def achievable_brier_skill(
    y_true: pd.Series,
    true_probability: pd.Series,
    market_price: pd.Series,
) -> float:
    """Brier skill an oracle would score, given the true probabilities.

    Brier score on binary outcomes is dominated by the irreducible variance
    ``q(1 - q)``, so even perfect knowledge of ``q`` yields a skill score of a
    few thousandths against a roughly efficient price. Without this ceiling a
    reader has no way to tell a model capturing most of the available signal
    from one capturing none, because both look like a number near zero.

    Only computable where the truth is known, which means simulation.
    """
    from .metrics import brier_skill_score

    return brier_skill_score(y_true, true_probability, market_price)
