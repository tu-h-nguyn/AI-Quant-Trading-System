"""Snapshot feature engineering for prediction-market forecasting.

The input is a long panel of market observations -- one row per market per
observation time -- and the output is a chronologically ordered design matrix
whose every column is computed from that row's own past. Per-market time series
are built inside ``groupby`` so one market's history never leaks into another's,
and every backward-looking window is shifted by one observation so the value
being predicted is never an input to its own prediction.

The panel is re-indexed by a monotone ``observation_id`` because several markets
share a timestamp. That keeps the index unique, keeps sort order identical to
chronological order, and therefore lets the existing walk-forward machinery in
:mod:`quant_system.evaluation.walk_forward` be reused unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("market_id", "timestamp", "price")

OBSERVATION_ID = "observation_id"

META_COLUMNS = (
    "market_id",
    "timestamp",
    "price",
    "yes_token_id",
    "no_token_id",
    "question",
    "end_date",
    "label",
)


def build_snapshot_features(
    snapshots: pd.DataFrame,
    momentum_windows: Sequence[int] = (1, 5),
    volatility_window: int = 5,
    min_history: int = 0,
    passthrough_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Attach leakage-safe features to a long panel of market snapshots.

    Required columns are ``market_id``, ``timestamp``, and ``price`` (the
    market's implied YES probability). Optional columns -- ``best_bid``,
    ``best_ask``, ``bid_depth``, ``ask_depth``, ``volume``, ``liquidity``,
    ``end_date`` -- produce extra features when present and are skipped when
    absent, so a thin snapshot source still yields a usable matrix.

    ``passthrough_columns`` promotes caller-supplied numeric columns into the
    design matrix unchanged. Use it for observables computed outside this
    module -- a news-derived score, an external model's output, a venue field
    this package does not parse. They are taken at face value, so it is the
    caller's responsibility that each one was knowable at its row's timestamp;
    nothing here can detect a column that encodes the future.
    """
    missing = set(REQUIRED_COLUMNS) - set(snapshots.columns)
    if missing:
        raise ValueError(f"snapshots is missing required columns: {sorted(missing)}")
    if snapshots.empty:
        return snapshots.copy()

    frame = snapshots.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp", "market_id"], kind="mergesort").reset_index(drop=True)
    frame.index.name = OBSERVATION_ID

    price = frame["price"].astype(float).clip(1e-4, 1 - 1e-4)
    frame["feat_price"] = price
    frame["feat_logit_price"] = np.log(price / (1.0 - price))
    frame["feat_distance_from_half"] = (price - 0.5).abs()

    if {"best_bid", "best_ask"}.issubset(frame.columns):
        bid = frame["best_bid"].astype(float)
        ask = frame["best_ask"].astype(float)
        spread = (ask - bid).clip(lower=0.0)
        frame["feat_spread"] = spread
        frame["feat_relative_spread"] = spread / price

    if {"bid_depth", "ask_depth"}.issubset(frame.columns):
        bid_depth = frame["bid_depth"].astype(float).clip(lower=0.0)
        ask_depth = frame["ask_depth"].astype(float).clip(lower=0.0)
        total = bid_depth + ask_depth
        frame["feat_book_imbalance"] = np.where(total > 0, (bid_depth - ask_depth) / total, 0.0)
        frame["feat_log_depth"] = np.log1p(total)

    for column, name in (("volume", "feat_log_volume"), ("liquidity", "feat_log_liquidity")):
        if column in frame.columns:
            frame[name] = np.log1p(frame[column].astype(float).clip(lower=0.0))

    if "end_date" in frame.columns:
        end = pd.to_datetime(frame["end_date"], utc=True)
        days = (end - frame["timestamp"]).dt.total_seconds() / 86_400.0
        frame["feat_days_to_resolution"] = days.clip(lower=0.0)
        frame["feat_log_days_to_resolution"] = np.log1p(frame["feat_days_to_resolution"])
        frame["feat_logit_price_per_day"] = frame["feat_logit_price"] / (
            1.0 + frame["feat_days_to_resolution"]
        )

    for column in passthrough_columns:
        if column not in frame.columns:
            raise ValueError(f"passthrough column {column!r} is not present in snapshots")
        frame[f"feat_{column}"] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    grouped = frame.groupby("market_id", sort=False)["feat_logit_price"]
    for window in momentum_windows:
        if window < 1:
            raise ValueError("momentum windows must be >= 1")
        frame[f"feat_momentum_{window}"] = grouped.diff(window)
    if volatility_window >= 2:
        frame[f"feat_volatility_{volatility_window}"] = grouped.transform(
            lambda series: series.diff().rolling(volatility_window).std()
        )
    frame["feat_observation_number"] = grouped.cumcount().astype(float)

    # Everything derived from the market's own history is shifted one step, so a
    # feature can only ever describe the state strictly before the decision.
    lagged = [column for column in frame.columns if column.startswith(("feat_momentum_",
                                                                       "feat_volatility_"))]
    if lagged:
        frame[lagged] = frame.groupby("market_id", sort=False)[lagged].shift(1)

    if min_history > 0:
        frame = frame[frame["feat_observation_number"] >= float(min_history)]

    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Names of the generated feature columns, in a stable order."""
    return sorted(column for column in frame.columns if column.startswith("feat_"))


def design_matrix(
    frame: pd.DataFrame,
    label_column: str = "label",
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a feature frame into ``(X, y, metadata)`` with aligned indices.

    Rows carrying a missing feature or label are dropped when ``dropna`` is set,
    which is the only supported mode for model fitting; imputing a warm-up
    window would fabricate history the market did not have.
    """
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("frame contains no generated feature columns")
    if label_column not in frame.columns:
        raise ValueError(f"frame is missing the label column {label_column!r}")

    X = frame[columns].replace([np.inf, -np.inf], np.nan)
    y = frame[label_column]
    if dropna:
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[valid], y.loc[valid]
    meta = frame.loc[X.index, [c for c in META_COLUMNS if c in frame.columns]]
    return X, y.astype(int), meta
