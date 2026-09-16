"""Characterization tests for the original single-asset stack.

These modules carried the whole equity study but had no direct coverage: the
suite exercised the portfolio and evaluation layers and reached the feature and
metric code only incidentally. They were reformatted from dense one-line
definitions, which is exactly the kind of change that can alter behaviour
without looking like it does, so the properties that matter are pinned here.
"""

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest.metrics import (
    cagr,
    equity_curve,
    max_drawdown,
    sharpe,
    sortino,
    summary,
    volatility,
)
from quant_system.features.core import (
    add_return_features,
    add_technical_features,
    add_volatility_features,
    build_feature_frame,
)
from quant_system.models.core import chronological_split, probability_to_signal, train_logistic
from quant_system.strategies.baselines import buy_and_hold, momentum, moving_average


def _prices(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=idx,
    )


def test_return_features_are_backward_looking():
    df = _prices(50)
    out = add_return_features(df, (1, 5))
    assert out["return_5d"].iloc[:5].isna().all()
    # A 5-day return must equal the 5-day price ratio, not a forward one.
    expected = df["close"].iloc[9] / df["close"].iloc[4] - 1
    assert out["return_5d"].iloc[9] == pytest.approx(expected)


def test_volatility_features_need_a_full_window():
    out = add_volatility_features(_prices(50), (5,))
    assert out["volatility_5d"].iloc[:5].isna().all()
    assert out["volatility_5d"].iloc[5:].notna().all()
    assert (out["volatility_5d"].dropna() >= 0).all()


def test_moving_average_ratio_is_signed_around_the_crossover():
    out = add_technical_features(_prices(120), 5, 20)
    crossed_up = out["ma_fast"] > out["ma_slow"]
    assert (out.loc[crossed_up, "ma_ratio"] > 0).all()
    assert (out.loc[~crossed_up & out["ma_ratio"].notna(), "ma_ratio"] <= 0).all()


def test_feature_frame_excludes_raw_prices_and_drops_the_unlabelled_tail():
    df = _prices(200)
    X, y, cols = build_feature_frame(df, (1, 5), (5,), 10, 20, horizon=5)
    assert not {"open", "high", "low", "close", "volume"} & set(cols)
    assert list(X.columns) == cols
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    # The last `horizon` bars have no forward return, so they cannot be labelled.
    assert X.index.max() <= df.index[-6]


def test_feature_label_is_the_sign_of_the_forward_return():
    df = _prices(120)
    X, y, _ = build_feature_frame(df, (1,), (5,), 5, 10, horizon=3)
    forward = df["close"].shift(-3) / df["close"] - 1
    assert (y == (forward.reindex(y.index) > 0).astype(int)).all()


def test_equity_curve_compounds_from_the_starting_capital():
    returns = pd.Series([0.10, -0.10, 0.05])
    curve = equity_curve(returns, 1_000)
    assert curve.iloc[0] == pytest.approx(1_100)
    assert curve.iloc[-1] == pytest.approx(1_000 * 1.10 * 0.90 * 1.05)


def test_cagr_annualizes_a_known_doubling():
    # One year of trading days that doubles capital is a 100% CAGR.
    daily = pd.Series([2 ** (1 / 252) - 1] * 252)
    assert cagr(daily) == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(cagr(pd.Series(dtype=float)))


def test_volatility_and_sharpe_use_the_same_annualization():
    returns = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1_000))
    assert volatility(returns) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
    assert sharpe(returns) == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(252))


def test_sortino_only_penalizes_downside_and_exceeds_sharpe_when_skewed():
    # Upside is far more volatile than downside, so ignoring it must help.
    returns = pd.Series([0.20, 0.01, 0.15, -0.01, 0.02, -0.02] * 20)
    assert sortino(returns) > sharpe(returns)


def test_sortino_is_undefined_without_dispersed_downside():
    assert np.isnan(sortino(pd.Series([0.01, 0.02, 0.03])))  # no losses at all
    assert np.isnan(sortino(pd.Series([0.01, 0.02, -0.03])))  # one loss, no spread
    assert np.isnan(sortino(pd.Series([0.05, -0.01, 0.05, -0.01])))  # identical losses


def test_a_flat_return_series_has_no_sharpe_rather_than_an_enormous_one():
    # The standard deviation of ten identical values is 1.8e-18, not zero, so an
    # equality guard leaves the mean divided by float residue.
    flat = pd.Series([0.01] * 10)
    assert flat.std(ddof=1) != 0.0
    assert np.isnan(sharpe(flat))
    assert np.isnan(sharpe(pd.Series([0.0] * 10)))
    assert np.isnan(sortino(pd.Series([-0.01] * 10)))


def test_max_drawdown_is_the_worst_peak_to_trough():
    returns = pd.Series([0.5, -0.5, 0.2])
    # 1.0 -> 1.5 -> 0.75 is a 50% fall from the peak.
    assert max_drawdown(returns) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.01, 0.01])) == pytest.approx(0.0)


def test_summary_reports_turnover_only_when_positions_are_supplied():
    returns = pd.Series([0.01, -0.02, 0.03])
    assert "turnover" not in summary(returns)
    positions = pd.Series([0.0, 1.0, 0.0])
    assert summary(returns, positions)["turnover"] == pytest.approx(2.0)


def test_chronological_split_never_puts_the_future_in_training():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))
    X_train, X_test, y_train, y_test = chronological_split(X, y, 0.25)
    assert len(X_train) == 75 and len(X_test) == 25
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()


def test_logistic_training_scores_only_the_held_out_tail():
    X, y, _ = build_feature_frame(_prices(400), (1, 5), (5,), 10, 20, horizon=5)
    _, probability, prediction, metrics = train_logistic(X, y, test_size=0.2)
    assert len(probability) == len(X) - int(len(X) * 0.8)
    assert probability.between(0, 1).all()
    assert prediction.isin({0, 1}).all()
    assert 0.0 <= metrics["accuracy"] <= 1.0 and 0.0 <= metrics["roc_auc"] <= 1.0


def test_probability_threshold_controls_how_often_the_signal_fires():
    probability = pd.Series([0.1, 0.45, 0.55, 0.9])
    assert probability_to_signal(probability, 0.5).sum() == 2
    assert probability_to_signal(probability, 0.8).sum() == 1
    assert probability_to_signal(probability, 0.5).name == "signal"


def test_baselines_are_binary_and_aligned_to_the_price_index():
    df = _prices(120)
    for signal in (buy_and_hold(df), momentum(df, 20), moving_average(df, 10, 30)):
        assert signal.index.equals(df.index)
        assert signal.dropna().isin({0.0, 1.0}).all()
        assert signal.name == "signal"


def test_moving_average_rejects_an_inverted_window_pair():
    with pytest.raises(ValueError):
        moving_average(_prices(50), fast=50, slow=20)
