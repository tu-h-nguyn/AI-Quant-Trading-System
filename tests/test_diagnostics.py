import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_system.evaluation.diagnostics import (
    calibration_metrics,
    calibration_table,
    model_feature_importance,
    threshold_analysis,
)


def test_calibration_metrics_perfect_probabilities():
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0.0, 0.0, 1.0, 1.0])
    metrics = calibration_metrics(y, p)
    assert metrics["brier_score"] == 0.0
    assert metrics["ece"] == 0.0


def test_threshold_analysis_changes_signal_rate():
    y = pd.Series([0, 1, 1, 0])
    p = pd.Series([0.2, 0.6, 0.9, 0.4])
    out = threshold_analysis(y, p, thresholds=np.array([0.5, 0.8]))
    assert out.loc[0, "signal_rate"] > out.loc[1, "signal_rate"]


def test_logistic_feature_importance_is_normalized():
    X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    model.fit(X, y)
    out = model_feature_importance(model, ["a", "b"])
    assert len(out) == 2
    assert out["normalized_importance"].sum() == 1.0


def test_calibration_table_has_observation_counts():
    y = pd.Series([0, 1, 1, 0])
    p = pd.Series([0.1, 0.2, 0.8, 0.9])
    out = calibration_table(y, p, bins=4)
    assert int(out["observations"].sum()) == 4
