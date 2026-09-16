from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.Series, pd.Series, dict]:
    """Chronologically split data and train a leakage-safe XGBoost classifier."""
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probability = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="probability")
    prediction = (probability >= 0.5).astype(int)

    metrics = {"accuracy": accuracy_score(y_test, prediction)}
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, probability)
    else:
        metrics["roc_auc"] = float("nan")
    return model, probability, prediction, metrics
