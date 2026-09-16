"""Calibrated probability estimation for binary prediction markets.

Trading a prediction market is a calibration problem, not a classification
problem. A signal that ranks outcomes well but reports 0.8 where the truth is
0.6 will size every position wrong and lose money at a perfectly respectable
AUC, because the stake is a function of the *level* of the probability and not
of its rank.

:class:`ChronologicalCalibratedClassifier` therefore fits the base estimator on
the earlier part of the training window and fits the calibration map on the
later part, using predictions the base model has never seen. Scikit-learn's own
``CalibratedClassifierCV`` shuffles its folds, which would let a market's later
observations calibrate its own earlier ones; the chronological split here does
not. The estimator implements the standard fit/predict_proba interface, so
:func:`quant_system.evaluation.walk_forward.walk_forward_predict` drives it
without modification and the out-of-sample discipline is shared with the
equities stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_EPSILON = 1e-6


class ChronologicalCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Fit a classifier and calibrate it on a strictly later held-out slice.

    Parameters
    ----------
    base_estimator:
        Any classifier exposing ``predict_proba``. Defaults to a scaled
        logistic regression.
    method:
        ``"isotonic"`` for a monotone non-parametric map, or ``"sigmoid"`` for
        a Platt scaling that is steadier on small calibration samples.
    calibration_fraction:
        Share of the training window reserved, chronologically last, for
        fitting the calibration map.
    min_calibration_size:
        Below this many calibration rows the map is skipped and the base
        estimator is refit on the full window, because a map estimated from a
        handful of points is noisier than no map at all.
    """

    def __init__(
        self,
        base_estimator=None,
        method: str = "isotonic",
        calibration_fraction: float = 0.3,
        min_calibration_size: int = 60,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.calibration_fraction = calibration_fraction
        self.min_calibration_size = min_calibration_size

    def fit(self, X, y) -> ChronologicalCalibratedClassifier:
        if self.method not in {"isotonic", "sigmoid"}:
            raise ValueError("method must be 'isotonic' or 'sigmoid'")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie in (0, 1)")

        y_array = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_array)
        n = len(y_array)
        split = int(n * (1.0 - self.calibration_fraction))
        holdout = n - split

        base = clone(self.base_estimator) if self.base_estimator is not None else _default_base()
        usable = (
            holdout >= self.min_calibration_size
            and split > 0
            and len(np.unique(y_array[:split])) > 1
            and len(np.unique(y_array[split:])) > 1
        )
        if not usable:
            # Not enough later history to estimate a map; use the raw model and
            # record that fact rather than silently reporting it as calibrated.
            base.fit(X, y_array)
            self.base_ = base
            self.calibrator_ = None
            self.calibrated_ = False
            return self

        X_fit, X_cal = _split_rows(X, split)
        base.fit(X_fit, y_array[:split])
        raw = base.predict_proba(X_cal)[:, 1]
        self.calibrator_ = _fit_calibrator(self.method, raw, y_array[split:])
        # Refit on the full window so the deployed model uses all information;
        # the map stays as estimated, which is the usual prefit-calibration
        # trade-off and keeps the map itself free of in-sample predictions.
        refit = clone(self.base_estimator) if self.base_estimator is not None else _default_base()
        refit.fit(X, y_array)
        self.base_ = refit
        self.calibrated_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            raw = _apply_calibrator(self.method, self.calibrator_, raw)
        positive = np.clip(np.asarray(raw, dtype=float), _EPSILON, 1.0 - _EPSILON)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def default_model(random_state: int = 42, method: str = "isotonic") -> ChronologicalCalibratedClassifier:
    """Scaled logistic regression with chronological calibration."""
    return ChronologicalCalibratedClassifier(
        base_estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
            ]
        ),
        method=method,
    )


def gradient_boosted_model(
    random_state: int = 42,
    method: str = "isotonic",
) -> ChronologicalCalibratedClassifier:
    """Calibrated gradient-boosted trees, for comparison against the linear model."""
    from xgboost import XGBClassifier

    return ChronologicalCalibratedClassifier(
        base_estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        method=method,
    )


def blend_with_market(
    probability: pd.Series,
    market_price: pd.Series,
    shrinkage: float,
) -> pd.Series:
    """Vectorized shrinkage of a forecast toward the market's implied price.

    ``shrinkage`` of 1 reproduces the market exactly and therefore guarantees
    zero measured edge, which is the correct behaviour for a model that has not
    demonstrated skill.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    p, price = pd.Series(probability).align(pd.Series(market_price), join="inner")
    blended = (1.0 - shrinkage) * p.astype(float) + shrinkage * price.astype(float)
    return blended.clip(_EPSILON, 1.0 - _EPSILON).rename("probability")


def _default_base() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )


def _split_rows(X, split: int):
    if hasattr(X, "iloc"):
        return X.iloc[:split], X.iloc[split:]
    array = np.asarray(X)
    return array[:split], array[split:]


def _fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray):
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
    logit = _logit(raw).reshape(-1, 1)
    return LogisticRegression(max_iter=1000).fit(logit, y)


def _apply_calibrator(method: str, calibrator, raw: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return calibrator.predict(raw)
    return calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


class MarketAnchoredClassifier(BaseEstimator, ClassifierMixin):
    """Forecast the market price plus a learned correction, in logit space.

    Fitting a classifier on the market price as an ordinary feature asks it to
    re-estimate the coefficient on the single strongest predictor available.
    With the few hundred *independent* settled markets a real panel supplies,
    that coefficient is estimated badly, and a coefficient of 0.8 where the
    truth is 1.0 shrinks every forecast toward a coin flip and scores worse
    than simply quoting the price. Measured on the synthetic panel, that is
    exactly what happens: an unanchored model loses to the price it was given.

    This estimator removes the problem by construction::

        logit(q) = logit(market price) + g(other features)

    The price enters as a fixed offset with coefficient one, so ``g = 0``
    reproduces the market exactly. Regularization pulls ``g`` toward zero,
    which means the default behaviour is to defer to the price and any
    departure from it has to be paid for out of the likelihood. The worst case
    degrades to the market's own forecast instead of to something worse.

    ``backend`` selects a regularized linear correction (``"linear"``) or a
    gradient-boosted one (``"boosted"``), the latter using XGBoost's
    ``base_margin`` for the same offset.

    ``alpha="auto"`` selects the penalty inside the training window by
    chronological hold-out, so the strength of the correction is not a knob the
    researcher turns after seeing out-of-sample results. That matters more here
    than usual: the achievable skill over an efficient price is a few
    thousandths of a Brier point, which is smaller than the swing an alpha
    chosen in hindsight can manufacture.
    """

    def __init__(
        self,
        anchor_column: str = "feat_logit_price",
        backend: str = "linear",
        alpha: float | str = "auto",
        alpha_grid: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        validation_fraction: float = 0.3,
        n_estimators: int = 200,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.anchor_column = anchor_column
        self.backend = backend
        self.alpha = alpha
        self.alpha_grid = alpha_grid
        self.validation_fraction = validation_fraction
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None) -> MarketAnchoredClassifier:
        if self.backend not in {"linear", "boosted"}:
            raise ValueError("backend must be 'linear' or 'boosted'")
        if not hasattr(X, "columns"):
            raise TypeError("MarketAnchoredClassifier requires a DataFrame with named columns")
        if self.anchor_column not in X.columns:
            raise ValueError(f"anchor column {self.anchor_column!r} is not in X")

        y_array = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_array)
        self.alpha_ = self._resolve_alpha(X, y_array, sample_weight)
        offset = np.asarray(X[self.anchor_column], dtype=float)
        features = X.drop(columns=[self.anchor_column])
        self.feature_names_ = list(features.columns)

        if self.backend == "linear":
            self.scaler_ = StandardScaler().fit(features)
            self.estimator_ = _OffsetLogistic(alpha=float(self.alpha_)).fit(
                self.scaler_.transform(features), y_array, offset, sample_weight
            )
        else:
            from xgboost import XGBClassifier

            self.scaler_ = None
            self.estimator_ = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_lambda=float(self.alpha_),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=1,
            )
            self.estimator_.fit(
                features, y_array, base_margin=offset, sample_weight=sample_weight
            )
        return self

    def predict_proba(self, X) -> np.ndarray:
        offset = np.asarray(X[self.anchor_column], dtype=float)
        features = X.drop(columns=[self.anchor_column])[self.feature_names_]
        if self.backend == "linear":
            positive = _sigmoid(
                self.estimator_.decision_function(self.scaler_.transform(features), offset)
            )
        else:
            positive = self.estimator_.predict_proba(features, base_margin=offset)[:, 1]
        positive = np.clip(np.asarray(positive, dtype=float), _EPSILON, 1.0 - _EPSILON)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _resolve_alpha(self, X, y: np.ndarray, sample_weight) -> float:
        """Pick the penalty by chronological hold-out inside the training window.

        Candidates are scored by log loss on the later slice, which rewards
        calibrated probabilities rather than rank ordering. When the hold-out is
        too small or single-class, the most conservative candidate wins: the
        strongest penalty keeps the forecast closest to the market price, which
        is the right default for a model that has not yet shown it can improve
        on it.
        """
        if self.alpha != "auto":
            return float(self.alpha)
        grid = tuple(sorted(self.alpha_grid))
        if not grid:
            raise ValueError("alpha_grid must not be empty")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")

        split = int(len(y) * (1.0 - self.validation_fraction))
        inner_y, holdout_y = y[:split], y[split:]
        if split < 2 or len(holdout_y) < 20 or len(np.unique(inner_y)) < 2 or len(
            np.unique(holdout_y)
        ) < 2:
            return float(grid[-1])

        inner_X, holdout_X = X.iloc[:split], X.iloc[split:]
        inner_weight = None if sample_weight is None else np.asarray(sample_weight)[:split]
        best_alpha, best_loss = float(grid[-1]), np.inf
        for candidate in grid:
            trial = MarketAnchoredClassifier(
                anchor_column=self.anchor_column,
                backend=self.backend,
                alpha=float(candidate),
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            ).fit(inner_X, inner_y, sample_weight=inner_weight)
            probability = np.clip(trial.predict_proba(holdout_X)[:, 1], _EPSILON, 1 - _EPSILON)
            loss = float(
                -np.mean(
                    holdout_y * np.log(probability) + (1 - holdout_y) * np.log(1 - probability)
                )
            )
            if loss < best_loss:
                best_alpha, best_loss = float(candidate), loss
        return best_alpha


def market_anchored_model(
    alpha: float | str = "auto",
    anchor_column: str = "feat_logit_price",
) -> MarketAnchoredClassifier:
    """Linear correction to the market price, with the penalty chosen nested."""
    return MarketAnchoredClassifier(
        anchor_column=anchor_column, backend="linear", alpha=alpha
    )


def market_anchored_boosted_model(
    alpha: float | str = "auto",
    random_state: int = 42,
    anchor_column: str = "feat_logit_price",
) -> MarketAnchoredClassifier:
    """Gradient-boosted correction to the market price."""
    return MarketAnchoredClassifier(
        anchor_column=anchor_column,
        backend="boosted",
        alpha=alpha,
        random_state=random_state,
    )


class _OffsetLogistic:
    """L2-penalized logistic regression with a fixed per-observation offset.

    Scikit-learn's ``LogisticRegression`` has no offset term, so the likelihood
    is optimized directly. The intercept is penalized alongside the
    coefficients on purpose: a non-zero intercept asserts the market is
    systematically biased, which is a claim that should have to be earned from
    the data rather than fitted for free.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        offset: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> _OffsetLogistic:
        from scipy.optimize import minimize

        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        base = np.asarray(offset, dtype=float)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("sample_weight must not sum to zero")

        n_features = features.shape[1]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients, intercept = theta[:n_features], theta[n_features]
            z = base + features @ coefficients + intercept
            # logaddexp keeps the log-likelihood stable for large |z|.
            loss = float(
                (weights * (np.logaddexp(0.0, z) - target * z)).sum() / total
            ) + 0.5 * self.alpha * float(coefficients @ coefficients + intercept**2)
            residual = weights * (_sigmoid(z) - target) / total
            gradient = np.empty_like(theta)
            gradient[:n_features] = features.T @ residual + self.alpha * coefficients
            gradient[n_features] = residual.sum() + self.alpha * intercept
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(n_features + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.coef_ = result.x[:n_features]
        self.intercept_ = float(result.x[n_features])
        self.converged_ = bool(result.success)
        return self

    def decision_function(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return np.asarray(offset, dtype=float) + np.asarray(X, dtype=float) @ self.coef_ + (
            self.intercept_
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -50.0, 50.0)))
