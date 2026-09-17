"""Measuring how much of a book is really one bet.

Kelly sizing is derived one wager at a time. Applying it independently across a
list of markets and then capping the total is only safe when the markets are
independent, and prediction markets rarely are: "Fed cuts in March" and "Fed
cuts in June" sit under different events, pass every per-market gate, and settle
together. Twenty-five positions at two percent each, capped at twenty percent in
aggregate, can be a single twenty-percent wager wearing twenty-five names -- an
order of magnitude above Kelly for the one bet it actually is.

**Which correlation matters depends on how the position ends.** A book held to
settlement is exposed to joint *settlement*; a maker marked to market daily is
exposed to joint *price paths*. They are not the same quantity, and this module
keeps them apart because conflating them was the first thing tried here and it
did not survive measurement:

* clustering the synthetic panel on price co-movement put 120 of 120
  *independent* markets into multi-member clusters, largest 24;
* on a panel with four planted themes, those clusters were 38% pure against 25%
  for chance.

The cause is not a bad linkage rule -- single linkage, a Bonferroni-corrected
significance bar, and average linkage all failed the same test. It is that
prediction-market prices share a mechanical dynamic: every market's quote drifts
toward its own truth as it matures, so any two markets co-move whether or not
their outcomes have anything to do with each other. Price correlation is real
and is the wrong quantity for a settlement book.

So settlement risk is measured where it is estimable. A single market settles
once, so no pairwise outcome correlation can be estimated from it; but a
*candidate grouping* -- the venue's events, a category, a hand-drawn theme --
can be tested across many groups at once, by asking whether markets inside a
group agree with each other more often than markets across groups.
:func:`outcome_concordance` does that and converts the answer into the
within-group correlation it implies, which :func:`block_correlation` turns into
the matrix the exposure maths needs.

``logit_price_changes``/``correlation_matrix``/``cluster_markets`` remain for the
mark-to-market case, where co-movement is exactly the exposure. Do not wire them
into sizing for a book held to settlement without running
:func:`outcome_concordance` on the grouping they produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

_EPSILON = 1e-12


def logit_price_changes(
    panel: pd.DataFrame,
    market_column: str = "market_id",
    time_column: str = "timestamp",
    price_column: str = "price",
    freq: str = "1D",
) -> pd.DataFrame:
    """Reshape a long snapshot panel into per-market changes on a common grid.

    Differences are taken in logit space: a move from 0.02 to 0.04 and one from
    0.50 to 0.52 are both two cents but are not remotely the same event, and
    correlating raw price changes would treat them as comparable.

    The changes are computed on a regular calendar grid rather than on the raw
    observations, because two markets almost never carry a quote at the same
    instant. Differencing the pivoted panel directly yields a difference only
    where consecutive rows both happen to be populated, which for a panel of
    independently sampled markets is almost nowhere -- every pairwise
    correlation then comes back undefined, and a caller reading that as
    "uncorrelated" would size the book as if it were diversified.

    Each market's last quote is carried forward only to the end of its own
    history, never past it, so a settled market does not contribute a flat tail
    that would drag every correlation toward zero.
    """
    required = {market_column, time_column, price_column}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {sorted(missing)}")
    if panel.empty:
        return pd.DataFrame()

    frame = panel[[market_column, time_column, price_column]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    price = frame[price_column].astype(float).clip(1e-4, 1 - 1e-4)
    frame["logit"] = np.log(price / (1.0 - price))

    wide = frame.pivot_table(
        index=time_column, columns=market_column, values="logit", aggfunc="last"
    ).sort_index()
    if wide.empty:
        return pd.DataFrame()

    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz="UTC")
    if len(grid) < 2:
        return pd.DataFrame()
    aligned = wide.reindex(wide.index.union(grid)).ffill().reindex(grid)

    # Blank out everything after each market stopped quoting, which ffill would
    # otherwise extend as a flat line to the end of the panel.
    for market in aligned.columns:
        last = wide[market].last_valid_index()
        if last is not None:
            aligned.loc[aligned.index > last, market] = np.nan
    return aligned.diff()


def correlation_matrix(
    changes: pd.DataFrame,
    min_overlap: int = 5,
) -> pd.DataFrame:
    """Pairwise correlation of price changes, blank where the overlap is thin.

    Markets run over different windows, so most pairs share only part of their
    history. Pairs with fewer than ``min_overlap`` common observations are left
    as ``NaN`` rather than given a correlation estimated from three points,
    which would be noise dressed as structure.
    """
    if changes.empty or changes.shape[1] < 2:
        return pd.DataFrame()
    corr = changes.corr(min_periods=max(int(min_overlap), 2))
    np.fill_diagonal(corr.to_numpy(), 1.0)
    return corr


def cluster_markets(
    corr: pd.DataFrame,
    threshold: float = 0.6,
) -> dict[str, str]:
    """Group markets that move together into shared-exposure clusters.

    Single-linkage on absolute correlation: a market joins a cluster if it is
    correlated with *any* member. That deliberately over-groups rather than
    under-groups, because the cost of treating two independent bets as one is a
    smaller position, while the cost of treating one bet as two is leverage the
    sizing model never agreed to.

    Absolute value, because a strong negative correlation is the same market
    seen from the other side; whether the pair hedges or concentrates depends on
    which sides are held, which is a question for the exposure calculation and
    not for deciding what shares a budget.
    """
    if corr.empty:
        return {}
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")

    markets = list(corr.columns)
    parent = {market: market for market in markets}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    values = corr.to_numpy()
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            value = values[i, j]
            if np.isfinite(value) and abs(value) >= threshold:
                union(markets[i], markets[j])
    return {market: find(market) for market in markets}


def effective_independent_bets(
    exposures: Mapping[str, float] | pd.Series,
    corr: pd.DataFrame | None = None,
    sides: Mapping[str, str] | None = None,
) -> float:
    """How many independent wagers a book of positions amounts to.

    Computed as ``(sum |w|)^2 / (v' C v)`` where ``v`` is each exposure signed by
    the side held. Equal independent positions give back their count; perfectly
    correlated ones give one, whatever their number. Reading it next to the
    position count is the fastest way to see whether a book is diversified or
    only looks it.

    Returns ``inf`` when the signed exposures cancel, which is a fully hedged
    book carrying no directional risk rather than an infinitely diversified one.
    """
    weights = pd.Series(exposures, dtype=float).dropna()
    if weights.empty:
        return 0.0
    gross = float(weights.abs().sum())
    if gross <= _EPSILON:
        return 0.0

    signs = pd.Series(1.0, index=weights.index)
    if sides is not None:
        signs = pd.Series(
            {key: (-1.0 if str(sides.get(key, "yes")).lower() == "no" else 1.0)
             for key in weights.index}
        )
    signed = (weights.abs() * signs).to_numpy()

    if corr is None or corr.empty:
        matrix = np.eye(len(signed))
    else:
        aligned = corr.reindex(index=weights.index, columns=weights.index)
        # An unmeasured pair is assumed independent; this is the optimistic
        # reading, so a book whose correlations could not be measured should be
        # judged on the cluster caps rather than on this number.
        matrix = np.nan_to_num(aligned.to_numpy(), nan=0.0)
        np.fill_diagonal(matrix, 1.0)

    variance = float(signed @ matrix @ signed)
    if variance <= _EPSILON:
        return float("inf")
    return gross**2 / variance


def concentration_report(
    exposures: Mapping[str, float] | pd.Series,
    corr: pd.DataFrame | None = None,
    clusters: Mapping[str, str] | None = None,
    sides: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Position count against what that count is actually worth."""
    weights = pd.Series(exposures, dtype=float).dropna()
    effective = effective_independent_bets(weights, corr, sides)
    cluster_count = (
        len({clusters.get(key, key) for key in weights.index}) if clusters else len(weights)
    )
    largest_cluster = 0.0
    if clusters and not weights.empty:
        grouped = weights.abs().groupby(
            pd.Series({key: clusters.get(key, key) for key in weights.index})
        )
        largest_cluster = float(grouped.sum().max())
    return {
        "n_positions": float(len(weights)),
        "n_clusters": float(cluster_count),
        "effective_independent_bets": effective,
        "gross_exposure": float(weights.abs().sum()),
        "largest_cluster_exposure": largest_cluster,
        "diversification_ratio": effective / len(weights) if len(weights) else float("nan"),
    }


def clusters_from_events(event_ids: Mapping[str, str] | Sequence[tuple[str, str]]) -> dict[str, str]:
    """Fall back to the venue's own grouping when no price history exists.

    Weaker than measured co-movement -- it catches only markets the venue
    already files together -- but it is never worse than treating every market
    as independent.
    """
    mapping = dict(event_ids)
    return {market: (event or market) for market, event in mapping.items()}


def outcome_concordance(
    labels: Mapping[str, int] | pd.Series,
    groups: Mapping[str, str] | pd.Series,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Test whether a grouping predicts joint settlement, and by how much.

    Markets inside a group are compared with markets across groups: if the
    grouping carries real settlement risk, pairs inside it agree more often. The
    gap is converted to the within-group outcome correlation it implies, using
    the identity for two binary variables with a shared marginal ``p``::

        P(X = Y) = p^2 + (1 - p)^2 + 2 * rho * p * (1 - p)

    ``p_value`` comes from reshuffling the group labels, which preserves every
    marginal and destroys only the association being tested. A grouping that
    fails this is not one to size against -- it would cut positions to control a
    concentration that is not there.

    Needs settled markets, so it is a check to run on history before trusting a
    grouping live, not something that can be computed on open markets.
    """
    y = pd.Series(labels, dtype=float).dropna()
    g = pd.Series(groups).reindex(y.index)
    valid = g.notna()
    y, g = y[valid], g[valid].astype(str)
    if len(y) < 4 or g.nunique() < 2:
        return {
            "n_markets": float(len(y)),
            "n_groups": float(g.nunique()) if len(g) else 0.0,
            "within_concordance": float("nan"),
            "cross_concordance": float("nan"),
            "implied_within_correlation": float("nan"),
            "p_value": float("nan"),
        }

    values = y.to_numpy()
    codes = pd.factorize(g)[0]
    observed = _concordance_gap(values, codes)

    rng = np.random.default_rng(seed)
    permuted = np.empty(max(int(n_permutations), 1))
    for i in range(len(permuted)):
        permuted[i] = _concordance_gap(values, rng.permutation(codes))[0]
    gap, within, cross = observed

    base = float(values.mean())
    spread = 2.0 * base * (1.0 - base)
    implied = gap / spread if spread > _EPSILON else float("nan")
    return {
        "n_markets": float(len(y)),
        "n_groups": float(g.nunique()),
        "within_concordance": within,
        "cross_concordance": cross,
        "concordance_gap": gap,
        "implied_within_correlation": float(np.clip(implied, -1.0, 1.0)),
        "p_value": float((np.abs(permuted) >= abs(gap)).mean()),
    }


def _concordance_gap(values: np.ndarray, codes: np.ndarray) -> tuple[float, float, float]:
    """Agreement rate inside groups minus agreement rate across them.

    Counted in closed form rather than over pairs: for binary outcomes the
    number of agreeing pairs in a set follows from its size and its number of
    ones, so the whole statistic is two group-by sums regardless of how many
    markets there are.
    """
    total = len(values)
    ones = float(values.sum())
    all_pairs = total * (total - 1) / 2.0
    agree_all = ones * (ones - 1) / 2.0 + (total - ones) * (total - ones - 1) / 2.0

    within_pairs = 0.0
    agree_within = 0.0
    for code in np.unique(codes):
        member = values[codes == code]
        size = len(member)
        positives = float(member.sum())
        within_pairs += size * (size - 1) / 2.0
        agree_within += (
            positives * (positives - 1) / 2.0
            + (size - positives) * (size - positives - 1) / 2.0
        )

    cross_pairs = all_pairs - within_pairs
    if within_pairs <= 0 or cross_pairs <= 0:
        return 0.0, float("nan"), float("nan")
    within = agree_within / within_pairs
    cross = (agree_all - agree_within) / cross_pairs
    return within - cross, within, cross


def block_correlation(
    groups: Mapping[str, str] | pd.Series,
    within_correlation: float,
) -> pd.DataFrame:
    """Build the correlation matrix a validated grouping implies.

    Markets in one group carry ``within_correlation``; markets in different
    groups are taken as independent. Deliberately coarse -- it asserts only what
    the concordance test can support, which is one number per grouping rather
    than a full pairwise matrix no amount of settled history could estimate.
    """
    mapping = pd.Series(groups).astype(str)
    if mapping.empty:
        return pd.DataFrame()
    rho = float(np.clip(within_correlation, -1.0, 1.0))
    codes = mapping.to_numpy()
    same = codes[:, None] == codes[None, :]
    matrix = np.where(same, rho, 0.0)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=mapping.index, columns=mapping.index)
