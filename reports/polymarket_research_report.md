# Polymarket Prediction-Market Research Study

> Research question: does a calibrated probability forecast beat the market's own implied probability by enough to survive spread, fees, and capital lock-up?

## Data provenance

**Synthetic data.** No cached Polymarket panel was present, so the study ran on generated markets in which an exploitable edge exists *by construction* (`signal_strength = 0.75`). These numbers validate that sizing, frictions, settlement, leakage control, and capital accounting are wired together correctly. They are **not** evidence of edge on Polymarket. Run `scripts/fetch_polymarket_data.py` to replace this panel with real settled markets.

## Experimental design

- Panel: 40000 snapshots across 2500 markets; 25000 usable observations after feature warm-up.
- Features: 17 columns, each computed from the market's own past and lagged one observation.
- Out-of-sample rule: 35 chronological blocks, each trained only on markets that had **already settled** when the block began. A market's snapshots therefore cannot appear on both sides of a split, and no label is used before it was knowable.
- Cold start: no forecast is produced before 2024-03-07, by which point 1400 observations from settled markets exist.
- Frictions: 0.020 spread + 0.005 adverse selection + 0.0 bps fee, charged on entry.
- Sizing: 0.25 x Kelly, capped at 2.0% per market and 20% aggregate, with forecasts shrunk 35% toward the market price.
- Edge gate: 0.040. At the median price of 0.494 the break-even edge is 0.0150, so the gate is above the friction floor.
- Positions are held to settlement; committed capital is unavailable to later trades.
- Baseline tilt: 0.0888, the smallest fixed mispricing claim that still clears the gate after shrinkage and frictions.
- Passthrough features supplied by the caller: `signal`. These are trusted, not validated for look-ahead.

## Forecast quality

Skill is measured against the market price, not against a coin flip. A positive Brier skill score means the forecast carries information the price does not.

**The achievable skill here is +0.02343.** That is what an oracle holding the true probabilities would score against this price series. Brier score on binary outcomes is dominated by the irreducible variance `q(1 - q)`, so skill against a roughly efficient price is always a few thousandths even when the economic edge is large. Read every number below as a fraction of that ceiling, not against 1.0.

| Strategy | Observations | Brier | Market Brier | Brier skill vs market | % of ceiling | Calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Market price | 17500 | 0.20286 | 0.20286 | +0.00000 | +0.0 | 0.0456 |
| Favourite | 17500 | 0.21224 | 0.20286 | -0.04623 | -197.3 | 0.0963 |
| Longshot | 17500 | 0.20907 | 0.20286 | -0.03060 | -130.6 | 0.0616 |
| Logistic (unanchored) | 17500 | 0.20469 | 0.20286 | -0.00901 | -38.4 | 0.0293 |
| Market-anchored linear | 17500 | 0.19938 | 0.20286 | +0.01720 | +73.4 | 0.0273 |
| Market-anchored boosted | 17500 | 0.19868 | 0.20286 | +0.02064 | +88.1 | 0.0203 |

## Trading results after frictions

| Strategy | Trades | ROI on capital | Total return | Hit rate | Max drawdown | Mean trade return | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Market price | 0 | — | — | — | — | — | — |
| Favourite | 45 | +0.0165 | +0.0105 | 0.7111 | -0.0529 | -0.0225 | [-0.1632, +0.1405] |
| Longshot | 48 | -0.0279 | -0.0191 | 0.3333 | -0.1408 | -0.0486 | [-0.3954, +0.2438] |
| Logistic (unanchored) | 47 | +0.2269 | +0.1784 | 0.5106 | -0.1136 | +0.2203 | [-0.3232, +0.7312] |
| Market-anchored linear | 49 | +0.4214 | +0.3719 | 0.6122 | -0.0752 | +0.4386 | [-0.0121, +0.9298] |
| Market-anchored boosted | 49 | +0.1543 | +0.1174 | 0.4898 | -0.0684 | +0.0994 | [-0.2838, +0.5484] |

## Edge realization

Realized profit per share regressed on the edge predicted before the trade. A slope near one means predicted edge materialized; a slope near zero means the forecast carried no information about outcomes and the trading was noise.

An edge gate accepts trades only in a narrow band of predicted edge, which leaves the regressor with little spread against a payoff that swings a full dollar. Read the slope next to its standard error: a large slope with a larger standard error is noise, not evidence.

| Strategy | Slope | Std error | t | R² | Mean predicted edge | Mean realized edge |
|---|---:|---:|---:|---:|---:|---:|
| Market price | — | — | — | — | — | — |
| Favourite | -37.819 | 177.765 | -0.21 | 0.0011 | +0.0477 | -0.0069 |
| Longshot | — | — | — | — | +0.0477 | +0.0284 |
| Logistic (unanchored) | +1.721 | 1.641 | +1.05 | 0.0239 | +0.0867 | +0.1028 |
| Market-anchored linear | +4.171 | 1.672 | +2.49 | 0.1169 | +0.0743 | +0.1827 |
| Market-anchored boosted | +2.891 | 2.644 | +1.09 | 0.0248 | +0.0675 | +0.0564 |

## Capital velocity

A position held to settlement earns nothing once its price has converged to the forecast, while its capital stays unavailable to every other opportunity. Exiting early gives up the tail of the edge and pays the spread a second time, so whether it is worth doing cannot be read off ROI per trade -- that metric scores a six-month hold and a one-week turn identically.

The column that decides it is **profit per capital-year**: dollars earned per dollar-year of capital actually committed.

| Exit rule | Trades | Mean hold (days) | ROI per trade | Capital-years | Profit per capital-year | Total profit |
|---|---:|---:|---:|---:|---:|---:|
| Hold to settlement | 49 | 29.0 | +0.4214 | 677 | +5.490 | +3,719 |
| Exit at 25% of edge remaining | 157 | 9.3 | +0.3239 | 742 | +12.420 | +9,218 |
| Exit at 50% of edge remaining | 172 | 8.3 | +0.2276 | 763 | +10.139 | +7,737 |
| Exit at 25%, 15c stop | 157 | 9.3 | +0.3239 | 742 | +12.420 | +9,218 |

Exit reasons: Hold to settlement — 49 settled, 0 converged, 0 stopped; Exit at 25% of edge remaining — 52 settled, 105 converged, 0 stopped; Exit at 50% of edge remaining — 45 settled, 127 converged, 0 stopped; Exit at 25%, 15c stop — 52 settled, 105 converged, 0 stopped.

A rule that raises profit per capital-year while lowering ROI per trade is doing exactly what it should. One that raises both is suspicious: early exit cannot manufacture edge, only recycle it.

## Concentration

Kelly is derived one wager at a time, so a book of independently sized positions under an aggregate cap is only as diversified as the positions are independent. Markets that settle together are one bet wearing several names, and the aggregate cap then describes a diversification the book does not have.

**Which correlation matters depends on how the position ends.** A book held to settlement is exposed to joint settlement; a maker marked to market is exposed to joint price paths. Clustering this panel on price co-movement put 120 of 120 *independent* markets into multi-member clusters and produced groups 38% pure against 25% for chance -- every market's quote drifts toward its own truth as it matures, so any two co-move whether or not their outcomes are related. Settlement risk is therefore measured directly, by asking whether markets inside a candidate group agree with each other more often than markets across groups.

| Grouping test | Value |
|---|---:|
| Markets tested | 2500 |
| Groups | 8 |
| Agreement within a group | 0.582 |
| Agreement across groups | 0.492 |
| Implied within-group outcome correlation | +0.181 |
| Permutation p-value | 0.000 |

The grouping **does** predict joint settlement at the configured significance, so it is sized against. Every row below is the same forecast and the same measured correlation; only the per-group exposure cap differs.

| Group cap | Positions | Effective bets | Peak exposure to one group | Profit | Profit per capital-year |
|---:|---:|---:|---:|---:|---:|
| 100% | 49 | 37.0 | 6.6% | +3,719 | +5.49 |
| 8% | 49 | 37.0 | 6.6% | +3,719 | +5.49 |
| 5% | 56 | 40.5 | 5.0% | +4,684 | +6.96 |
| 3% | 78 | 55.4 | 3.0% | +3,563 | +5.79 |
| 2% | 64 | 34.5 | 2.0% | +2,994 | +6.30 |

Read the peak-exposure column against the cap in the same row. Where the peak sits at the cap, the cap bound and shaped the book; where it sits comfortably below, the per-market limit was already holding the group under and the cap did nothing. Here it starts binding at 5%.

The profit column is not the thing to optimize against. These books hold dozens of trades whose outcomes are correlated by construction, so the differences between rows sit well inside the noise; the columns that carry information are the position and effective-bet counts, which are structural.

A position count is not a bet count, and the gap between them is what the cap exists to close. Whether it needs to bind depends on how many markets in one group clear the edge gate at once, which is a property of the universe rather than something a default can know -- hence a sweep rather than a number.

One consequence worth carrying into every other table in this report: when outcomes are correlated, the effective sample behind any performance estimate is nearer the group count than the observation count. The bootstrap intervals quoted elsewhere resample trades, not groups, so they are narrower than the truth.

## Resolution risk

A prediction market pays out on what the resolver decides, not on what happened. Questions get settled on technicalities, disputed, or voided. That risk is charged here twice: the forecast is discounted for it before sizing, and the simulation realizes it at the same rate.

### How much can an edge absorb

This table is analytic rather than simulated, so it carries no sampling noise. It reads: for a position entered at each price with the configured 4% edge, the settlement failure rate at which expected value reaches zero.

| Entry price | Forecast | Max tolerable resolution risk |
|---:|---:|---:|
| 0.10 | 0.14 | 28.6% |
| 0.25 | 0.29 | 13.8% |
| 0.50 | 0.54 | 7.4% |
| 0.75 | 0.79 | 5.1% |
| 0.90 | 0.94 | 4.3% |

Expensive contracts are the fragile ones. The same 4% edge tolerates several times more settlement failure at a ten-cent entry than at ninety, because at ninety there is almost nothing above the entry price left to win. A strategy concentrated in high-priced favourites is betting on the resolver as much as on the outcome.

### What it does to the strategy

| Assumed resolution risk | Trades | Total profit | ROI per trade |
|---:|---:|---:|---:|
| 0% | 49 | +3,719 | +0.4214 |
| 1% | 51 | +3,677 | +0.4420 |
| 2% | 51 | +4,310 | +0.5194 |
| 5% | 48 | +4,294 | +0.5408 |
| 10% | 50 | +1,802 | +0.2454 |

Read the trade count, not the P&L. The gate rejects more positions as the assumed risk rises -- from 49 trades at zero down the column -- and the profit figures at the bottom rest on a sample too small to carry a conclusion.

## Market making

A taker pays the spread; a maker is paid it. On a venue quoting a few cents against a one-dollar payoff that is frequently larger than any forecast edge available, so it is the other half of the profitability question -- and it fails for a different reason.

A resting quote is filled precisely when someone wants the other side, which is disproportionately when they know something. The simulation splits flow accordingly: a price move through a quote fills it and marks the position at the new price, and a configurable share of periods produce fills unrelated to any move. **All maker profit comes from that second group**, and its size is a property of the venue that price history cannot measure. It is therefore swept, not assumed.

### Quoting around: Market price

Break-even uninformed fill rate: **38.7%**. Below this share of benign flow the book loses money however tightly it quotes.

| Uninformed fill rate | Fills | P&L | Quoted spread | Adverse selection | Capture ratio | Peak capital |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1132 | -4,230 | 2,262 | -6,492 | -1.870 | 7,200 |
| 10% | 1307 | -3,358 | 2,611 | -5,968 | -1.286 | 7,100 |
| 20% | 1475 | -1,563 | 2,945 | -4,507 | -0.531 | 7,000 |
| 30% | 1638 | -1,289 | 3,269 | -4,557 | -0.394 | 7,800 |
| 40% | 1793 | +199 | 3,577 | -3,378 | +0.056 | 7,700 |
| 50% | 1976 | +802 | 3,942 | -3,141 | +0.203 | 7,100 |
| 60% | 2148 | +1,940 | 4,289 | -2,350 | +0.452 | 6,300 |
| 80% | 2482 | +2,916 | 4,958 | -2,041 | +0.588 | 5,400 |
| 100% | 2848 | +5,688 | 5,688 | +0 | +1.000 | 0 |

### Quoting around: Market-anchored linear

Break-even uninformed fill rate: **13.6%**. Below this share of benign flow the book loses money however tightly it quotes.

| Uninformed fill rate | Fills | P&L | Quoted spread | Adverse selection | Capture ratio | Peak capital |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1108 | -1,356 | 2,214 | -3,571 | -0.613 | 9,500 |
| 10% | 1275 | -498 | 2,548 | -3,047 | -0.196 | 9,100 |
| 20% | 1452 | +899 | 2,899 | -2,000 | +0.310 | 8,200 |
| 30% | 1623 | +859 | 3,241 | -2,382 | +0.265 | 7,800 |
| 40% | 1793 | +2,241 | 3,581 | -1,340 | +0.626 | 7,700 |
| 50% | 1978 | +2,693 | 3,950 | -1,258 | +0.682 | 7,100 |
| 60% | 2144 | +3,133 | 4,284 | -1,151 | +0.731 | 6,600 |
| 80% | 2483 | +3,507 | 4,963 | -1,456 | +0.707 | 5,400 |
| 100% | 2848 | +5,692 | 5,692 | +0 | +1.000 | 0 |

Capture ratio is realized P&L over the spread that was quoted. One means every quoted cent was kept; zero or below means the flow took back more than the spread paid. Adverse selection is roughly constant across the sweep because it depends on how often the price moves, not on how much benign flow arrives alongside it.

## Structural arbitrage self-test

The scanner was run against a synthetic snapshot containing 11 planted basket mispricings and reported 11. Because the planted set is known exactly, this checks the scanner for both false negatives and false positives; it says nothing about how many such baskets exist on the live venue. Run `scripts/scan_polymarket_arbitrage.py --source live` for that.

## How to read this

1. Start with **Brier skill vs market**, as a fraction of the achievable ceiling. If it is not positive, the forecast has no edge and the trading columns are measuring luck plus frictions.
2. Check that **Market price** trades rarely or loses. It has zero edge by definition, so profit from it would indicate a bug in the friction model.
3. Compare against **Favourite** and **Longshot**. These always clear the edge gate without any information, so they price what trading on a false signal costs.
4. Compare **Market-anchored** against **Logistic (unanchored)**. The unanchored entrant must re-estimate the coefficient on the market price from a few hundred independent settled markets; the anchored one takes that coefficient as one and learns only a correction. The gap between them is the cost of that free parameter.
5. Only then read ROI, and read it next to the **edge realization** slope and the trade count. A high ROI on a few dozen binary settlements is not a result.

## Limitations

- Entries assume the quoted size is available; prediction-market books are thin and a real order moves them.
- Resolution risk is charged at an assumed rate, not a measured one; the rate itself is the assumption, and the sweep above is the honest form of it.
- Capital lock-up until settlement is modelled, but the opportunity cost of that capital is not.
- Settled markets are a survivorship-inflected sample; markets that were voided or never resolved do not appear in the panel.
- Multiple models were examined, which creates researcher degrees of freedom. The regularization strength is chosen inside each training window rather than from these results, but the choice of model family was not. Treat a single positive result as a hypothesis, not a finding.

