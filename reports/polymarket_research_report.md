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
- Edge gate: 0.040. At the median price of 0.495 the break-even edge is 0.0150, so the gate is above the friction floor.
- Positions are held to settlement; committed capital is unavailable to later trades.
- Baseline tilt: 0.0888, the smallest fixed mispricing claim that still clears the gate after shrinkage and frictions.
- Passthrough features supplied by the caller: `signal`. These are trusted, not validated for look-ahead.

## Forecast quality

Skill is measured against the market price, not against a coin flip. A positive Brier skill score means the forecast carries information the price does not.

**The achievable skill here is +0.01107.** That is what an oracle holding the true probabilities would score against this price series. Brier score on binary outcomes is dominated by the irreducible variance `q(1 - q)`, so skill against a roughly efficient price is always a few thousandths even when the economic edge is large. Read every number below as a fraction of that ceiling, not against 1.0.

| Strategy | Observations | Brier | Market Brier | Brier skill vs market | % of ceiling | Calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Market price | 25000 | 0.20002 | 0.20002 | +0.00000 | +0.0 | 0.0064 |
| Favourite | 25000 | 0.20775 | 0.20002 | -0.03866 | -349.2 | 0.0877 |
| Longshot | 25000 | 0.20779 | 0.20002 | -0.03885 | -350.8 | 0.0428 |
| Logistic (unanchored) | 17500 | 0.20252 | 0.19888 | -0.01832 | -165.4 | 0.0175 |
| Market-anchored linear | 17500 | 0.19799 | 0.19888 | +0.00443 | +40.0 | 0.0115 |
| Market-anchored boosted | 17500 | 0.19848 | 0.19888 | +0.00198 | +17.8 | 0.0087 |

## Trading results after frictions

| Strategy | Trades | ROI on capital | Total return | Hit rate | Max drawdown | Mean trade return | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Market price | 0 | — | — | — | — | — | — |
| Favourite | 69 | -0.0417 | -0.0404 | 0.6957 | -0.1086 | -0.0006 | [-0.1358, +0.0589] |
| Longshot | 67 | -0.0481 | -0.0454 | 0.3433 | -0.1592 | +0.0831 | [-0.1813, +0.5543] |
| Logistic (unanchored) | 49 | +0.3179 | +0.2433 | 0.6122 | -0.1162 | +0.4262 | [-0.0732, +0.8117] |
| Market-anchored linear | 44 | +0.3282 | +0.2174 | 0.5909 | -0.0668 | +0.3548 | [+0.1132, +0.5521] |
| Market-anchored boosted | 55 | +0.0432 | +0.0339 | 0.5455 | -0.1188 | +0.0399 | [-0.2400, +0.3451] |

## Edge realization

Realized profit per share regressed on the edge predicted before the trade. A slope near one means predicted edge materialized; a slope near zero means the forecast carried no information about outcomes and the trading was noise.

An edge gate accepts trades only in a narrow band of predicted edge, which leaves the regressor with little spread against a payoff that swings a full dollar. Read the slope next to its standard error: a large slope with a larger standard error is noise, not evidence.

| Strategy | Slope | Std error | t | R² | Mean predicted edge | Mean realized edge |
|---|---:|---:|---:|---:|---:|---:|
| Market price | — | — | — | — | — | — |
| Favourite | -12.980 | 70.956 | -0.18 | 0.0005 | +0.0477 | -0.0048 |
| Longshot | — | — | — | — | +0.0477 | +0.0261 |
| Logistic (unanchored) | +0.535 | 2.244 | +0.24 | 0.0012 | +0.0685 | +0.1713 |
| Market-anchored linear | -12.581 | 8.841 | -1.42 | 0.0460 | +0.0482 | +0.1376 |
| Market-anchored boosted | -1.273 | 6.529 | -0.19 | 0.0007 | +0.0531 | +0.0577 |

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
- Resolution risk is not modelled: markets can settle on a technicality, be disputed, or resolve differently from the plain reading of the question.
- Capital lock-up until settlement is modelled, but the opportunity cost of that capital is not.
- Settled markets are a survivorship-inflected sample; markets that were voided or never resolved do not appear in the panel.
- Multiple models were examined, which creates researcher degrees of freedom. The regularization strength is chosen inside each training window rather than from these results, but the choice of model family was not. Treat a single positive result as a hypothesis, not a finding.

