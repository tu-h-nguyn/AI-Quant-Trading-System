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

## Market making

A taker pays the spread; a maker is paid it. On a venue quoting a few cents against a one-dollar payoff that is frequently larger than any forecast edge available, so it is the other half of the profitability question -- and it fails for a different reason.

A resting quote is filled precisely when someone wants the other side, which is disproportionately when they know something. The simulation splits flow accordingly: a price move through a quote fills it and marks the position at the new price, and a configurable share of periods produce fills unrelated to any move. **All maker profit comes from that second group**, and its size is a property of the venue that price history cannot measure. It is therefore swept, not assumed.

### Quoting around: Market price

Break-even uninformed fill rate: **28.7%**. Below this share of benign flow the book loses money however tightly it quotes.

| Uninformed fill rate | Fills | P&L | Quoted spread | Adverse selection | Capture ratio | Peak capital |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1013 | -1,845 | 2,026 | -3,871 | -0.911 | 4,900 |
| 10% | 1190 | -1,189 | 2,380 | -3,569 | -0.499 | 5,100 |
| 20% | 1367 | -231 | 2,734 | -2,965 | -0.085 | 5,600 |
| 30% | 1551 | +34 | 3,102 | -3,068 | +0.011 | 6,000 |
| 40% | 1722 | +451 | 3,444 | -2,993 | +0.131 | 7,000 |
| 50% | 1912 | +1,911 | 3,824 | -1,913 | +0.500 | 6,800 |
| 60% | 2093 | +3,482 | 4,186 | -704 | +0.832 | 5,600 |
| 80% | 2464 | +4,564 | 4,928 | -364 | +0.926 | 5,700 |
| 100% | 2848 | +5,690 | 5,696 | -6 | +0.999 | 0 |

### Quoting around: Market-anchored linear

**Profitable across the entire swept range, including with no benign flow at all.** A maker that makes money on purely informed flow is being paid for its forecast, not for its spread.

| Uninformed fill rate | Fills | P&L | Quoted spread | Adverse selection | Capture ratio | Peak capital |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 958 | +72 | 1,916 | -1,844 | +0.038 | 7,400 |
| 10% | 1136 | +386 | 2,272 | -1,886 | +0.170 | 7,700 |
| 20% | 1323 | +781 | 2,646 | -1,865 | +0.295 | 6,800 |
| 30% | 1508 | +1,128 | 3,016 | -1,888 | +0.374 | 6,400 |
| 40% | 1691 | +1,180 | 3,382 | -2,202 | +0.349 | 7,000 |
| 50% | 1896 | +2,396 | 3,792 | -1,396 | +0.632 | 7,400 |
| 60% | 2080 | +3,564 | 4,160 | -596 | +0.857 | 6,000 |
| 80% | 2451 | +4,409 | 4,902 | -493 | +0.899 | 5,700 |
| 100% | 2848 | +5,692 | 5,696 | -4 | +0.999 | 0 |

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
- Resolution risk is not modelled: markets can settle on a technicality, be disputed, or resolve differently from the plain reading of the question.
- Capital lock-up until settlement is modelled, but the opportunity cost of that capital is not.
- Settled markets are a survivorship-inflected sample; markets that were voided or never resolved do not appear in the panel.
- Multiple models were examined, which creates researcher degrees of freedom. The regularization strength is chosen inside each training window rather than from these results, but the choice of model family was not. Treat a single positive result as a hypothesis, not a finding.

