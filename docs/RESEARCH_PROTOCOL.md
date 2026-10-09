# Research protocol

This protocol decides whether any strategy has *earned* deployment. It is
applied identically to every model, including the simple baselines.

## 1. Data

* Record with `apexmind collect`. Every message carries a local arrival
  timestamp (monotonic-anchored) and the venue timestamp. Books are resynced
  every 30 minutes. Clock probes, chrony status, metadata, mappings and live
  execution records are stored as `meta` records.
* Data quality per stream (`QualityMonitor`): message counts, sequence gaps,
  disconnects, maximum inter-arrival gap, timestamp regressions and observed
  delay distribution. Decision samples are `valid` only when both books are
  usable and fresh, and invalid samples never produce decisions.
* Asynchronous events are never treated as simultaneous. The decision grid
  (default 250 ms) samples state *as of* each tick.

## 2. Targets (execution-conditioned)

For each valid decision and both sides: entry after a latency draw (measured
decision-to-fill-print distribution, or a flagged conservative prior), a book
walk for the label notional, exit after horizon *H* plus a second latency
draw, fees on both legs, and funding if a funding time is crossed. Further
targets are MAE, slippage, passive fill and its outcome, and holding time.

## 3. Evaluation design

* **Purged walk-forward.** Expanding fit window, then a gap, a calibration
  window, another gap and a test window. Gaps are at least the longest label
  span. Test windows are disjoint, chronological and identical for every model.
* **Feature selection** on fit data only (block IC stability plus redundancy).
* **Policy** on calibration data only. For each side and horizon, candidate
  thresholds at score quantiles. The selection is the (horizon, threshold)
  that maximizes capacity × LCB, where LCB = mean − z·SE with cluster-robust
  SE over time blocks. A side with no LCB > 0 does not trade.
* **Simulation** on test data. Every signal becomes a trade or a recorded
  rejection (position open, below exchange minimum, caps, execution
  unavailable, passive unfilled, ...).

## 4. Statistics

* Per-trade net return CI: bootstrap that resamples whole time blocks.
* Profit factor, expectancy, maximum drawdown (realized and MAE-based),
  Expected Shortfall (per trade and per period).
* Comparison against baselines: paired stationary-bootstrap test on
  per-period unit-notional PnL over the same periods.
* Multiple testing: Hansen's SPA (vs not trading, and vs the best baseline);
  the deflated Sharpe ratio counting every configuration evaluated; and PBO
  via CSCV across models.
* Regime breakdown (volatility, spread, trend; thresholds from fit data) and
  per-fold results.

## 5. Selection and promotion

A model/execution pair is *credible* if it has at least 30 OOS trades, a CI
lower bound > 0, a profit factor > 1, and positive results in a majority of
OOS periods. Among credible pairs, the **simplest** one not significantly worse
than the best is selected. Promotion for live use additionally requires:
recorded data, measured latency, at least 3 OOS periods, at least 200 OOS trades,
SPA p < 0.05, beating every baseline, PBO ≤ 0.3, DSR ≥ 0.95, an identical results
hash on a full re-run, and (if a champion exists) significant out-performance on
periods the champion never saw.

## 6. After deployment

The `EdgeMonitor` compares live net returns to the calibrated LCB. A shortfall
halves size, and repeated shortfall suspends the strategy. The multiplier
never exceeds 1. The lab re-evaluates the champion on unseen data each cycle
and retires it if its edge is gone. Live execution records feed back into the
measured latency model.

## 7. Validating the methodology itself

`apexmind validate-methodology` runs the complete pipeline on synthetic data
with known truth. The positive control (Lighter lags the reference) must
yield a credible model. The negative control (no lag) must not. These results
validate the *procedure*, not any market.
