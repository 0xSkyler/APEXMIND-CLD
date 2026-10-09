# Results

## 1. Real-market evidence: none yet

* **No market data has been collected.** The development environment's
  network policy blocked every exchange endpoint (Lighter, Binance, Bybit,
  OKX and Hyperliquid all returned HTTP 403 at the egress proxy), so the
  collector could not run.
* **No trades** (live or paper on real data) have been made, and none are
  reported here.
* **No claim** is made that CVIFA, or any model in this repository, is
  profitable on Lighter or outperforms any market participant. Whether a
  cross-venue informed-flow edge survives Lighter's real costs and latency is
  an open empirical question. It can only be answered by running the
  collector, the integration test (with latency calibration) and the research
  protocol on a host with exchange access.

## 2. Methodology validation (synthetic, known ground truth)

These results test the **research procedure**, not any market. The data comes
from `apexmind.data.synthetic` (two symbols, 6 hours, 100 ms steps, realistic
message formats, feed delays, spreads, depth, trade flow, funding). Latency
uses the conservative **prior** (lognormal, median 400 ms), and fees are 2 bps
taker per leg. Full reports: [`validation/positive_control/report.md`](validation/positive_control/report.md),
[`validation/negative_control/report.md`](validation/negative_control/report.md).
Reproduce with `apexmind validate-methodology --out <dir> --hours 6 --lag 3 --seed 5`
(code `54b7136`).

Both controls use the same design: three walk-forward folds, each with 1 h of
out-of-sample test data; 30-minute book resyncs; 10-minute clustering blocks;
six models × two execution modes; and $1,000 primary equity plus a $10
scenario.

### Positive control: Lighter lags the reference by 3 s

| Model | OOS trades | Net bps per trade (95% CI) | Profit factor | Max DD | OOS periods positive |
|---|---|---|---|---|---|
| ridge | 1,847 | **6.09 [5.33, 6.91]** | 4.26 | 0.5% | 3/3 |
| gbm | 1,879 | 5.21 [4.57, 5.96] | 3.34 | 0.5% | 3/3 |
| lag_flow | 1,721 | 3.57 [2.99, 4.05] | 2.09 | 1.2% | 3/3 |
| lag_only | 1,684 | 3.60 [2.71, 4.48] | 2.09 | 1.6% | 3/3 |
| ref_momentum | 1,963 | 3.58 [3.14, 4.11] | 2.26 | 0.8% | 3/3 |
| flow_only | 639 | 0.76 [-0.56, 3.21] | 1.24 | 4.6% | 1/3 (no trades in one) |

* Selected: **ridge / aggressive**, the simplest model not significantly worse
  than the best. It beats every baseline in paired tests (p < 0.001). Hansen
  SPA versus not trading: p < 0.001. PBO = 0.00.
* Execution: gross 10.1 bps, minus 4.0 bps fees, gives 6.1 bps net. Mean entry
  slippage versus the decision mid was 1.5 bps, including latency drift. The
  optimizer never preferred passive entry: waiting for a fill forfeits a
  latency-sensitive edge.
* **It would still fail the live promotion gate.** Its deflated Sharpe ratio is
  0.72, below the 0.95 required with 288 counted trials and only 19
  out-of-sample periods. The data is also synthetic and latency was not
  measured. Even a strong planted edge does not earn deployment on 3 h of
  out-of-sample evidence.
* A $10 account was feasible (ridge: 1,078 trades). Many signals were
  rejected as `below_exchange_minimum_cap`. With one position already open, a
  second minimum-size ($10) order would have breached the 50%
  margin-utilization cap, so it was rejected rather than levered up.

### Negative control: no lead-lag

| Model | OOS trades | Net bps per trade (95% CI) | Profit factor |
|---|---|---|---|
| ref_momentum, lag_only, flow_only, lag_flow | 0 | no threshold had a positive LCB on calibration data | - |
| ridge | 111 | -7.75 [-15.53, -0.31] | 0.63 |
| gbm | 28 | 1.34 [-18.49, 31.38] | 1.07 |

* Verdict: **NO MODEL demonstrated a credible out-of-sample edge after costs.**
  Hansen SPA p = 0.39 and PBO = 0.93: the best in-sample configuration is
  usually below median out of sample, the signature of overfitting.
* Ridge and GBM both found apparent edges on calibration windows that failed
  out of sample. The pipeline caught this through OOS CIs, fold consistency,
  SPA and PBO, so nothing would be promoted.

### Other verified properties (test suite)

* Features at time *t* are bit-identical with or without later data
  (no look-ahead). Label arithmetic is exact on a hand-built order book.
* Research results are reproducible: an identical results hash on re-run.
* The paper trader runs end to end on replayed data, recovers after a
  simulated crash without exit-order spam, honours the kill file, and refuses
  live mode without validated evidence.
* Live transaction encoding was verified offline with the official Lighter
  signer. No code path references withdrawals or transfers.

## 3. How real evidence will be produced

1. Deploy (`deploy/install.sh`) on a host that can reach Lighter and Binance,
   and run the collector for at least several days, ideally weeks.
2. Run `apexmind integration-test --place-test-order --latency-samples 30`.
   This produces measured latency.
3. The Alpha Lab builds datasets, runs the protocol in
   [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md), writes `report.md` per
   cycle and promotes a champion only if every gate check passes.
4. If the evidence is negative, the correct outcome is **no deployment**.
