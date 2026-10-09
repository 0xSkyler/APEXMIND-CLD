# APEX MIND v4: Cross-Venue Informed Flow Alpha (CVIFA)

An evidence-gated research and execution system for Lighter.xyz perpetual
futures. It records synchronized Lighter and reference-venue (Binance USD-M)
market data, builds point-in-time features, learns **execution-conditioned**
net-return forecasts, and only trades strategies that pass automated
out-of-sample validation against simple baselines after realistic costs.

## Current status

| | |
|---|---|
| Real market data collected | **None yet.** The development container could not reach any exchange (network policy). |
| Trading history | **None.** No live or paper trades on real data exist. |
| Evidence of a market edge | **None.** No claim of profitability or competitive advantage is made. |
| Code and methodology | Implemented and tested (61 tests, plus a slow end-to-end control experiment). |
| Methodology validation | Synthetic positive/negative controls: the pipeline finds a planted edge and rejects a world with none. See [docs/RESULTS.md](docs/RESULTS.md). |

The system is built to *earn* deployment: the trader refuses live mode until a
model has passed the promotion gate on **recorded** data with **measured**
execution latency and a recent exchange integration test has passed.

## What it does

1. **Collect**: Lighter order book, trades and market stats, plus Binance USD-M
   book ticker, diff depth, trades and mark/funding. Raw messages are stored
   with local monotonic arrival timestamps and exchange timestamps, along
   with clock probes, metadata snapshots and periodic book resyncs.
2. **Features**: a single streaming, point-in-time engine (used identically in
   research and live) computes cross-venue return differences, basis z-score,
   order-book and order-flow imbalance, aggressive volume ratio, spread,
   depth, impact, realized volatility, replenishment, cancellation intensity,
   funding, regime and freshness/latency features.
3. **Targets**: for every decision, simulated orders are filled after a
   sampled execution latency against the book that existed at that instant.
   Targets include net return per side and horizon after fees and funding,
   adverse excursion, slippage, passive fill outcomes and holding time.
4. **Models**: reference momentum, lag-only, flow-only and lag+flow baselines;
   ridge; gradient boosting; and a temporal model that is gated on evidence.
   All are compared on identical purged walk-forward windows.
5. **Policy**: entry thresholds and holding horizons are learned per side on a
   held-out calibration window, using a cluster-robust lower confidence bound
   on realized net returns.
6. **Evaluation**: an execution-aware portfolio backtest (capital, exchange
   minimums, margin, concurrency, passive versus aggressive), with bootstrap CIs,
   Hansen's SPA, the deflated Sharpe ratio, PBO, regime breakdowns and per-fold
   results. The *simplest* credible model is selected.
7. **Alpha Lab**: resource-limited continuous research with reproducibility
   re-runs, champion/challenger promotion and retirement of decayed champions.
8. **Trading**: a 10-step decision pipeline, fractional-Kelly allocation from $10
   (never raising leverage to meet minimums), an execution optimizer,
   write-ahead order state, reconciliation, crash recovery, a dead-man switch,
   a kill file and an edge monitor that reduces or suspends (and never levers up).
   There is no withdrawal or transfer surface.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # add ",live" on the trading host for lighter-sdk
pytest                              # fast suite
pytest -m slow                      # synthetic positive/negative controls (~15 min)

apexmind validate-methodology --out runs/validation   # same controls via the CLI
apexmind collect                                      # record real data (needs exchange access)
apexmind research --hours 168                         # walk-forward study + report.md
apexmind integration-test --place-test-order --latency-samples 30
apexmind lab                                          # continuous research and promotion
apexmind trade --mode paper
```

Deployment on Ubuntu: [docs/OPERATIONS.md](docs/OPERATIONS.md). Design:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Statistical protocol:
[docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md). Protocol assumptions
and known limitations: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).
