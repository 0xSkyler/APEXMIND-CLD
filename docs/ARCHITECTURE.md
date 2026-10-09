# Architecture

```
            +-------------------- collector (systemd) ---------------------+
 Lighter WS -> LighterFeed --+--> RawRecorder (raw msgs, local+exch ts) ----> data/raw/<venue>/<day>/<hour>.*.tsv.gz
 Binance WS -> BinanceFeed --+--> QualityMonitor / FeedDelayMonitor -> health json
 REST probes (metadata, clock, funding) -> meta records                        |
            +---------------------------------------------------------------+
                                                                               v
 research / lab (systemd, Nice 15, CPU+memory capped)          Replay (same parsers as live)
   build_dataset: per-UTC-day cached chunks  <------------------  MarketState -> FeatureEngine (grid ticks)
                                                                 LabelEngine (latency-delayed fills)
   run_research: purged walk-forward x models -> fit_policy (calibration LCB) -> simulate (portfolio)
   stats: cluster bootstrap, SPA, DSR, PBO -> choose_champion (simplest credible) -> report.md
   promotion_gate (+ reproduction run, vs champion) -> Registry (hash-verified bundles)
                                                                               |
 trader (systemd, paper|live)                                                  v
   feeds -> MarketState -> FeatureEngine.sample(t) -> DecisionEngine (10 steps) -> Allocator
         -> ExecutionOptimizer -> OrderManager (write-ahead SQLite) -> PaperGateway | LighterGateway
   reconciliation (REST), dead-man switch, kill file, margin state, EdgeMonitor, heartbeat
```

## Module map

| Area | Module | Responsibility |
|---|---|---|
| Core | `core/clock.py` | Monotonic-anchored local clock, NTP-style exchange offset estimation, feed delay and staleness |
| | `core/orderbook.py` | L2 book; executable walks by quantity/notional; impact |
| | `core/events.py` | Normalized events with exchange and local timestamps |
| Venues | `venues/lighter/*` | Market metadata and integer encoding, WS/REST parsing with nonce continuity, trading gateway |
| | `venues/binance/*` | USD-M combined streams, documented diff-depth synchronisation, REST |
| | `venues/mapping.py` | Instrument mapping validated by price ratio (contract multipliers) |
| Data | `collector/*` | Recorder, live service, data quality |
| | `data/replay.py`, `data/dataset.py` | Byte-identical replay, chunked feature/label datasets |
| | `data/synthetic.py` | Known-truth generator for methodology validation only |
| Signals | `features/state.py`, `features/engine.py` | Point-in-time state and features (one code path for research and live) |
| | `features/selection.py` | Block-IC stability, redundancy pruning (training data only) |
| | `labels/executable.py` | Execution-conditioned targets |
| | `latency/model.py` | Measured (or flagged prior) decision-to-fill latency |
| Models | `models/*` | Baselines, ridge, GBM, gated temporal GBM, secondary families |
| Research | `research/*` | Splits, calibrated policy, evaluation, regimes, statistics, pipeline |
| Decisions | `strategy/decision.py` | 10-step per-opportunity pipeline |
| | `portfolio/allocator.py` | Growth-optimal sizing under hard caps and exchange minimums |
| | `execution/optimizer.py`, `execution/order_manager.py` | Passive vs aggressive; order lifecycle |
| | `risk/guard.py` | Drawdown scaling, data checks, margin states, edge monitor |
| Ops | `live/*` | State store, trader daemon, integration test |
| | `lab/*` | Registry, promotion gate, Alpha Lab |
| | `report/proof.py` | Proof-of-advantage report |

## Key invariants

* **Point in time.** A feature at decision time *t* uses only events with local
  arrival time ≤ *t*. This is enforced by construction (one ordered event loop)
  and tested (`test_no_lookahead_features`).
* **Same code in research and live.** Parsers, `MarketState`, `FeatureEngine`,
  the policy, the allocator and the execution optimizer are shared.
* **Executable outcomes.** Labels walk the recorded book at the simulated fill
  time after a latency draw. Positions are never sized beyond the notional at
  which labels were computed.
* **Write-ahead state.** An order row exists before its transaction is signed,
  and reconciliation decides the fate of anything uncertain after a crash.
* **No fund movement.** The gateway exposes only order, cancel, cancel-all,
  leverage and auth. A static test forbids withdrawal/transfer references, and
  the wallet key is never on the host.
