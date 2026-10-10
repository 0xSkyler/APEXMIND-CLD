# Protocol assumptions and known limitations

## Exchange protocol assumptions (verify on first deployment)

The Lighter API surface was taken from the official `lighter-sdk` 1.1.6
(endpoints, order types, time-in-force codes, signer, nonce handling,
response fields). The live API was **not reachable** from the development
environment, so the following details are assumptions. The integration test
checks or records each one, and the parsers fail safe where they can.

| # | Assumption | Where | Safeguard |
|---|---|---|---|
| 1 | `order_book` updates carry `nonce` and `begin_nonce`, and continuity means `begin_nonce == previous nonce` | `venues/lighter/parse.py` | On mismatch the book is invalidated and resubscribed. If the fields are absent, continuity is not checked, but the 30-minute resync and crossed-book detection still apply. The integration test counts gaps and crossed updates. |
| 2 | `market_stats.current_funding_rate` is in percent per funding interval | `LighterParser._stats` | The integration test records the WS value next to REST `funding-rates` for comparison |
| 3 | `orderBookDetails.taker_fee` / `maker_fee` are in percent | `LighterMarket.from_detail` | `lighter.fee_source: config` overrides with your account tier's verified fees |
| 4 | `is_maker_ask = true` means the taker bought | `LighterParser._trades` | Unit-tested; verify with the latency calibration fills |
| 5 | Public trades include `bid_client_id` / `ask_client_id` | own-fill detection | Without them, fills still arrive through REST reconciliation, but tape-based latency cannot be measured. The integration test reports `tape_client_ids`. |
| 6 | REST `account` returns `accounts[0]` with `total_asset_value`, `cross_maintenance_margin_requirement`, and `positions[].position/sign` | `live/trader.py` reconciliation | Exceptions are logged and entries stay disabled until reconciliation succeeds |
| 7 | Margin fractions are in 1/10 000 units | `markets.py` | Consistent with the SDK (`imf = 10_000 / leverage`) |
| 8 | Order expiry is a millisecond timestamp, 5 min to 30 days ahead (lighter-go `MinOrderExpiryPeriod`); IOC uses 0 | gateway `order_expiry_ms` clamps with a 1 min margin | Verified by the post-only lifecycle test |
| 9 | Default rate limits (50 REST/min, 4 tx/s) are within your tier | config | Tune after checking Lighter's limits for your account tier |
| 10 | Binance USD-M serves book streams on `/public` and `aggTrade`/`markPrice` on `/market` (combined `.../stream?streams=`) | `reference.ws_url`, `reference.market_ws_url` | The integration test requires Binance trades and mark-price updates; the collector status shows per-stream `trades` counts |

## Methodological limitations

* **Label notional.** Executable outcomes are computed at one notional (default
  $1,000). Orders larger than that are never placed, because costs beyond the
  measured depth are not extrapolated. Raise `labels.notional_usd` and rebuild
  to study larger sizes.
* **Passive fills** are conservative: back of the queue, no queue improvement
  from cancels, no partial fills. Real passive performance may be better.
  This model can understate the edge but should not overstate it.
* **Outcome distribution granularity.** The expected net return is
  calibrated per decision (isotonic regression of realized on predicted). The
  probability of a favorable outcome, return quantiles, Expected Shortfall,
  adverse excursion and significant-slippage probability are estimated over the
  calibration trades selected by each side's policy, not predicted per row.
  Conditional per-row distribution models (for example quantile GBMs) would
  sharpen sizing, but each would add another model that must earn its place.
* **Own market impact** on later book states is not simulated. That's
  negligible at small notional but matters at size.
* **Latency prior.** Until at least 30 live samples exist, labels use a
  conservative lognormal prior (median 400 ms). Reports flag this, and live
  promotion is impossible under it.
* **Single reference venue** (Binance USD-M) is implemented. The adapter
  structure supports adding others.
* **Strategy families.** Liquidity-shock reversion and volatility-regime
  momentum are implemented and gated behind a credible primary champion.
  Funding-aware positioning, cross-asset relative value, event-driven and
  liquidity-provision strategies need different horizons or data and are
  **not implemented**.
* **Temporal models** are limited to a lag-feature GBM that must beat the
  plain GBM on held-out data to be used. Deep sequence models are not
  implemented.
* **Throughput.** The research path is pure Python, at roughly 25 s per
  symbol-hour for the first build. Day-chunk caching makes later builds
  incremental.

## Live-path verification status

| Component | Verified how |
|---|---|
| Transaction signing and encoding | Real `lighter-sdk` signer, offline (`tests/test_gateway_signing.py`) |
| Order lifecycle, fills, exits, persistence, restart, kill switch | Paper gateway driven by replayed data (`tests/test_trader_paper.py`) |
| WebSocket handling (subscribe, ping/pong, parse) | Local WebSocket server (`tests/test_venues.py`) |
| Against the real exchange | **Not yet.** Requires `apexmind integration-test` on a host with access |
