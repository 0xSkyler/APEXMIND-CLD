"""Exchange integration test suite (run on the deployment host).

Phases (each later phase only runs if requested and earlier ones passed):

1. public data: REST metadata, instrument mapping, Lighter and reference
   WebSocket books (snapshot + continuous updates, never crossed), clock;
2. account: API key registered to the account, auth token, account and
   order queries, nonce;
3. order lifecycle (``--place-test-order``): a minimum-size post-only order
   far from the touch (cannot fill) is placed, found by client order index,
   cancelled and confirmed gone;
4. latency calibration (``--latency-samples N``): N minimum-size IOC round
   trips (open then reduce-only close). Each costs roughly spread + fees on
   the exchange minimum notional. Fill prints are matched on the public tape
   by client order index and recorded as ``meta/exec`` records, which is how
   research obtains *measured* execution latency.

The report (``LiveConfig.integration_report``) gates live trading.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

from apexmind.collector.recorder import RawRecorder
from apexmind.config import Config, load_lighter_api_key
from apexmind.core.clock import Clock, OffsetEstimator, system_clock_status
from apexmind.core.events import BUY, SELL, BookDelta, BookSnapshot, FeedStatus, MarketStats, Trade
from apexmind.core.orderbook import L2Book
from apexmind.util import atomic_write_json
from apexmind.venues.binance.parse import VENUE as BINANCE, BinanceParser
from apexmind.venues.binance.rest import BinanceRest
from apexmind.venues.feeds import BinanceFeed, LighterFeed
from apexmind.venues.lighter.markets import parse_order_book_details
from apexmind.venues.lighter.parse import LighterParser
from apexmind.venues.lighter.rest import LighterRest
from apexmind.venues.mapping import build_instruments


class Report:
    def __init__(self, cfg: Config) -> None:
        self.data = {"started_at": time.time(), "account_index": cfg.lighter.account_index, "checks": [],
                     "assumptions": {}, "latency_samples": [], "order_lifecycle_tested": False}

    def check(self, name: str, ok: bool, detail="") -> bool:
        self.data["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.data["checks"])


async def _collect(feeds, until, timeout_s: float) -> bool:
    stop = asyncio.Event()
    tasks = [asyncio.create_task(f.run(stop)) for f in feeds]
    t0 = time.monotonic()
    ok = False
    while time.monotonic() - t0 < timeout_s:
        if until():
            ok = True
            break
        await asyncio.sleep(0.2)
    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    return ok


async def run_integration_test(cfg: Config, place_test_order: bool = False, latency_samples: int = 0,
                               out_path: str | None = None) -> dict:
    rep = Report(cfg)
    clock = Clock()
    lrest, brest = LighterRest(cfg.lighter, clock), BinanceRest(cfg.reference, clock)
    try:
        await _public_phase(cfg, rep, clock, lrest, brest)
        if rep.ok and cfg.lighter.account_index >= 0:
            gw = await _account_phase(cfg, rep, clock, lrest)
            if rep.ok and gw is not None and (place_test_order or latency_samples):
                await _order_phase(cfg, rep, clock, lrest, gw, latency_samples)
            if gw is not None:
                await gw.close()
        elif cfg.lighter.account_index < 0:
            rep.check("account_configured", False, "lighter.account_index not set; account phase skipped")
    except Exception as e:  # any unexpected failure fails the suite
        rep.check("unexpected_error", False, repr(e))
    finally:
        await lrest.close()
        await brest.close()
    rep.data["finished_at"] = time.time()
    rep.data["passed"] = rep.ok
    atomic_write_json(out_path or cfg.live.integration_report, rep.data)
    return rep.data


async def _public_phase(cfg, rep: Report, clock, lrest, brest) -> None:
    details = await lrest.order_book_details()
    markets = parse_order_book_details(details.data)
    rep.check("lighter_metadata", len(markets) > 0, f"{len(markets)} perp markets")
    prices = await brest.prices()
    inst = [i for i in build_instruments(markets, prices, cfg.instruments, BINANCE) if i.eligible]
    if not rep.check("instrument_mapping", len(inst) > 0, ", ".join(f"{i.symbol}->{i.ref_symbol}" for i in inst)):
        return
    target = inst[0]
    rep.data["test_market"] = target.to_dict()
    lp = LighterParser({target.market_id: target.symbol})
    bp = BinanceParser({target.ref_symbol: target.symbol}, multipliers={target.ref_symbol: target.price_multiplier})
    seen = {"l_snap": 0, "l_upd": 0, "l_gap": 0, "crossed": 0, "b_snap": 0, "b_upd": 0, "stats": None}
    book = L2Book()

    def on_events(evs):
        for e in evs:
            if isinstance(e, BookSnapshot) and e.venue == "lighter":
                seen["l_snap"] += 1
                book.apply_snapshot(e.bids, e.asks)
            elif isinstance(e, BookDelta) and e.venue == "lighter":
                seen["l_upd"] += 1
                book.apply_delta(e.bids, e.asks)
                seen["crossed"] += int(book.is_crossed())
            elif isinstance(e, FeedStatus) and e.status == "gap" and e.venue == "lighter":
                seen["l_gap"] += 1
            elif isinstance(e, BookSnapshot) and e.venue == BINANCE:
                seen["b_snap"] += 1
            elif isinstance(e, BookDelta) and e.venue == BINANCE:
                seen["b_upd"] += 1
            elif isinstance(e, MarketStats) and e.venue == "lighter":
                seen["stats"] = e

    lf = LighterFeed(cfg.lighter.ws_url, [target.market_id], lp, clock, lambda *a: None, on_events)
    bf = BinanceFeed(cfg.reference.ws_url, [target.ref_symbol], bp, brest, clock, lambda *a: None, on_events)
    ok = await _collect([lf, bf], lambda: seen["l_snap"] and seen["l_upd"] >= 20 and seen["b_snap"]
                        and seen["b_upd"] >= 20 and seen["stats"] is not None, 45.0)
    rep.check("websocket_books", ok, json.dumps({k: v for k, v in seen.items() if k != "stats"}))
    rep.check("lighter_book_continuity", seen["l_gap"] == 0 and seen["crossed"] == 0,
              f"gaps={seen['l_gap']} crossed_updates={seen['crossed']}")
    off = OffsetEstimator()
    for _ in range(5):
        off.add_probe(*await brest.server_time_probe())
    rep.check("reference_clock_offset", abs(off.offset_ns) / 1e6 < cfg.risk.max_clock_uncertainty_ms,
              f"offset {off.offset_ns / 1e6:.1f} ms (+/- {off.uncertainty_ns / 1e6:.1f})")
    rep.data["system_clock"] = system_clock_status()
    # Protocol assumptions that the code relies on, recorded for review.
    fr = await lrest.funding_rates()
    rest_rate = next((r.get("rate") for r in (fr.data.get("funding_rates") or [])
                      if r.get("market_id") == target.market_id and r.get("exchange", "lighter") == "lighter"), None)
    ws_rate = seen["stats"].funding_rate if seen["stats"] else None
    rep.data["assumptions"]["funding_rate_units"] = {
        "ws_market_stats_parsed_fraction": ws_rate, "rest_funding_rates_raw": rest_rate,
        "note": "parser assumes WS funding is in percent per interval; confirm the two agree in magnitude"}
    rep.data["assumptions"]["fee_fields"] = {"taker_fee_raw": details.data["order_book_details"][0].get("taker_fee"),
                                             "parsed_as": "percent"}


async def _account_phase(cfg, rep: Report, clock, lrest):
    from apexmind.venues.lighter.trading import LighterGateway

    try:
        key = load_lighter_api_key(cfg.live)
    except Exception as e:
        rep.check("api_key_loaded", False, repr(e))
        return None
    gw = LighterGateway(cfg.lighter, key, clock)
    try:
        await gw.start(verify=True)
        rep.check("api_key_registered", True, f"account {cfg.lighter.account_index} key {cfg.lighter.api_key_index}")
    except Exception as e:
        rep.check("api_key_registered", False, repr(e))
        return None
    auth = gw.auth_token()
    acct = await lrest.account(cfg.lighter.account_index)
    a = (acct.data.get("accounts") or [acct.data])[0]
    rep.check("account_query", "collateral" in a or "total_asset_value" in a,
              f"collateral={a.get('collateral')} positions={len(a.get('positions') or [])}")
    lim = await lrest.account_limits(cfg.lighter.account_index, auth)
    rep.data["account_tier"] = {k: lim.data.get(k) for k in ("user_tier", "user_tier_name", "current_maker_fee_tick",
                                                              "current_taker_fee_tick")}
    rep.check("authenticated_query", lim.status == 200, json.dumps(rep.data["account_tier"]))
    nonce = await lrest.next_nonce(cfg.lighter.account_index, cfg.lighter.api_key_index)
    rep.check("nonce", nonce >= 0, f"next nonce {nonce}")
    return gw


async def _order_phase(cfg, rep: Report, clock, lrest, gw, latency_samples: int) -> None:
    t = rep.data["test_market"]
    details = await lrest.order_book_details(t["market_id"])
    m = next(iter(parse_order_book_details(details.data).values()))
    lp = LighterParser({m.market_id: m.symbol}, cfg.lighter.account_index)
    book = L2Book()
    prints: dict[int, int] = {}
    tape_fields = {"client_ids_present": False}

    def on_events(evs):
        for e in evs:
            if isinstance(e, BookSnapshot):
                book.apply_snapshot(e.bids, e.asks)
            elif isinstance(e, BookDelta):
                book.apply_delta(e.bids, e.asks)
            elif isinstance(e, Trade) and lp.is_own_fill(e):
                coi = e.bid_client_id if e.bid_account == cfg.lighter.account_index else e.ask_client_id
                if coi >= 0:
                    tape_fields["client_ids_present"] = True
                    prints.setdefault(coi, e.ts_local_ns)

    feed = LighterFeed(cfg.lighter.ws_url, [m.market_id], lp, clock, lambda *a: None, on_events)
    stop = asyncio.Event()
    task = asyncio.create_task(feed.run(stop))
    try:
        for _ in range(100):
            if book.usable():
                break
            await asyncio.sleep(0.2)
        if not rep.check("order_book_ready", book.usable(), "book for test market"):
            return
        auth = gw.auth_token()
        # --- post-only far from the touch: cannot fill ---
        bid = book.best_bid()[0]
        px = bid * 0.9
        qty = m.min_feasible_qty(px)
        coi = int(time.time() * 1000) % 10**12
        r = await gw.create_order(m, coi, BUY, qty, px, "post_only", False, 120)
        rep.check("post_only_submitted", r.ok, f"tx {r.tx_hash} ack {(r.t_ack - r.t_sent) / 1e6:.1f} ms {r.error}")
        found = None
        for _ in range(30):
            act = await lrest.active_orders(cfg.lighter.account_index, m.market_id, auth)
            found = next((o for o in act.data.get("orders") or [] if int(o["client_order_index"]) == coi), None)
            if found:
                break
            await asyncio.sleep(0.5)
        if not rep.check("order_visible", found is not None, f"client_order_index {coi}"):
            return
        c = await gw.cancel_order(m, int(found["order_index"]))
        gone = False
        for _ in range(30):
            act = await lrest.active_orders(cfg.lighter.account_index, m.market_id, auth)
            if not any(int(o["client_order_index"]) == coi for o in act.data.get("orders") or []):
                gone = True
                break
            await asyncio.sleep(0.5)
        rep.check("order_cancelled", c.ok and gone, f"cancel tx {c.tx_hash} {c.error}")
        rep.data["order_lifecycle_tested"] = rep.ok
        # --- latency calibration round trips (explicit opt-in) ---
        if latency_samples > 0:
            rec = RawRecorder(cfg.live.record_dir, clock, session_id=f"latency-{int(time.time())}")
            n_ok = 0
            for i in range(latency_samples):
                ask = book.best_ask()[0]
                q = m.min_feasible_qty(ask)
                coi_in = coi + 1 + 2 * i
                t_dec = clock.now_ns()
                r1 = await gw.create_order(m, coi_in, BUY, q, ask * 1.002, "ioc", False)
                await asyncio.sleep(3.0)
                bid = book.best_bid()[0]
                r2 = await gw.create_order(m, coi_in + 1, SELL, q, bid * 0.998, "ioc", True)
                await asyncio.sleep(1.0)
                if r1.ok and coi_in in prints:
                    n_ok += 1
                    sample = {"kind": "taker", "t_decision": t_dec, "t_sent": r1.t_sent, "t_ack": r1.t_ack,
                              "t_fill_print": prints[coi_in], "symbol": m.symbol, "mode": "calibration"}
                    rep.data["latency_samples"].append(sample)
                    rec.write_meta("exec", sample)
                if not r2.ok:
                    rep.check("calibration_close", False, f"reduce-only close failed: {r2.error}")
                    break
            rec.close()
            lat = [(s["t_fill_print"] - s["t_decision"]) / 1e6 for s in rep.data["latency_samples"]]
            rep.check("latency_calibration", n_ok >= max(1, latency_samples // 2),
                      f"{n_ok}/{latency_samples} fills matched on tape; median "
                      f"{(sorted(lat)[len(lat) // 2] if lat else math.nan):.0f} ms")
            rep.data["assumptions"]["tape_client_ids"] = tape_fields
            acct = await lrest.account(cfg.lighter.account_index)
            a = (acct.data.get("accounts") or [acct.data])[0]
            pos = [p for p in a.get("positions") or [] if int(p.get("market_id", -1)) == m.market_id
                   and float(p.get("position") or 0) != 0]
            rep.check("flat_after_calibration", not pos, f"residual position {pos[0] if pos else 0}")
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def main_integration(cfg: Config, place_test_order: bool, latency_samples: int) -> int:
    rep = asyncio.run(run_integration_test(cfg, place_test_order, latency_samples))
    print(json.dumps({"passed": rep["passed"], "report": str(Path(cfg.live.integration_report).resolve())}))
    return 0 if rep["passed"] else 1
