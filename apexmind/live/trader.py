"""Autonomous trading daemon (paper or live).

Event flow mirrors research exactly: events -> MarketState; on every
decision-grid tick -> FeatureEngine.sample -> DecisionEngine -> allocator ->
execution optimizer -> OrderManager. Same code, same features, same policy.

Safety properties:
* live mode refuses to start unless the champion bundle was validated on
  recorded data with measured latency and a recent exchange integration test
  passed for the same account;
* every order is persisted before it is sent and reconciled against the
  exchange afterwards; unknown exchange positions block trading in that
  market instead of being touched;
* an exchange-side scheduled cancel-all (dead-man switch) is re-armed
  continuously; positions are flattened on shutdown, kill-file, margin
  emergency or drawdown halt;
* no code path can withdraw or transfer funds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np

from apexmind.collector.recorder import RawRecorder
from apexmind.config import Config, load_lighter_api_key
from apexmind.core.clock import Clock, system_clock_status
from apexmind.core.events import BUY, LIGHTER, FeedStatus, Trade
from apexmind.execution.order_manager import OrderManager, exec_record_meta
from apexmind.features.engine import FeatureEngine
from apexmind.features.state import MarketState
from apexmind.lab.registry import Registry
from apexmind.latency.model import LatencyModel, build_latency_model
from apexmind.live.state import StateStore
from apexmind.risk.guard import EdgeMonitor, drawdown_multiplier, margin_state
from apexmind.strategy.decision import DecisionEngine, StrategyBundle
from apexmind.util import atomic_write_json
from apexmind.venues.lighter.markets import LighterMarket
from apexmind.venues.lighter.trading import FillReport, PaperGateway

log = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    pass


def live_preflight(cfg: Config, bundle_manifest: dict) -> None:
    """Refuse live trading without the required evidence and checks."""
    problems = []
    if bundle_manifest.get("data_kind") != "recorded":
        problems.append(f"champion validated on {bundle_manifest.get('data_kind')} data")
    if bundle_manifest.get("latency", {}).get("source") != "measured":
        problems.append("champion validated without measured execution latency")
    gate = bundle_manifest.get("gate", {})
    if not gate.get("passed") or gate.get("purpose") != "live":
        problems.append("champion did not pass the live promotion gate")
    p = Path(cfg.live.integration_report)
    if not p.exists():
        problems.append(f"no integration test report at {p}")
    else:
        rep = json.loads(p.read_text())
        age_h = (time.time() - rep.get("finished_at", 0)) / 3600
        if not rep.get("passed"):
            problems.append("last integration test failed")
        if age_h > cfg.live.integration_max_age_hours:
            problems.append(f"integration test is {age_h:.1f}h old")
        if rep.get("account_index") != cfg.lighter.account_index:
            problems.append("integration test was run for a different account")
        if not rep.get("order_lifecycle_tested"):
            problems.append("integration test did not exercise order placement/cancel/reconciliation")
    if problems:
        raise PreflightError("live trading refused: " + "; ".join(problems))


class CorrelationTracker:
    """EWMA correlation of per-tick reference returns between symbols."""

    def __init__(self, symbols: list[str], halflife_ticks: float) -> None:
        self.s = list(symbols)
        self.alpha = 1 - 0.5 ** (1 / max(halflife_ticks, 1.0))
        n = len(self.s)
        self.cov = np.eye(n) * 1e-8
        self.last = {s: math.nan for s in self.s}
        self.n = 0

    def update(self, mids: dict[str, float]) -> None:
        r = np.array([math.log(mids[s] / self.last[s]) if mids.get(s, 0) > 0 and self.last[s] > 0 else 0.0
                      for s in self.s])
        self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(r, r)
        self.last = {s: mids.get(s, math.nan) for s in self.s}
        self.n += 1

    def corr(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if self.n < 100:
            return 0.5  # conservative default until estimated
        i, j = self.s.index(a), self.s.index(b)
        d = math.sqrt(self.cov[i, i] * self.cov[j, j])
        return float(self.cov[i, j] / d) if d > 0 else 0.5


class Trader:
    def __init__(self, cfg: Config, bundle_id: str, bundle: StrategyBundle, markets: dict[str, LighterMarket],
                 eligible: set[str], ref_venue: str, mode: str, clock: Clock, gateway=None,
                 latency: LatencyModel | None = None, recorder: RawRecorder | None = None) -> None:
        self.cfg, self.mode, self.clock = cfg, mode, clock
        self.bundle_id, self.bundle = bundle_id, bundle
        self.markets = markets
        self.symbols = sorted(eligible)
        self.store = StateStore(cfg.live.state_db)
        self.recorder = recorder
        self.state = MarketState(self.symbols, ref_venue, cfg.features.depth_band_bps)
        self.fe = FeatureEngine(self.symbols, cfg.features, exec_latency_ms=lambda t: self.latency_now_ms())
        self.corr = CorrelationTracker(self.symbols, cfg.portfolio.correlation_halflife_s * 1000 / cfg.features.grid_ms)
        self.decider = DecisionEngine(cfg, bundle, markets, set(self.symbols), self.corr.corr)
        self.latency = latency or build_latency_model(self.store.latency_records(), "measured",
                                                      cfg.labels.prior_latency_median_ms, cfg.labels.prior_latency_sigma)
        if gateway is None:
            gateway = PaperGateway(self.state, markets, self.latency, clock, on_fill=lambda f: self.om.on_fill(f),
                                   on_cancel=lambda c, r: self.om.on_cancel(c, r))
        self.gw = gateway
        self.om = OrderManager(gateway, self.store, markets, cfg, clock, mode, self._on_position_closed,
                               self._on_exec_record)
        saved = self.store.get(f"edge:{bundle.name}")
        lcb = min((sp.lcb for sp in bundle.policy.sides.values() if sp is not None), default=0.0)
        self.edge = EdgeMonitor.restore(cfg.risk, saved) if saved else EdgeMonitor(cfg.risk, lcb)
        last = self.store.last_equity()
        self.equity = last["equity"] if last else (cfg.live.paper_equity if mode == "paper" else 0.0)
        self.high_water = max(self.store.high_water(), self.equity)
        self.maint_req = 0.0
        self.unreconciled: set[str] = set()
        self.entries_enabled = mode == "paper"  # live: enabled after the first clean reconciliation
        self.halted_reason = ""
        self.counters: Counter = Counter()
        self.grid_ns = cfg.features.grid_ms * 1_000_000
        self.next_tick: int | None = None
        self.started_ns = clock.now_ns()
        self.clock_unc_ms = 0.0 if mode == "paper" else self.refresh_clock_status()
        self._recover()

    def _recover(self) -> None:
        """Startup recovery for orders left in flight by a previous process.

        Paper: the simulated venue died with the process, so in-flight orders
        are cancelled (positions keep their exits, which ``manage`` re-sends).
        Live: their fate is decided by reconciliation against the exchange
        before any new entry is allowed (``entries_enabled`` stays False).
        """
        stale = self.store.open_orders()
        if not stale:
            return
        self.store.log_event("recovery", {"mode": self.mode, "open_orders": [o["client_order_index"] for o in stale]})
        if self.mode == "paper":
            for o in stale:
                self.om.on_cancel(o["client_order_index"], "process_restart")
        else:
            for o in stale:
                if o["status"] == "pending":
                    # never acknowledged as sent: may or may not have reached the exchange
                    self.store.update_order(o["client_order_index"], status="unknown")

    # -- inputs ---------------------------------------------------------------------
    def on_events(self, events: list) -> None:
        for ev in events:
            self.state.apply(ev)
            if isinstance(ev, Trade) and ev.venue == LIGHTER:
                if isinstance(self.gw, PaperGateway):
                    self.gw.on_trade(ev)
                else:
                    self._own_fill(ev)
            elif isinstance(ev, FeedStatus) and ev.status != "connected":
                self.counters[f"feed_{ev.status}"] += 1
        if isinstance(self.gw, PaperGateway):
            self.gw.poll()

    def _own_fill(self, t: Trade) -> None:
        acct = self.cfg.lighter.account_index
        for side, account, coi in ((BUY, t.bid_account, t.bid_client_id), (-BUY, t.ask_account, t.ask_client_id)):
            if account != acct or coi < 0:
                continue
            self.om.on_own_print(coi, t.ts_local_ns)
            m = self.markets[t.symbol]
            liq = "taker" if t.taker_side == side else "maker"
            fee = t.size * t.price * (m.taker_fee if liq == "taker" else m.maker_fee)
            self.om.on_fill(FillReport(coi, t.ts_local_ns, t.price, t.size, side, fee, liq, f"tape-{t.trade_id}-{side}",
                                       False))

    # -- callbacks ---------------------------------------------------------------------
    def _on_position_closed(self, p: dict) -> None:
        net = p["net_return"]
        if self.mode == "paper":
            self.equity += net * p["entry_notional"]
            self.high_water = max(self.high_water, self.equity)
            self.store.record_equity(self.clock.now_ns(), self.equity, self.equity, 0.0, "paper")
        action = self.edge.record(net)
        self.store.put(f"edge:{self.bundle.name}", self.edge.state())
        if action in ("reduce", "suspend"):
            self.store.log_event(f"edge_{action}", self.edge.state())
            log.warning("edge monitor: %s (%s)", action, self.edge.state())
        if self.edge.suspended:
            self.decider.suspended.add(self.bundle.name)

    def _on_exec_record(self, rec: dict) -> None:
        if self.recorder is not None:
            self.recorder.write_meta("exec", exec_record_meta(rec))

    # -- helpers ----------------------------------------------------------------------
    def latency_now_ms(self) -> float:
        recs = self.store.latency_records()[-200:]
        vals = [(r["t_fill_print"] - r["t_decision"]) / 1e6 for r in recs if r["t_fill_print"] and r["t_decision"]]
        return float(np.median(vals)) if len(vals) >= 10 else self.bundle.calib_latency_ms

    def refresh_clock_status(self) -> float:
        """Query the OS time-sync daemon (a subprocess): call from the
        housekeeping loop, not per tick. Unknown sync counts as uncertain."""
        st = system_clock_status()
        unc = abs(self.clock.wall_drift_ns()) / 1e6
        if "offset_ms" in st:
            unc += abs(float(st["offset_ms"])) + float(st.get("root_dispersion_ms", 0.0))
        self.clock_unc_ms = unc
        return unc

    def risk_multiplier(self) -> float:
        return drawdown_multiplier(self.equity, self.high_water, self.cfg.risk) * self.edge.multiplier

    # -- the decision tick ---------------------------------------------------------------
    async def tick(self, t: int) -> None:
        rows = self.fe.sample(t, self.state)
        self.corr.update({s: info["ref_mid"] for s, (_, info) in rows.items()})
        kill = Path(self.cfg.live.kill_file).exists()
        mstate = margin_state(self.maint_req, self.equity, self.cfg.risk) if self.mode == "live" else "ok"
        mult = self.risk_multiplier()
        dd_halt = drawdown_multiplier(self.equity, self.high_water, self.cfg.risk) <= 0
        if kill or mstate == "reduce" or dd_halt:
            reason = "kill_file" if kill else ("margin_reduce" if mstate == "reduce" else "drawdown_halt")
            if self.halted_reason != reason:
                self.halted_reason = reason
                self.store.log_event("halt", {"reason": reason})
                await self.om.flatten_all(self.state, reason)
        else:
            self.halted_reason = ""
        can_enter = self.entries_enabled and not self.halted_reason and mstate == "ok"
        open_pos = self.om.open_positions()
        busy = self.om.busy_symbols()
        clock_unc = self.clock_unc_ms if self.mode == "live" else 0.0
        self.decider.observe({s: v for s, (v, _) in rows.items()})
        for s, (_, info) in rows.items():
            if not info["valid"]:
                self.counters["step2:invalid_sample"] += 1
        todo = [s for s, (_, info) in rows.items()
                if can_enter and s not in busy and s not in self.unreconciled and info["valid"]]
        policy_rows = self.decider.policy_rows({s: rows[s][0] for s in todo}) if todo else {}
        for sym in todo:
            vals, info = rows[sym]
            lv, rv = self.state.lit[sym], self.state.ref[sym]
            names = self.fe.names
            idx = {n: i for i, n in enumerate(names)}
            imp = 0.5 * (vals[idx["lit_impact_buy_bps"]] + vals[idx["lit_impact_sell_bps"]])
            eta = (lv.next_funding_ns - t) / 1e9 if lv.next_funding_ns else math.nan
            d = self.decider.decide(
                t, sym, vals, info, equity=self.equity, open_positions=open_pos, risk_multiplier=mult,
                lit_staleness_ms=(t - lv.last_book_ns) / 1e6 if lv.last_book_ns else math.inf,
                ref_staleness_ms=(t - rv.last_book_ns) / 1e6 if rv.last_book_ns else math.inf,
                clock_uncertainty_ms=clock_unc, latency_ms_now=self.latency_now_ms(),
                tx_capacity=getattr(self.gw, "quota_remaining", math.inf), funding_rate=lv.funding_rate,
                funding_eta_s=eta, spread_bps=vals[idx["lit_spread_bps"]], impact_bps=imp,
                policy_row=policy_rows.get(sym))
            self.counters[f"step{d.step}:{d.reason}"] += 1
            if d.step >= 7:  # a signal existed: keep the full record
                self.store.record_decision(d.to_dict())
            if d.ok:
                pid = await self.om.open_position(d, self.bundle.name, self.state)
                if pid:
                    busy.add(sym)
                    open_pos = self.om.open_positions()
        await self.om.manage(self.state)

    async def run_ticks_until(self, t_end: int) -> None:
        """Advance the grid to ``t_end`` (used by the live loop and tests)."""
        if self.next_tick is None:
            self.next_tick = (self.clock.now_ns() // self.grid_ns + 1) * self.grid_ns
        while self.next_tick <= t_end:
            await self.tick(self.next_tick)
            self.next_tick += self.grid_ns

    def health(self) -> dict:
        return {"ts": self.clock.now_ns(), "mode": self.mode, "bundle": self.bundle_id, "equity": self.equity,
                "high_water": self.high_water, "halted": self.halted_reason, "entries_enabled": self.entries_enabled,
                "edge": {k: v for k, v in self.edge.state().items() if k != "window"},
                "open_positions": len(self.om.open_positions()), "unreconciled": sorted(self.unreconciled),
                "decisions": dict(self.counters.most_common(12)), "latency_ms": self.latency_now_ms()}


# ---------------------------------------------------------------------------------------
# live process wiring (network); exercised against the exchange by the integration test
# ---------------------------------------------------------------------------------------

async def _reconcile_live(tr: Trader, rest, gw) -> None:
    acct = tr.cfg.lighter.account_index
    res = await rest.account(acct)
    accounts = res.data.get("accounts") or [res.data]
    a = accounts[0]
    equity = float(a.get("total_asset_value") or a.get("collateral") or 0.0)
    maint = float(a.get("cross_maintenance_margin_requirement") or 0.0)
    tr.equity, tr.maint_req = equity, maint
    tr.high_water = max(tr.high_water, equity)
    tr.store.record_equity(tr.clock.now_ns(), equity, float(a.get("available_balance") or 0.0), maint, "exchange")
    exch_pos = {}
    for p in a.get("positions") or []:
        size = float(p.get("position") or 0.0) * (1 if int(p.get("sign", 1)) >= 0 else -1)
        if abs(size) > 0:
            exch_pos[p["symbol"]] = size
    auth = gw.auth_token()
    for m in {o["symbol"] for o in tr.store.open_orders()}:
        mk = tr.markets[m]
        act = await rest.active_orders(acct, mk.market_id, auth)
        live_cois = set()
        for o in act.data.get("orders") or []:
            coi = int(o["client_order_index"])
            live_cois.add(coi)
            tr.om.apply_exchange_order(coi, float(o["filled_base_amount"]), float(o["filled_quote_amount"]), "open",
                                       int(o["order_index"]))
        stale = [o for o in tr.store.open_orders() if o["symbol"] == m and o["client_order_index"] not in live_cois
                 and tr.clock.now_ns() - (o["t_sent"] or o["t_created"]) > 5e9]
        if stale:
            ina = await rest.inactive_orders(acct, auth, 100)
            final = {int(o["client_order_index"]): o for o in ina.data.get("orders") or []}
            for o in stale:
                f = final.get(o["client_order_index"])
                if f is not None:
                    tr.om.apply_exchange_order(o["client_order_index"], float(f["filled_base_amount"]),
                                               float(f["filled_quote_amount"]), str(f["status"]), int(f["order_index"]))
                elif tr.clock.now_ns() - (o["t_sent"] or o["t_created"]) > 60e9:
                    tr.om.on_cancel(o["client_order_index"], "not_found_on_exchange")
    ours: dict[str, float] = {}
    for p in tr.store.positions(("opening", "open", "closing")):
        ours[p["symbol"]] = ours.get(p["symbol"], 0.0) + p["side"] * p["qty"]
    tr.unreconciled = set()
    for sym in set(exch_pos) | set(ours):
        if sym not in tr.markets:
            continue
        lot = tr.markets[sym].lot
        if abs(exch_pos.get(sym, 0.0) - ours.get(sym, 0.0)) > lot / 2:
            working = any(o["symbol"] == sym for o in tr.store.open_orders())
            if not working:
                tr.unreconciled.add(sym)
                tr.store.log_event("position_mismatch", {"symbol": sym, "exchange": exch_pos.get(sym, 0.0),
                                                         "ours": ours.get(sym, 0.0)})
    tr.entries_enabled = True


async def run_trader(cfg: Config, mode: str) -> None:
    from apexmind.collector.service import Collector  # reuses bootstrap (markets + mapping)
    from apexmind.venues.feeds import BinanceFeed, LighterFeed
    from apexmind.venues.lighter.trading import LighterGateway

    reg = Registry(cfg.lab.registry_dir)
    champ = reg.load_champion()
    if champ is None:
        raise PreflightError("no champion bundle registered: nothing has earned the right to trade")
    bid, bundle = champ
    if mode == "live":
        live_preflight(cfg, reg.manifest(bid))
    clock = Clock()
    boot = Collector(cfg)
    boot.recorder = RawRecorder(cfg.live.record_dir, clock, session_id=f"trader-{mode}-{int(time.time())}")
    boot.clock = clock
    market_ids, ref_symbols = await boot.bootstrap()
    from apexmind.venues.lighter.markets import parse_order_book_details

    details = await boot.lrest.order_book_details()
    markets = {m.symbol: m for m in parse_order_book_details(details.data).values()}
    eligible = {i.symbol for i in boot.instruments if i.eligible} & set(bundle.manifest.get("markets", {}) or markets)
    gw = None
    if mode == "live":
        gw = LighterGateway(cfg.lighter, load_lighter_api_key(cfg.live), clock)
        await gw.start()
    tr = Trader(cfg, bid, bundle, markets, eligible, cfg.reference.venue, mode, clock, gateway=gw,
                recorder=boot.recorder)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    lfeed = LighterFeed(cfg.lighter.ws_url, market_ids, boot.lparser, clock, lambda *a: None, tr.on_events,
                        cfg.lighter.ws_max_subscriptions_per_connection)
    bfeed = BinanceFeed(cfg.reference.ws_url, ref_symbols, boot.bparser, boot.brest, clock, lambda *a: None,
                        tr.on_events, cfg.reference.depth_stream_ms, market_ws_url=cfg.reference.market_ws_url)

    async def ticker():
        while not stop.is_set():
            now = clock.now_ns()
            await tr.run_ticks_until(now)
            await asyncio.sleep(max(0.0, (tr.next_tick - clock.now_ns()) / 1e9))

    async def housekeeping():
        while not stop.is_set():
            if mode == "live":
                tr.refresh_clock_status()
                try:
                    await _reconcile_live(tr, boot.lrest, gw)
                    await gw.schedule_cancel_all(int(time.time() * 1000) + cfg.live.deadman_seconds * 1000)
                except Exception as e:
                    tr.store.log_event("reconcile_error", repr(e))
                    log.warning("reconcile failed: %r", e)
            atomic_write_json(cfg.live.heartbeat_path, tr.health())
            boot.recorder.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.live.reconcile_interval_s)
            except asyncio.TimeoutError:
                pass

    tasks = [asyncio.create_task(c) for c in (lfeed.run(stop), bfeed.run(stop), ticker(), housekeeping())]
    await stop.wait()
    tr.entries_enabled = False
    if cfg.live.flatten_on_shutdown:
        await tr.om.flatten_all(tr.state, "shutdown")
        await asyncio.sleep(3.0)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    boot.recorder.close()
    if gw is not None:
        await gw.close()
    tr.store.close()
    await boot.lrest.close()
    await boot.brest.close()
    os.sync()
