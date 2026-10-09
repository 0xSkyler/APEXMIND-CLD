"""Order and position lifecycle with write-ahead persistence.

* Every order row is written (``pending``) before it is signed/sent.
* Entries: IOC limit capped at ``max_slippage_bps`` beyond the touch, or
  post-only at the touch cancelled after ``passive_wait_s``.
* Exits are mandatory: reduce-only IOC at the touch plus a cap that widens
  on every retry until the position is flat.
* Fill handling is idempotent: fills arrive as :class:`FillReport` from the
  paper gateway, from our prints on the public tape, or as deltas derived
  from authoritative exchange order state during reconciliation.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable

from apexmind.config import Config
from apexmind.core.clock import Clock
from apexmind.core.events import BUY
from apexmind.features.state import MarketState
from apexmind.live.state import StateStore
from apexmind.portfolio.allocator import OpenPosition
from apexmind.strategy.decision import Decision
from apexmind.venues.lighter.markets import LighterMarket
from apexmind.venues.lighter.trading import FillReport

log = logging.getLogger(__name__)

EXIT_CAPS_BPS = (10.0, 25.0, 60.0, 150.0, 400.0)


class OrderManager:
    def __init__(self, gateway, store: StateStore, markets: dict[str, LighterMarket], cfg: Config, clock: Clock,
                 mode: str, on_position_closed: Callable[[dict], None] | None = None,
                 on_exec_record: Callable[[dict], None] | None = None) -> None:
        self.gw, self.store, self.markets, self.cfg, self.clock, self.mode = gateway, store, markets, cfg, clock, mode
        self.on_position_closed = on_position_closed or (lambda p: None)
        self.on_exec_record = on_exec_record or (lambda r: None)
        self.exit_attempts: dict[str, int] = {}
        self.last_exit_ns: dict[str, int] = {}

    # -- queries ---------------------------------------------------------------
    def open_positions(self) -> list[OpenPosition]:
        out = []
        for p in self.store.positions(("opening", "open", "closing")):
            if p["qty"] > 0 or p["status"] == "opening":
                notional = (p["entry_notional"] or 0.0) or p["qty"] * (p["entry_price"] or 0.0)
                out.append(OpenPosition(p["symbol"], p["side"], notional, self.cfg.portfolio.max_gross_leverage,
                                        abs(p["lcb"] or 0.001) * 4))
        return out

    def busy_symbols(self) -> set[str]:
        return {p["symbol"] for p in self.store.positions(("opening", "open", "closing"))}

    # -- entries -----------------------------------------------------------------
    async def open_position(self, d: Decision, strategy: str, state: MarketState) -> str | None:
        m = self.markets[d.symbol]
        top = state.lit[d.symbol].top()
        if top is None:
            return None
        pid = uuid.uuid4().hex[:16]
        now = self.clock.now_ns()
        if d.exec_mode == "passive":
            price, kind = (top[0] if d.side == BUY else top[2]), "post_only"
        else:
            touch = top[2] if d.side == BUY else top[0]
            price, kind = touch * (1 + d.side * self.cfg.execution.max_slippage_bps * 1e-4), "ioc"
        self.store.upsert_position(position_id=pid, symbol=d.symbol, side=d.side, qty=0.0, entry_price=None,
                                   entry_notional=0.0, t_entry=None, horizon_s=d.horizon_s, exit_due_ns=None,
                                   status="opening", strategy=strategy, mu=d.mu, lcb=d.lcb, exec_mode=d.exec_mode,
                                   mode=self.mode)
        coi = await self._send(m, d.side, d.qty, price, kind, False, "entry", pid, d.t_ns)
        return pid if coi is not None else None

    async def _send(self, m: LighterMarket, side: int, qty: float, price: float, kind: str, reduce_only: bool,
                    purpose: str, pid: str, t_decision: int) -> int | None:
        coi = self.store.next_client_order_index()
        now = self.clock.now_ns()
        self.store.insert_order(client_order_index=coi, symbol=m.symbol, market_id=m.market_id, side=side, qty=qty,
                                price=price, order_type="limit", tif=kind, reduce_only=int(reduce_only), purpose=purpose,
                                position_id=pid, status="pending", mode=self.mode, t_decision=t_decision, t_created=now)
        res = await self.gw.create_order(m, coi, side, qty, price, kind, reduce_only, self.cfg.execution.order_expiry_s)
        if res.ok:
            self.store.update_order(coi, status="sent", tx_hash=res.tx_hash, t_sent=res.t_sent, t_ack=res.t_ack)
            self.store.record_latency(coi, "maker" if kind == "post_only" else "taker", t_decision=t_decision,
                                      t_sent=res.t_sent, t_ack=res.t_ack)
            return coi
        if res.error.startswith("exception"):
            # Outcome unknown (timeout etc.): reconciliation decides.
            self.store.update_order(coi, status="unknown", error=res.error, t_sent=res.t_sent)
            return coi
        self.store.update_order(coi, status="rejected", error=res.error, t_done=now)
        self.store.log_event("order_rejected", {"coi": coi, "error": res.error, "purpose": purpose})
        if purpose == "entry":
            self._close_position(pid, reason=f"entry_rejected: {res.error}")
        return None

    # -- fills -----------------------------------------------------------------------
    def on_fill(self, f: FillReport) -> None:
        o = self.store.order(f.client_order_index)
        if o is None:
            self.store.log_event("fill_unknown_order", f.__dict__)
            return
        if not self.store.insert_fill(f.trade_id, f.client_order_index, f.t_ns, f.price, f.qty, f.side, f.fee,
                                      f.liquidity):
            return  # duplicate delivery
        filled = o["filled_qty"] + f.qty
        avg = ((o["avg_price"] or 0.0) * o["filled_qty"] + f.price * f.qty) / filled
        done = f.done or filled >= o["qty"] - 1e-12
        self.store.update_order(f.client_order_index, filled_qty=filled, avg_price=avg, fees=(o["fees"] or 0) + f.fee,
                                status="filled" if done else "partially_filled", t_done=f.t_ns if done else None)
        p = self._position(o["position_id"])
        if p is None:
            return
        if o["purpose"] == "entry":
            qty = p["qty"] + f.qty
            entry = ((p["entry_price"] or 0.0) * p["qty"] + f.price * f.qty) / qty
            t_entry = p["t_entry"] or f.t_ns
            self.store.update_position(p["position_id"], qty=qty, entry_price=entry,
                                       entry_notional=qty * entry, t_entry=t_entry, status="open",
                                       exit_due_ns=int(t_entry + p["horizon_s"] * 1e9), fees=(p["fees"] or 0) + f.fee)
        else:
            qty = max(0.0, p["qty"] - f.qty)
            pnl = (p["pnl"] or 0.0) + p["side"] * (f.price - p["entry_price"]) * f.qty
            self.store.update_position(p["position_id"], qty=qty, pnl=pnl, fees=(p["fees"] or 0) + f.fee,
                                       exit_price=f.price, status="closing" if qty > 1e-12 else "open")
            if qty <= 1e-12:
                self._close_position(p["position_id"], reason="exited", t_exit=f.t_ns)

    def on_cancel(self, coi: int, reason: str) -> None:
        o = self.store.order(coi)
        if o is None or o["status"] in ("filled", "canceled", "rejected"):
            return
        self.store.update_order(coi, status="canceled", error=reason, t_done=self.clock.now_ns())
        p = self._position(o["position_id"])
        if p and o["purpose"] == "entry" and p["qty"] <= 1e-12:
            self._close_position(p["position_id"], reason=f"entry_unfilled: {reason}")

    def on_own_print(self, coi: int, t_ns: int) -> None:
        """First appearance of our fill on the public tape (latency end)."""
        o = self.store.order(coi)
        if o is None or o["t_first_fill_print"]:
            return
        self.store.update_order(coi, t_first_fill_print=t_ns)
        self.store.record_latency(coi, "maker" if o["tif"] == "post_only" else "taker", t_fill_print=t_ns)
        self.on_exec_record({"client_order_index": coi, "kind": "maker" if o["tif"] == "post_only" else "taker",
                             "t_decision": o["t_decision"], "t_sent": o["t_sent"], "t_ack": o["t_ack"],
                             "t_fill_print": t_ns, "symbol": o["symbol"], "mode": self.mode})

    # -- lifecycle management -------------------------------------------------------------
    async def manage(self, state: MarketState) -> None:
        now = self.clock.now_ns()
        wait_ns = int(self.cfg.labels.passive_wait_s * 1e9)
        for o in self.store.open_orders():
            if o["tif"] == "post_only" and o["purpose"] == "entry" and o["status"] in ("sent", "open") \
                    and now - (o["t_sent"] or now) > wait_ns:
                await self.gw.cancel_order(self.markets[o["symbol"]], o["exchange_order_index"] or o["client_order_index"])
        for p in self.store.positions(("open", "closing")):
            if p["qty"] <= 1e-12:
                continue
            if p["exit_due_ns"] and now >= p["exit_due_ns"] and not self._exit_working(p["position_id"]) \
                    and now >= self._next_exit_ns(p["position_id"]):
                await self._exit(p, state, "horizon")

    def _next_exit_ns(self, pid: str) -> int:
        """Exponential backoff between exit attempts (0.5 s doubling to 10 s)
        so a missing book or rejected exits cannot turn into order spam."""
        n = self.exit_attempts.get(pid, 0)
        if n == 0:
            return 0
        return self.last_exit_ns.get(pid, 0) + int(min(10.0, 0.5 * 2 ** (n - 1)) * 1e9)

    def _exit_working(self, pid: str) -> bool:
        return any(o["position_id"] == pid and o["purpose"] == "exit" for o in self.store.open_orders())

    async def _exit(self, p: dict, state: MarketState, reason: str) -> None:
        m = self.markets[p["symbol"]]
        top = state.lit[p["symbol"]].top()
        n = self.exit_attempts.get(p["position_id"], 0)
        cap = EXIT_CAPS_BPS[min(n, len(EXIT_CAPS_BPS) - 1)]
        self.exit_attempts[p["position_id"]] = n + 1
        side = -p["side"]
        self.last_exit_ns[p["position_id"]] = self.clock.now_ns()
        if top is None:
            if self.mode == "paper":
                return  # no book to simulate against; retry after backoff
            # live: our view of the book is missing but the exchange's is not
            ref = p["entry_price"]
            price = ref * (1 + side * EXIT_CAPS_BPS[-1] * 1e-4)
        else:
            touch = top[2] if side == BUY else top[0]
            price = touch * (1 + side * cap * 1e-4)
        self.store.update_position(p["position_id"], status="closing")
        await self._send(m, side, m.quantize_size(p["qty"]) or p["qty"], price, "ioc", True, "exit",
                         p["position_id"], self.clock.now_ns())
        self.store.log_event("exit_sent", {"position": p["position_id"], "reason": reason, "cap_bps": cap})

    async def flatten_all(self, state: MarketState, reason: str) -> None:
        await self.gw.cancel_all()
        for p in self.store.positions(("open", "closing")):
            if p["qty"] > 1e-12 and not self._exit_working(p["position_id"]):
                await self._exit(p, state, reason)

    # -- reconciliation (live) ---------------------------------------------------------------
    def apply_exchange_order(self, coi: int, filled_qty: float, filled_quote: float, status: str,
                             exchange_order_index: int | None = None) -> None:
        """Bring our order in line with authoritative exchange state."""
        o = self.store.order(coi)
        if o is None:
            self.store.log_event("unknown_exchange_order", {"coi": coi, "status": status})
            return
        if exchange_order_index is not None and o["exchange_order_index"] != exchange_order_index:
            self.store.update_order(coi, exchange_order_index=exchange_order_index)
        delta = filled_qty - (o["filled_qty"] or 0.0)
        if delta > 1e-12:
            prev_quote = (o["avg_price"] or 0.0) * (o["filled_qty"] or 0.0)
            px = (filled_quote - prev_quote) / delta if filled_quote > prev_quote else (o["avg_price"] or o["price"])
            m = self.markets[o["symbol"]]
            fee = delta * px * (m.maker_fee if o["tif"] == "post_only" else m.taker_fee)
            self.on_fill(FillReport(coi, self.clock.now_ns(), px, delta, o["side"], fee,
                                    "maker" if o["tif"] == "post_only" else "taker", f"recon-{coi}-{filled_qty:.10g}",
                                    status in ("filled", "canceled", "expired")))
        elif status in ("canceled", "expired", "rejected") or status.startswith("canceled"):
            self.on_cancel(coi, f"exchange:{status}")
        elif status == "open" and o["status"] in ("sent", "unknown", "pending"):
            self.store.update_order(coi, status="open")

    # -- helpers ----------------------------------------------------------------------------
    def _position(self, pid: str | None) -> dict | None:
        if pid is None:
            return None
        r = self.store.db.execute("SELECT * FROM positions WHERE position_id=?", (pid,)).fetchone()
        return dict(r) if r else None

    def _close_position(self, pid: str, reason: str, t_exit: int | None = None) -> None:
        p = self._position(pid)
        if p is None or p["status"] == "closed":
            return
        self.store.update_position(pid, status="closed", t_exit=t_exit or self.clock.now_ns())
        self.store.log_event("position_closed", {"position": pid, "reason": reason})
        self.exit_attempts.pop(pid, None)
        self.last_exit_ns.pop(pid, None)
        p = self._position(pid)
        if p and p["entry_notional"]:
            net = ((p["pnl"] or 0.0) - (p["fees"] or 0.0) - (p["funding"] or 0.0)) / p["entry_notional"]
            self.on_position_closed({**p, "net_return": net, "reason": reason})


def exec_record_meta(rec: dict) -> dict:
    """Shape of the ``meta/exec`` records the latency model consumes."""
    return {k: rec.get(k) for k in ("kind", "t_decision", "t_sent", "t_ack", "t_fill_print", "symbol", "mode")}


def _nan(x) -> float:
    return math.nan if x is None else x
