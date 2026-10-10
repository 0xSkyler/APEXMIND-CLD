"""Lighter transaction gateway (live) and a paper gateway with the same API.

The live gateway wraps the official ``lighter-sdk`` signer and exposes only:
create order, cancel order, cancel-all (immediate or scheduled as a dead-man
switch), leverage updates and read-auth tokens. It holds no reference to the
SDK's withdrawal or transfer paths, and the process never has access to the
Ethereum wallet key (only the API key), so it cannot produce L1-signed
transfers. ``tests/test_no_withdrawals.py`` enforces this statically.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from apexmind.config import LighterConfig
from apexmind.core.clock import Clock
from apexmind.core.events import BUY, SELL, Trade
from apexmind.core.ratelimit import TokenBucket
from apexmind.features.state import MarketState
from apexmind.latency.model import LatencyModel
from apexmind.venues.lighter.markets import LighterMarket

ORDER_TYPE_LIMIT = 0
# lighter-go txtypes: resting-order expiry must be 5 min..30 days from now (ms).
# The margin absorbs signing/network time and local clock error.
MIN_ORDER_EXPIRY_S = 5 * 60 + 60
MAX_ORDER_EXPIRY_S = 30 * 86400 - 3600


def order_expiry_ms(kind: str, expiry_s: int, now_ms: int) -> int:
    """Exchange expiry field: 0 for IOC, otherwise clamped into Lighter's window."""
    if kind == "ioc":
        return 0
    return now_ms + min(max(int(expiry_s), MIN_ORDER_EXPIRY_S), MAX_ORDER_EXPIRY_S) * 1000
ORDER_TYPE_MARKET = 1
TIF_IOC = 0
TIF_GTT = 1
TIF_POST_ONLY = 2
CANCEL_ALL_IMMEDIATE = 0
CANCEL_ALL_SCHEDULED = 1
CROSS_MARGIN = 0


@dataclass
class TxResult:
    ok: bool
    tx_hash: str = ""
    error: str = ""
    t_sent: int = 0
    t_ack: int = 0
    predicted_ms: float = math.nan
    quota_remaining: float = math.inf


@dataclass
class FillReport:
    """Execution report delivered to the order manager by either gateway."""

    client_order_index: int
    t_ns: int  # local time we observed the fill
    price: float
    qty: float
    side: int
    fee: float
    liquidity: str  # "taker" | "maker"
    trade_id: str
    done: bool  # order finished (filled or remainder cancelled)


class LighterGateway:
    """Live gateway. Construct and ``await start()`` inside the event loop."""

    def __init__(self, cfg: LighterConfig, api_private_key: str, clock: Clock) -> None:
        if cfg.account_index < 0 or cfg.api_key_index < 0:
            raise ValueError("account_index and api_key_index must be configured for live trading")
        self.cfg, self.clock = cfg, clock
        self._key = api_private_key
        self._signer = None
        self.bucket = TokenBucket(cfg.max_tx_per_second, burst=max(1.0, cfg.max_tx_per_second))
        self.quota_remaining = math.inf

    async def start(self, verify: bool = True) -> None:
        """Create the signer; ``verify`` checks the key is registered to the
        account on Lighter (network)."""
        from lighter.signer_client import SignerClient  # optional dependency

        self._signer = SignerClient(url=self.cfg.api_url, account_index=self.cfg.account_index,
                                    api_private_keys={self.cfg.api_key_index: self._key}, chain_id=self.cfg.chain_id)
        self._key = ""  # the signer keeps its own copy; do not hold it twice
        if verify:
            err = self._signer.check_client()
            if err is not None:
                raise RuntimeError(f"API key not valid for account: {err}")

    async def close(self) -> None:
        if self._signer is not None:
            await self._signer.close()

    def auth_token(self, ttl_s: int = 600) -> str:
        token, err = self._signer.create_auth_token_with_expiry(ttl_s, api_key_index=self.cfg.api_key_index)
        if err:
            raise RuntimeError(f"auth token: {err}")
        return token

    async def _send(self, coro_fn: Callable) -> TxResult:
        await self.bucket.acquire()
        t0 = self.clock.now_ns()
        try:
            _tx, resp, err = await coro_fn()
        except Exception as e:  # network or SDK error: outcome unknown until reconciled
            return TxResult(False, error=f"exception: {e!r}", t_sent=t0, t_ack=self.clock.now_ns())
        t1 = self.clock.now_ns()
        if err or resp is None or getattr(resp, "code", 200) != 200:
            return TxResult(False, error=str(err or getattr(resp, "message", "")), t_sent=t0, t_ack=t1)
        q = getattr(resp, "volume_quota_remaining", None)
        if q is not None:
            self.quota_remaining = float(q)
        return TxResult(True, resp.tx_hash, "", t0, t1, float(getattr(resp, "predicted_execution_time_ms", math.nan)),
                        self.quota_remaining)

    async def create_order(self, m: LighterMarket, coi: int, side: int, qty: float, price: float, kind: str,
                           reduce_only: bool = False, expiry_s: int = 600) -> TxResult:
        tif = {"ioc": TIF_IOC, "post_only": TIF_POST_ONLY, "gtt": TIF_GTT}[kind]
        expiry = order_expiry_ms(kind, expiry_s, int(time.time() * 1000))
        s = self._signer
        return await self._send(lambda: s.create_order(
            m.market_id, coi, m.size_to_int(qty), m.price_to_int(price, side_is_buy=(side == BUY)), side == SELL,
            ORDER_TYPE_LIMIT, tif, reduce_only, 0, expiry))

    async def cancel_order(self, m: LighterMarket, order_index: int) -> TxResult:
        s = self._signer
        return await self._send(lambda: s.cancel_order(m.market_id, order_index))

    async def cancel_all(self) -> TxResult:
        s = self._signer
        return await self._send(lambda: s.cancel_all_orders(CANCEL_ALL_IMMEDIATE, 0))

    async def schedule_cancel_all(self, at_ms: int) -> TxResult:
        """Dead-man switch: the exchange cancels all resting orders at
        ``at_ms`` unless re-armed later."""
        s = self._signer
        return await self._send(lambda: s.cancel_all_orders(CANCEL_ALL_SCHEDULED, at_ms))

    async def update_leverage(self, m: LighterMarket, leverage: float) -> TxResult:
        s = self._signer
        return await self._send(lambda: s.update_leverage(m.market_id, CROSS_MARGIN, leverage))


@dataclass
class _PaperOrder:
    coi: int
    symbol: str
    side: int
    qty: float
    price: float
    kind: str
    reduce_only: bool
    active_at: int
    expires_at: int
    queue: float = math.nan
    filled: float = 0.0
    canceled: bool = False
    fee_taker: float = 0.0
    fee_maker: float = 0.0
    notes: list = field(default_factory=list)


class PaperGateway:
    """Simulated execution against live market data.

    Orders become active after a latency draw; IOC orders walk the then-
    current Lighter book within their limit; post-only orders rest at their
    price and fill only when the tape trades through it or exhausts the
    displayed queue ahead (same conservative rule as the research labels).
    """

    def __init__(self, state: MarketState, markets: dict[str, LighterMarket], latency: LatencyModel, clock: Clock,
                 on_fill: Callable[[FillReport], None], on_cancel: Callable[[int, str], None], seed: int = 0,
                 fee_override: tuple[float, float] | None = None) -> None:
        self.state, self.markets, self.lat, self.clock = state, markets, latency, clock
        self.on_fill, self.on_cancel = on_fill, on_cancel
        self.rng = np.random.default_rng(seed)
        self.fee_override = fee_override
        self._pending: list = []  # (active_at, coi)
        self.orders: dict[int, _PaperOrder] = {}
        self.quota_remaining = math.inf

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _fees(self, m: LighterMarket) -> tuple[float, float]:
        return self.fee_override or (m.taker_fee, m.maker_fee)

    async def create_order(self, m: LighterMarket, coi: int, side: int, qty: float, price: float, kind: str,
                           reduce_only: bool = False, expiry_s: int = 300) -> TxResult:
        now = self.clock.now_ns()
        lat = int(float(self.lat.sample_ms(self.rng)) * 1e6)
        taker, maker = self._fees(m)
        o = _PaperOrder(coi, m.symbol, side, qty, price, kind, reduce_only, now + lat, now + expiry_s * 10**9,
                        fee_taker=taker, fee_maker=maker)
        self.orders[coi] = o
        heapq.heappush(self._pending, (o.active_at, coi))
        return TxResult(True, f"paper-{coi}", "", now, now)

    async def cancel_order(self, m: LighterMarket, order_index: int) -> TxResult:
        o = self.orders.get(order_index)
        now = self.clock.now_ns()
        if o is not None and not o.canceled and o.filled < o.qty:
            o.canceled = True
            self.on_cancel(o.coi, "canceled")
        return TxResult(True, f"paper-cancel-{order_index}", "", now, now)

    async def cancel_all(self) -> TxResult:
        for coi in list(self.orders):
            await self.cancel_order(self.markets[self.orders[coi].symbol], coi)
        now = self.clock.now_ns()
        return TxResult(True, "paper-cancel-all", "", now, now)

    async def schedule_cancel_all(self, at_ms: int) -> TxResult:
        now = self.clock.now_ns()
        return TxResult(True, "paper-schedule", "", now, now)

    async def update_leverage(self, m: LighterMarket, leverage: float) -> TxResult:
        now = self.clock.now_ns()
        return TxResult(True, "paper-leverage", "", now, now)

    def poll(self) -> None:
        """Activate orders whose latency has elapsed; call after each event."""
        now = self.clock.now_ns()
        while self._pending and self._pending[0][0] <= now:
            _, coi = heapq.heappop(self._pending)
            o = self.orders[coi]
            if o.canceled:
                continue
            book = self.state.lit[o.symbol].book
            if o.kind == "ioc":
                self._ioc(o, book, now)
            else:
                top = book.best_bid() if o.side == BUY else book.best_ask()
                crosses = top is not None and ((o.side == BUY and book.best_ask() and book.best_ask()[0] <= o.price)
                                               or (o.side == SELL and book.best_bid() and book.best_bid()[0] >= o.price))
                if crosses:
                    o.canceled = True  # post-only would have been rejected
                    self.on_cancel(coi, "post_only_would_cross")
                    continue
                levels = book.bids if o.side == BUY else book.asks
                o.queue = levels.get(o.price, 0.0)
        for o in self.orders.values():
            if o.kind != "ioc" and not o.canceled and o.filled < o.qty and now > o.expires_at:
                o.canceled = True
                self.on_cancel(o.coi, "expired")

    def _ioc(self, o: _PaperOrder, book, now: int) -> None:
        levels = book.asks if o.side == BUY else book.bids
        it = iter(levels) if o.side == BUY else reversed(levels)
        remaining, cost, got = o.qty, 0.0, 0.0
        for p in it:
            if (o.side == BUY and p > o.price) or (o.side == SELL and p < o.price):
                break
            take = min(levels[p], remaining)
            cost += take * p
            got += take
            remaining -= take
            if remaining <= 1e-12:
                break
        o.canceled = True  # IOC remainder never rests
        if got > 0:
            o.filled = got
            vwap = cost / got
            self.on_fill(FillReport(o.coi, now, vwap, got, o.side, got * vwap * o.fee_taker, "taker",
                                    f"paper-{o.coi}", True))
        else:
            self.on_cancel(o.coi, "ioc_no_liquidity")

    def on_trade(self, tr: Trade) -> None:
        """Feed public Lighter trades to fill resting paper orders."""
        for o in self.orders.values():
            if o.kind == "ioc" or o.canceled or o.symbol != tr.symbol or o.filled >= o.qty or math.isnan(o.queue):
                continue
            if tr.ts_local_ns < o.active_at or tr.taker_side != -o.side:
                continue
            through = tr.price < o.price if o.side == BUY else tr.price > o.price
            at = abs(tr.price - o.price) <= o.price * 1e-12
            if at:
                o.queue -= tr.size
            if through or (at and o.queue <= -o.qty):
                o.filled = o.qty
                self.on_fill(FillReport(o.coi, tr.ts_local_ns, o.price, o.qty, o.side, o.qty * o.price * o.fee_maker,
                                        "maker", f"paper-{o.coi}", True))
