"""Live WebSocket feeds for Lighter and the reference venue.

Each message is stamped on the local timeline the moment it is read, handed
raw to the recorder (so research replays byte-identical input) and parsed by
the same parser classes used in replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from apexmind.core.clock import Clock
from apexmind.core.events import FeedStatus
from apexmind.venues.binance.parse import VENUE as BINANCE, BinanceParser
from apexmind.venues.binance.rest import BinanceRest
from apexmind.venues.lighter.parse import LighterParser

log = logging.getLogger(__name__)

RawSink = Callable[[str, str, int, str], None]  # venue, conn_id, ts_local_ns, raw
EventSink = Callable[[list], None]


class WsRunner:
    def __init__(self, name: str, url: str, clock: Clock, on_open: Callable[[Any], Awaitable[None]],
                 on_message: Callable[[Any, str, int], Awaitable[None]], on_status: Callable[[str, str], None]):
        self.name, self.url, self.clock = name, url, clock
        self.on_open, self.on_message, self.on_status = on_open, on_message, on_status
        self.ws = None
        self.connected = asyncio.Event()

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(self.url, max_size=None, open_timeout=15, ping_interval=20,
                                              ping_timeout=20, close_timeout=5) as ws:
                    self.ws = ws
                    self.connected.set()
                    self.on_status("connected", self.url)
                    backoff = 1.0
                    await self.on_open(ws)
                    async for message in ws:
                        t = self.clock.now_ns()
                        if isinstance(message, bytes):
                            message = message.decode()
                        await self.on_message(ws, message, t)
                        if stop.is_set():
                            break
                reason = "closed by server"
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError, ValueError) as e:
                reason = repr(e)
            finally:
                self.ws = None
                was_connected = self.connected.is_set()
                self.connected.clear()
            if was_connected or not stop.is_set():
                self.on_status("disconnected", reason)
            if not stop.is_set():
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60.0)


class LighterFeed:
    """Subscribes to order_book / trade / market_stats per market."""

    def __init__(self, ws_url: str, market_ids: list[int], parser: LighterParser, clock: Clock,
                 on_raw: RawSink, on_events: EventSink, max_subs: int = 50, account_index: int = -1):
        self.parser, self.clock = parser, clock
        self.on_raw, self.on_events = on_raw, on_events
        channels = []
        for m in market_ids:
            channels += [f"order_book/{m}", f"trade/{m}", f"market_stats/{m}"]
        if account_index >= 0:
            channels.append(f"account_all/{account_index}")
        groups = [channels[i : i + max_subs] for i in range(0, len(channels), max_subs)]
        self.runners = []
        for gi, group in enumerate(groups):
            name = f"lighter-{gi}"
            self.runners.append(WsRunner(name, ws_url, clock, self._opener(group), self._handler(name),
                                         self._status(name)))
        self._channel_runner = {c: r for r, g in zip(self.runners, groups) for c in g}

    def _opener(self, group: list[str]):
        async def on_open(ws):
            for ch in group:
                await ws.send(json.dumps({"type": "subscribe", "channel": ch}))
        return on_open

    def _handler(self, name: str):
        async def on_message(ws, raw: str, t: int):
            self.on_raw("lighter", name, t, raw)
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                await ws.send('{"type":"pong"}')
                return
            events = self.parser.parse(msg, t)
            if events:
                self.on_events(events)
                for ev in events:
                    if isinstance(ev, FeedStatus) and ev.status == "gap":
                        await self._resubscribe_book(ws, ev.symbol)
        return on_message

    def _status(self, name: str):
        def on_status(status: str, detail: str):
            log.info("%s %s %s", name, status, detail)
            t = self.clock.now_ns()
            if status != "connected":
                self.parser.reset_all()
            self.on_raw("lighter", name, t, json.dumps({"type": f"_feed/{status}", "detail": detail}))
            self.on_events([FeedStatus("lighter", "*", t, status, detail)])
        return on_status

    async def _resubscribe_book(self, ws, symbol: str, only_if_owned: WsRunner | None = None) -> None:
        for mid, sym in self.parser.symbols.items():
            if sym == symbol:
                ch = f"order_book/{mid}"
                if only_if_owned is not None and self._channel_runner.get(ch) is not only_if_owned:
                    continue
                await ws.send(json.dumps({"type": "unsubscribe", "channel": ch}))
                await ws.send(json.dumps({"type": "subscribe", "channel": ch}))

    async def _resync_loop(self, stop: asyncio.Event, minutes: float) -> None:
        """Periodically resubscribe books so every recording hour contains a
        recent snapshot (bounds replay lookback, heals silent drift)."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=minutes * 60)
                return
            except asyncio.TimeoutError:
                pass
            for runner in self.runners:
                if runner.ws is None:
                    continue
                for sym in self.parser.symbols.values():
                    try:
                        await self._resubscribe_book(runner.ws, sym, only_if_owned=runner)
                    except Exception as e:  # connection dropped mid-way; runner reconnects
                        log.warning("resubscribe %s failed: %s", sym, e)

    async def run(self, stop: asyncio.Event, resync_minutes: float = 30.0) -> None:
        await asyncio.gather(self._resync_loop(stop, resync_minutes), *(r.run(stop) for r in self.runners))


class BinanceFeed:
    """bookTicker + diff depth + aggTrade + markPrice, with REST snapshot sync."""

    def __init__(self, ws_url: str, venue_symbols: list[str], parser: BinanceParser, rest: BinanceRest,
                 clock: Clock, on_raw: RawSink, on_events: EventSink, depth_ms: int = 100):
        self.parser, self.rest, self.clock = parser, rest, clock
        self.on_raw, self.on_events = on_raw, on_events
        self.venue_symbols = [s.upper() for s in venue_symbols]
        streams = []
        for s in self.venue_symbols:
            ls = s.lower()
            streams += [f"{ls}@bookTicker", f"{ls}@depth@{depth_ms}ms", f"{ls}@aggTrade", f"{ls}@markPrice@1s"]
        # Binance allows up to 1024 streams per connection; stay far below.
        self.runners = []
        for gi in range(0, len(streams), 200):
            url = f"{ws_url}?streams={'/'.join(streams[gi:gi + 200])}"
            name = f"binance-{gi // 200}"
            self.runners.append(WsRunner(name, url, clock, self._noop_open, self._handler(name), self._status(name)))

    async def _noop_open(self, ws) -> None:
        return None

    def _handler(self, name: str):
        async def on_message(ws, raw: str, t: int):
            self.on_raw(BINANCE, name, t, raw)
            events = self.parser.parse(raw, t)
            if events:
                self.on_events(events)
        return on_message

    def _status(self, name: str):
        def on_status(status: str, detail: str):
            log.info("%s %s %s", name, status, detail)
            t = self.clock.now_ns()
            self.on_raw(BINANCE, name, t, json.dumps({"stream": f"_feed/{status}", "data": {"detail": detail}}))
            if status != "connected":
                # Depth continuity is lost across reconnects.
                for s in self.venue_symbols:
                    self.parser.reset(s)
            self.on_events([FeedStatus(BINANCE, "*", t, status, detail)])
        return on_status

    async def _snapshot_loop(self, stop: asyncio.Event, resync_minutes: float) -> None:
        last_full = self.clock.now_ns()
        while not stop.is_set():
            if (self.clock.now_ns() - last_full) > resync_minutes * 60e9:
                # Forced periodic re-snapshot (see LighterFeed._resync_loop).
                last_full = self.clock.now_ns()
                for s in self.venue_symbols:
                    self.parser.reset(s)
            if any(r.connected.is_set() for r in self.runners):
                for s in self.venue_symbols:
                    if self.parser.needs_snapshot(s):
                        # Give the stream a moment to start buffering first.
                        await asyncio.sleep(0.5)
                        try:
                            res = await self.rest.depth(s)
                        except Exception as e:  # network errors: retry next round
                            log.warning("depth snapshot %s failed: %s", s, e)
                            continue
                        raw = json.dumps({"stream": f"rest:depth:{s}", "data": res.data}, separators=(",", ":"))
                        t = self.clock.now_ns()
                        self.on_raw(BINANCE, "rest", t, raw)
                        events = self.parser.parse(raw, t)
                        if events:
                            self.on_events(events)
            await asyncio.sleep(1.0)

    async def run(self, stop: asyncio.Event, resync_minutes: float = 30.0) -> None:
        await asyncio.gather(self._snapshot_loop(stop, resync_minutes), *(r.run(stop) for r in self.runners))
