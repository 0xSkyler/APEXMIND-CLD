"""Synchronized market-data collector (long-running service)."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import socket
from pathlib import Path

from apexmind import __version__
from apexmind.collector.quality import QualityMonitor
from apexmind.collector.recorder import RawRecorder
from apexmind.config import Config
from apexmind.core.clock import Clock, FeedDelayMonitor, OffsetEstimator, system_clock_status
from apexmind.core.events import FeedStatus
from apexmind.util import atomic_write_json, config_hash, git_revision
from apexmind.venues.binance.parse import VENUE as BINANCE, BinanceParser
from apexmind.venues.binance.rest import BinanceRest
from apexmind.venues.feeds import BinanceFeed, LighterFeed
from apexmind.venues.lighter.markets import parse_order_book_details
from apexmind.venues.lighter.parse import LighterParser
from apexmind.venues.lighter.rest import LighterRest
from apexmind.venues.mapping import build_instruments

log = logging.getLogger(__name__)


class Collector:
    def __init__(self, cfg: Config, fee_unit: str = "percent") -> None:
        if cfg.reference.venue != BINANCE:
            raise ValueError(f"unsupported reference venue {cfg.reference.venue}")
        self.cfg = cfg
        self.fee_unit = fee_unit
        self.clock = Clock()
        self.recorder = RawRecorder(cfg.collector.data_dir, self.clock)
        self.lrest = LighterRest(cfg.lighter, self.clock)
        self.brest = BinanceRest(cfg.reference, self.clock)
        self.quality = QualityMonitor()
        self.delays = FeedDelayMonitor()
        self.offsets = {"lighter": OffsetEstimator(), BINANCE: OffsetEstimator()}
        self.stop = asyncio.Event()
        self.instruments = []

    def _on_raw(self, venue: str, conn: str, t: int, raw: str) -> None:
        self.recorder.write(venue, conn, t, raw)

    def _on_events(self, events: list) -> None:
        for ev in events:
            self.quality.observe(ev)
            if not isinstance(ev, FeedStatus):
                off = self.offsets.get(ev.venue)
                self.delays.observe(f"{ev.venue}:{ev.symbol}", ev.ts_local_ns, ev.ts_exch_ns,
                                    off.offset_ns if off and off.best() else 0.0)

    async def bootstrap(self) -> tuple[list[int], list[str]]:
        self.recorder.write_meta("session", {
            "version": __version__, "git": git_revision(), "host": socket.gethostname(),
            "python": platform.python_version(), "config_hash": config_hash(self.cfg),
            "config": self.cfg.to_dict(), "system_clock": system_clock_status(),
        })
        details = await self.lrest.order_book_details()
        self.recorder.write_meta("lighter_markets", details.data)
        markets = parse_order_book_details(details.data, self.fee_unit)
        prices = await self.brest.prices()
        self.instruments = build_instruments(markets, prices, self.cfg.instruments, BINANCE)
        self.recorder.write_meta("instruments", [i.to_dict() for i in self.instruments])
        elig = [i for i in self.instruments if i.eligible]
        if not elig:
            raise RuntimeError("no eligible instruments; check mapping report in the meta stream")
        for i in self.instruments:
            log.info("instrument %s -> %s x%s eligible=%s (%s)", i.symbol, i.ref_symbol, i.price_multiplier,
                     i.eligible, i.reason)
        self.lparser = LighterParser({i.market_id: i.symbol for i in elig}, self.cfg.lighter.account_index)
        self.bparser = BinanceParser({i.ref_symbol: i.symbol for i in elig},
                                     multipliers={i.ref_symbol: i.price_multiplier for i in elig})
        return [i.market_id for i in elig], [i.ref_symbol for i in elig]

    async def _clock_loop(self) -> None:
        while not self.stop.is_set():
            for venue, probe in ((BINANCE, self.brest.server_time_probe), ("lighter", self.lrest.server_time_probe)):
                try:
                    p = await probe()
                except Exception as e:
                    log.warning("clock probe %s failed: %s", venue, e)
                    continue
                if p is None:
                    continue
                s = self.offsets[venue].add_probe(*p)
                self.recorder.write_meta("clock_probe", {
                    "venue": venue, "t_send": p[0], "t_server": p[1], "t_recv": p[2],
                    "offset_ms": s.offset_ns / 1e6, "rtt_ms": s.rtt_ns / 1e6})
            self.recorder.write_meta("system_clock", system_clock_status())
            await _sleep_or_stop(self.stop, self.cfg.collector.clock_probe_interval_s)

    async def _metadata_loop(self) -> None:
        while not self.stop.is_set():
            await _sleep_or_stop(self.stop, 3600)
            try:
                details = await self.lrest.order_book_details()
                self.recorder.write_meta("lighter_markets", details.data)
                fr = await self.lrest.funding_rates()
                self.recorder.write_meta("lighter_funding_rates", fr.data)
            except Exception as e:
                log.warning("metadata refresh failed: %s", e)

    async def _housekeeping_loop(self) -> None:
        health = Path(self.cfg.collector.health_path)
        while not self.stop.is_set():
            self.recorder.flush()
            now = self.clock.now_ns()
            atomic_write_json(health, {
                "ts": now, "lines": self.recorder.lines, "bytes": self.recorder.bytes,
                "wall_drift_ms": self.clock.wall_drift_ns() / 1e6,
                "streams": self.delays.snapshot(now), "quality": self.quality.report(),
                "offsets_ms": {v: {"offset": o.offset_ns / 1e6, "uncertainty": o.uncertainty_ns / 1e6}
                               for v, o in self.offsets.items() if o.best()},
            })
            await _sleep_or_stop(self.stop, self.cfg.collector.flush_interval_s)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        market_ids, ref_symbols = await self.bootstrap()
        lfeed = LighterFeed(self.cfg.lighter.ws_url, market_ids, self.lparser, self.clock, self._on_raw,
                            self._on_events, self.cfg.lighter.ws_max_subscriptions_per_connection)
        bfeed = BinanceFeed(self.cfg.reference.ws_url, ref_symbols, self.bparser, self.brest, self.clock,
                            self._on_raw, self._on_events, self.cfg.reference.depth_stream_ms)
        resync = self.cfg.collector.snapshot_resync_minutes
        tasks = [asyncio.create_task(c) for c in (
            lfeed.run(self.stop, resync), bfeed.run(self.stop, resync), self._clock_loop(), self._metadata_loop(),
            self._housekeeping_loop())]
        await self.stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.recorder.close()
        await self.lrest.close()
        await self.brest.close()


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def run_collector(cfg: Config) -> None:
    os.makedirs(cfg.collector.data_dir, exist_ok=True)
    asyncio.run(Collector(cfg).run())
